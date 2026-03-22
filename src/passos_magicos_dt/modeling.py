from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    FEATURE_IMPORTANCE_NAME,
    HOLDOUT_YEAR,
    MIN_BASELINE_PRECISION,
    MIN_MODEL_PRECISION,
    MODEL_BUNDLE_NAME,
    MODEL_CONFIG_NAME,
    MODEL_DIR,
    MODEL_METRICS_NAME,
    PREDICTIONS_HOLDOUT_NAME,
    PRODUCTION_YEARS,
    RANDOM_SEED,
    SELECTION_YEAR,
    TARGET_COLUMN,
    THRESHOLD_REPORT_NAME,
)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


FEATURE_COLUMNS = [
    "idade",
    "genero",
    "ano_ingresso",
    "anos_no_programa",
    "ciclo_programa",
    "fase",
    "fase_ideal_num",
    "inde_atual",
    "inde_anterior",
    "delta_inde",
    "cg",
    "cf",
    "ct",
    "n_av",
    "iaa",
    "ieg",
    "ips",
    "ipp",
    "ida",
    "ipv",
    "mat",
    "por",
    "ing",
    "media_academica",
    "media_comportamental",
]


@dataclass
class TrainingArtifacts:
    pipeline: Pipeline
    model_name: str
    threshold: float
    metrics_holdout: pd.DataFrame
    threshold_report: pd.DataFrame
    predictions_holdout: pd.DataFrame
    feature_importance: pd.DataFrame
    model_config: dict[str, object]


def split_modeling_frames(base_pares: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selection = base_pares[base_pares["ano_referencia"] == SELECTION_YEAR].copy()
    holdout = base_pares[base_pares["ano_referencia"] == HOLDOUT_YEAR].copy()
    production = base_pares[base_pares["ano_referencia"].isin(PRODUCTION_YEARS)].copy()
    if selection.empty or holdout.empty:
        raise ValueError("Nao ha pares suficientes para selecao e holdout.")
    return selection, holdout, production


def build_feature_frame(frame: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    columns = feature_columns or FEATURE_COLUMNS
    data = frame.copy()
    for column in columns:
        if column not in data.columns:
            data[column] = np.nan
    return data[columns]


def build_preprocessor(feature_frame: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns = [column for column in feature_frame.columns if pd.api.types.is_numeric_dtype(feature_frame[column])]
    categorical_columns = [column for column in feature_frame.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )
    return preprocessor, numeric_columns, categorical_columns


def build_models(y_train: pd.Series) -> dict[str, object]:
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    pos_weight = float(negatives / positives) if positives else 1.0

    models: dict[str, object] = {
        "regressao_logistica": LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=450,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
    }
    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=500,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            scale_pos_weight=pos_weight,
        )
    return models


def compute_classification_metrics(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "f2": float(fbeta_score(y_true, predicted, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_positive": float(tp),
    }


def choose_threshold(y_true: pd.Series, scores: np.ndarray, min_precision: float) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    best_threshold = 0.5
    best_tuple: tuple[float, float, float] = (-1.0, -1.0, -1.0)

    for threshold in np.linspace(0.15, 0.85, 71):
        predicted = (scores >= threshold).astype(int)
        precision = float(precision_score(y_true, predicted, zero_division=0))
        recall = float(recall_score(y_true, predicted, zero_division=0))
        f2 = float(fbeta_score(y_true, predicted, beta=2, zero_division=0))
        f1 = float(f1_score(y_true, predicted, zero_division=0))
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f2": f2,
            }
        )

        candidate_key = (f2, recall, precision)
        if precision >= min_precision and candidate_key > best_tuple:
            best_tuple = candidate_key
            best_threshold = float(threshold)

    report = pd.DataFrame(rows)
    if best_tuple[0] < 0:
        best_row = report.sort_values(["f2", "recall", "precision"], ascending=[False, False, False]).iloc[0]
        best_threshold = float(best_row["threshold"])
    return best_threshold, report


def business_rule_score(feature_frame: pd.DataFrame) -> np.ndarray:
    score_parts = []
    if "inde_atual" in feature_frame.columns:
        inde = feature_frame["inde_atual"].fillna(feature_frame["inde_atual"].median())
        score_parts.append((10 - inde.clip(0, 10)) / 10)
    if "ida" in feature_frame.columns:
        ida = feature_frame["ida"].fillna(feature_frame["ida"].median())
        score_parts.append((10 - ida.clip(0, 10)) / 10)
    if "ieg" in feature_frame.columns:
        ieg = feature_frame["ieg"].fillna(feature_frame["ieg"].median())
        score_parts.append((10 - ieg.clip(0, 10)) / 10)
    if not score_parts:
        return np.zeros(len(feature_frame))
    stacked = np.vstack(score_parts)
    return stacked.mean(axis=0)


def get_top_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["prep"]
    feature_names = list(preprocessor.get_feature_names_out())

    if hasattr(model, "feature_importances_"):
        raw_values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        raw_values = np.abs(np.asarray(model.coef_).ravel())
    else:
        raw_values = np.zeros(len(feature_names))

    importance = pd.DataFrame({"feature": feature_names, "importance": raw_values})
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def train_temporal_model(base_pares: pd.DataFrame) -> TrainingArtifacts:
    selection, holdout, production = split_modeling_frames(base_pares)
    active_feature_columns = [
        column for column in FEATURE_COLUMNS if column in selection.columns and selection[column].notna().any()
    ]
    x_selection = build_feature_frame(selection, active_feature_columns)
    y_selection = selection[TARGET_COLUMN].astype(int)
    x_holdout = build_feature_frame(holdout, active_feature_columns)
    y_holdout = holdout[TARGET_COLUMN].astype(int)
    x_production = build_feature_frame(production, active_feature_columns)
    y_production = production[TARGET_COLUMN].astype(int)

    preprocessor, numeric_columns, categorical_columns = build_preprocessor(x_selection)
    candidate_models = build_models(y_selection)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED)

    selection_rows: list[dict[str, float | str]] = []
    holdout_rows: list[dict[str, float | str]] = []
    trained_on_selection: dict[str, Pipeline] = {}
    holdout_probabilities: dict[str, np.ndarray] = {}
    threshold_report_frames: list[pd.DataFrame] = []

    baseline_selection_scores = business_rule_score(x_selection)
    baseline_threshold, baseline_threshold_report = choose_threshold(
        y_true=y_selection,
        scores=baseline_selection_scores,
        min_precision=MIN_BASELINE_PRECISION,
    )
    baseline_selection_metrics = compute_classification_metrics(y_selection, baseline_selection_scores, baseline_threshold)
    baseline_holdout_scores = business_rule_score(x_holdout)
    baseline_holdout_metrics = compute_classification_metrics(y_holdout, baseline_holdout_scores, baseline_threshold)
    selection_rows.append({"model": "baseline_regra_negocio", "threshold": baseline_threshold, **baseline_selection_metrics})
    holdout_rows.append({"model": "baseline_regra_negocio", "threshold": baseline_threshold, **baseline_holdout_metrics})
    baseline_threshold_report["model"] = "baseline_regra_negocio"
    threshold_report_frames.append(baseline_threshold_report)

    for model_name, estimator in candidate_models.items():
        pipeline = Pipeline(
            steps=[
                ("prep", clone(preprocessor)),
                ("model", estimator),
            ]
        )
        oof_scores = cross_val_predict(
            estimator=pipeline,
            X=x_selection,
            y=y_selection,
            cv=cv,
            method="predict_proba",
            n_jobs=None,
        )[:, 1]
        threshold, threshold_report = choose_threshold(
            y_true=y_selection,
            scores=oof_scores,
            min_precision=MIN_MODEL_PRECISION,
        )
        threshold_report["model"] = model_name
        threshold_report_frames.append(threshold_report)

        selection_metrics = compute_classification_metrics(y_selection, oof_scores, threshold)
        selection_rows.append({"model": model_name, "threshold": threshold, **selection_metrics})

        pipeline.fit(x_selection, y_selection)
        holdout_scores = pipeline.predict_proba(x_holdout)[:, 1]
        holdout_metrics = compute_classification_metrics(y_holdout, holdout_scores, threshold)
        holdout_rows.append({"model": model_name, "threshold": threshold, **holdout_metrics})
        trained_on_selection[model_name] = pipeline
        holdout_probabilities[model_name] = holdout_scores

    selection_metrics_frame = pd.DataFrame(selection_rows)
    selection_ranked = selection_metrics_frame[selection_metrics_frame["model"] != "baseline_regra_negocio"].copy()
    selection_ranked = selection_ranked.sort_values(["f2", "pr_auc", "recall"], ascending=[False, False, False])
    best_model_name = str(selection_ranked.iloc[0]["model"])
    best_threshold = float(selection_ranked.iloc[0]["threshold"])

    holdout_metrics_frame = pd.DataFrame(holdout_rows).sort_values(["f2", "pr_auc", "recall"], ascending=[False, False, False])
    best_selection_pipeline = trained_on_selection[best_model_name]

    production_pipeline = Pipeline(
        steps=[
            ("prep", clone(preprocessor)),
            ("model", clone(best_selection_pipeline.named_steps["model"])),
        ]
    )
    production_pipeline.fit(x_production, y_production)

    best_holdout_scores = holdout_probabilities[best_model_name]
    predictions_holdout = holdout[
        ["ra", "nome_exibicao", "ano_referencia", "ano_alvo", "ciclo_programa", "genero", "idade"]
    ].copy()
    predictions_holdout["probabilidade_risco"] = best_holdout_scores
    predictions_holdout["predicao_modelo"] = (best_holdout_scores >= best_threshold).astype(int)
    predictions_holdout["risco_real"] = y_holdout.values

    threshold_report = pd.concat(threshold_report_frames, ignore_index=True)
    feature_importance = get_top_feature_importance(production_pipeline)

    config = {
        "model_name": best_model_name,
        "threshold": best_threshold,
        "target_column": TARGET_COLUMN,
        "selection_year": SELECTION_YEAR,
        "holdout_year": HOLDOUT_YEAR,
        "feature_columns": active_feature_columns,
        "all_feature_columns": FEATURE_COLUMNS,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "example_profiles": [
            "Engajamento em queda",
            "Desempenho consistente",
            "Risco psicossocial",
        ],
    }

    return TrainingArtifacts(
        pipeline=production_pipeline,
        model_name=best_model_name,
        threshold=best_threshold,
        metrics_holdout=holdout_metrics_frame,
        threshold_report=threshold_report,
        predictions_holdout=predictions_holdout,
        feature_importance=feature_importance,
        model_config=config,
    )


def save_training_artifacts(artifacts: TrainingArtifacts) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.pipeline, MODEL_DIR / MODEL_BUNDLE_NAME)
    with (MODEL_DIR / MODEL_CONFIG_NAME).open("w", encoding="utf-8") as stream:
        json.dump(artifacts.model_config | {"selected_model": artifacts.model_name, "selected_threshold": artifacts.threshold}, stream, indent=2, ensure_ascii=False)
    artifacts.metrics_holdout.to_csv(MODEL_DIR / MODEL_METRICS_NAME, index=False, encoding="utf-8")
    artifacts.threshold_report.to_csv(MODEL_DIR / THRESHOLD_REPORT_NAME, index=False, encoding="utf-8")
    artifacts.predictions_holdout.to_csv(MODEL_DIR / PREDICTIONS_HOLDOUT_NAME, index=False, encoding="utf-8")
    artifacts.feature_importance.to_csv(MODEL_DIR / FEATURE_IMPORTANCE_NAME, index=False, encoding="utf-8")


def load_model_bundle(model_dir: Path | None = None) -> tuple[Pipeline, dict[str, object]]:
    bundle_dir = model_dir or MODEL_DIR
    pipeline = joblib.load(bundle_dir / MODEL_BUNDLE_NAME)
    with (bundle_dir / MODEL_CONFIG_NAME).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    return pipeline, config
