from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_DIR / "assets"
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ANALYTICS_DIR = ARTIFACTS_DIR / "analytics"
MODEL_DIR = ARTIFACTS_DIR / "model"
LOGS_DIR = ARTIFACTS_DIR / "logs"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
REFERENCES_DIR = ROOT_DIR / "references"
TESTS_DIR = ROOT_DIR / "tests"
LEGACY_DIR = ROOT_DIR / "archive" / "legacy"
LEGACY_DATA_DIR = LEGACY_DIR / "data"

DEFAULT_EXCEL_NAME = "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
DEFAULT_CSV_NAME = "PEDE_PASSOS_DATASET_FIAP.csv"
DEFAULT_SHEETS = ("PEDE2022", "PEDE2023", "PEDE2024")
TARGET_COLUMN = "risco_proximo_ano"
RANDOM_SEED = 42
SELECTION_YEAR = 2022
HOLDOUT_YEAR = 2023
PRODUCTION_YEARS = (2022, 2023)
APP_EXAMPLE_PROFILES = (
    "Engajamento em queda",
    "Desempenho consistente",
    "Risco psicossocial",
)

MODEL_BUNDLE_NAME = "model_pipeline.joblib"
MODEL_CONFIG_NAME = "model_config.json"
MODEL_METRICS_NAME = "metrics_holdout.csv"
THRESHOLD_REPORT_NAME = "threshold_report.csv"
PREDICTIONS_HOLDOUT_NAME = "predictions_holdout.csv"
FEATURE_IMPORTANCE_NAME = "feature_importance.csv"

ANALYTICAL_BASE_NAME = "base_analitica.parquet"
PAIR_DATASET_NAME = "base_modelagem_proximo_ano.parquet"
QNA_REPORT_NAME = "resumo_analitico.md"
PAINEL_PAYLOAD_NAME = "painel_analitico.json"
STORYBOARD_RUNTIME_NAME = "storyboard.json"
DATA_NOTEBOOK_NAME = "01_tratamento_e_bases_modelo.ipynb"
PAINEL_NOTEBOOK_NAME = "02_painel_analitico.ipynb"
RUNNER_NOTEBOOK_NAME = "03_executar_entrega_e_streamlit.ipynb"

MIN_BASELINE_PRECISION = 0.40
MIN_MODEL_PRECISION = 0.45

MODELS = ("regressao_logistica", "random_forest", "xgboost")
