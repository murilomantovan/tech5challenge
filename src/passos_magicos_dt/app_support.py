from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

from .modeling import FEATURE_COLUMNS


EXAMPLE_PROFILES = {
    "Engajamento em queda": {
        "idade": 13,
        "genero": "Masculino",
        "ano_ingresso": 2022,
        "ciclo_programa": "Quartzo",
        "fase": "Fase 3",
        "fase_ideal_num": 3.0,
        "inde_atual": 5.8,
        "inde_anterior": 6.6,
        "cg": 420,
        "cf": 80,
        "ct": 10,
        "n_av": 3,
        "iaa": 6.2,
        "ieg": 4.8,
        "ips": 6.9,
        "ipp": 6.4,
        "ida": 5.4,
        "ipv": 4.7,
        "mat": 5.2,
        "por": 5.5,
        "ing": 5.0,
    },
    "Desempenho consistente": {
        "idade": 12,
        "genero": "Feminino",
        "ano_ingresso": 2021,
        "ciclo_programa": "Ametista",
        "fase": "Fase 2",
        "fase_ideal_num": 2.0,
        "inde_atual": 8.3,
        "inde_anterior": 7.9,
        "cg": 290,
        "cf": 52,
        "ct": 6,
        "n_av": 4,
        "iaa": 8.1,
        "ieg": 8.4,
        "ips": 7.4,
        "ipp": 7.8,
        "ida": 8.0,
        "ipv": 8.3,
        "mat": 8.2,
        "por": 7.9,
        "ing": 7.8,
    },
    "Risco psicossocial": {
        "idade": 15,
        "genero": "Feminino",
        "ano_ingresso": 2023,
        "ciclo_programa": "Agata",
        "fase": "Fase 5",
        "fase_ideal_num": 5.0,
        "inde_atual": 6.4,
        "inde_anterior": 6.9,
        "cg": 510,
        "cf": 92,
        "ct": 12,
        "n_av": 2,
        "iaa": 5.8,
        "ieg": 6.1,
        "ips": 4.3,
        "ipp": 5.9,
        "ida": 6.0,
        "ipv": 5.4,
        "mat": 5.9,
        "por": 6.1,
        "ing": 6.0,
    },
}


@dataclass
class PredictionExplanation:
    chart_data: pd.DataFrame
    method: str


def coerce_optional_number(value: object) -> float:
    if value in ("", None):
        return np.nan
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    texto = str(value).strip().replace(",", ".")
    if not texto:
        return np.nan
    return float(texto)


def build_input_dataframe(raw_input: dict[str, object], reference_year: int = 2024) -> pd.DataFrame:
    quadro_entrada = pd.DataFrame([{coluna: raw_input.get(coluna, np.nan) for coluna in FEATURE_COLUMNS}])
    colunas_numericas = (
        "idade",
        "ano_ingresso",
        "fase_ideal_num",
        "inde_atual",
        "inde_anterior",
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
    )
    for coluna in colunas_numericas:
        quadro_entrada[coluna] = quadro_entrada[coluna].map(coerce_optional_number)
    quadro_entrada["media_academica"] = quadro_entrada[["mat", "por", "ing"]].mean(axis=1)
    quadro_entrada["media_comportamental"] = quadro_entrada[["iaa", "ieg", "ips", "ipp"]].mean(axis=1)
    quadro_entrada["delta_inde"] = quadro_entrada["inde_atual"] - quadro_entrada["inde_anterior"]
    quadro_entrada["anos_no_programa"] = np.where(
        quadro_entrada["ano_ingresso"].notna(),
        reference_year - quadro_entrada["ano_ingresso"],
        np.nan,
    )
    for coluna in FEATURE_COLUMNS:
        if coluna not in quadro_entrada.columns:
            quadro_entrada[coluna] = np.nan
    return quadro_entrada[FEATURE_COLUMNS]


def explain_prediction(pipeline, input_frame: pd.DataFrame) -> PredictionExplanation:
    preprocessador = pipeline.named_steps["prep"]
    modelo = pipeline.named_steps["model"]
    dados_transformados = preprocessador.transform(input_frame)
    if hasattr(dados_transformados, "toarray"):
        dados_transformados = dados_transformados.toarray()
    nomes_variaveis = list(preprocessador.get_feature_names_out())

    if SHAP_AVAILABLE and hasattr(modelo, "feature_importances_"):
        explicador = shap.TreeExplainer(modelo)
        valores_shap = explicador.shap_values(dados_transformados)
        if isinstance(valores_shap, list):
            contribuicoes = np.asarray(valores_shap[1][0])
        else:
            arranjo = np.asarray(valores_shap)
            contribuicoes = arranjo[0, :, 1] if arranjo.ndim == 3 else arranjo[0]
        metodo = "SHAP"
    elif hasattr(modelo, "coef_"):
        contribuicoes = dados_transformados[0] * np.asarray(modelo.coef_).ravel()
        metodo = "Coeficientes"
    else:
        contribuicoes = np.zeros(len(nomes_variaveis))
        metodo = "Importancia aproximada"

    quadro_contribuicoes = pd.DataFrame({"feature": nomes_variaveis, "impact": contribuicoes})
    quadro_contribuicoes["abs_impact"] = quadro_contribuicoes["impact"].abs()
    quadro_contribuicoes = quadro_contribuicoes.sort_values("abs_impact", ascending=False).head(8)
    return PredictionExplanation(chart_data=quadro_contribuicoes, method=metodo)
