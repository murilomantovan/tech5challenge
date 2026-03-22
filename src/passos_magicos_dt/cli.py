from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import build_analytics
from .config import ROOT_DIR
from .data import prepare_datasets
from .materials import (
    write_data_preparation_notebook,
    write_painel_analitico_notebook,
    write_painel_payload,
    write_qna_report,
    write_runner_notebook,
)
from .modeling import save_training_artifacts, train_temporal_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline analitico e preditivo da ferramenta Passos Magicos Analytics"
    )
    subcomandos = parser.add_subparsers(dest="comando", required=True)

    for nome_subcomando in ("analyze", "train", "build-all"):
        subcomando = subcomandos.add_parser(nome_subcomando)
        subcomando.add_argument("--excel", help="Caminho opcional para o Excel oficial")

    return parser.parse_args()


def run_training(caminho_excel: str | None = None):
    pacote_dados = prepare_datasets(
        excel_path=Path(caminho_excel) if caminho_excel else None,
        root=ROOT_DIR,
    )
    artefatos_treinamento = train_temporal_model(pacote_dados.base_pares)
    save_training_artifacts(artefatos_treinamento)
    print(f"Modelo final: {artefatos_treinamento.model_name}")
    print(f"Limiar final: {artefatos_treinamento.threshold:.2f}")
    return pacote_dados, artefatos_treinamento


def run_analysis(caminho_excel: str | None = None):
    pacote_dados, artefatos_treinamento = run_training(caminho_excel)
    artefatos_analiticos = build_analytics(
        pacote_dados.base_analitica,
        pacote_dados.base_pares,
        artefatos_treinamento,
    )

    write_qna_report(artefatos_analiticos)
    write_painel_payload(artefatos_analiticos)
    write_data_preparation_notebook()
    write_painel_analitico_notebook()
    write_runner_notebook()

    print(json.dumps(artefatos_analiticos.resumo_painel, indent=2, ensure_ascii=False))
    return pacote_dados, artefatos_treinamento, artefatos_analiticos


def run_build_all(caminho_excel: str | None = None) -> None:
    run_analysis(caminho_excel)


def main() -> None:
    argumentos = parse_args()
    if argumentos.comando == "train":
        run_training(argumentos.excel)
        return
    if argumentos.comando == "analyze":
        run_analysis(argumentos.excel)
        return
    if argumentos.comando == "build-all":
        run_build_all(argumentos.excel)
        return
    raise ValueError(f"Comando nao suportado: {argumentos.comando}")
