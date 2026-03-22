from __future__ import annotations

import shutil
import threading
from pathlib import Path

from .analysis import build_analytics
from .config import (
    ANALYTICS_DIR,
    ASSETS_DIR,
    FEATURE_IMPORTANCE_NAME,
    MODEL_BUNDLE_NAME,
    MODEL_CONFIG_NAME,
    MODEL_DIR,
    MODEL_METRICS_NAME,
    PAINEL_PAYLOAD_NAME,
    ROOT_DIR,
    STORYBOARD_RUNTIME_NAME,
)
from .data import prepare_datasets
from .materials import write_painel_payload, write_qna_report
from .modeling import save_training_artifacts, train_temporal_model


_RUNTIME_LOCK = threading.Lock()

RUNTIME_REQUIRED_FILES = (
    MODEL_DIR / MODEL_BUNDLE_NAME,
    MODEL_DIR / MODEL_CONFIG_NAME,
    MODEL_DIR / MODEL_METRICS_NAME,
    MODEL_DIR / FEATURE_IMPORTANCE_NAME,
    ANALYTICS_DIR / PAINEL_PAYLOAD_NAME,
    ANALYTICS_DIR / "01_ian_perfil_evolucao.png",
    ANALYTICS_DIR / "09_modelo_holdout.png",
    ANALYTICS_DIR / "12_transicoes_risco.png",
    ANALYTICS_DIR / STORYBOARD_RUNTIME_NAME,
)


def get_package_storyboard_path() -> Path:
    return ASSETS_DIR / STORYBOARD_RUNTIME_NAME


def get_runtime_storyboard_path() -> Path:
    return ANALYTICS_DIR / STORYBOARD_RUNTIME_NAME


def get_storyboard_source_path() -> Path | None:
    candidates = (
        get_runtime_storyboard_path(),
        get_package_storyboard_path(),
        ROOT_DIR / "support" / "video" / STORYBOARD_RUNTIME_NAME,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def runtime_artifacts_ready() -> bool:
    return all(path.exists() for path in RUNTIME_REQUIRED_FILES)


def copy_storyboard_to_runtime(source_path: Path | None = None) -> Path | None:
    source = source_path or get_storyboard_source_path()
    if source is None or not source.exists():
        return None
    target = get_runtime_storyboard_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.read_bytes() != source.read_bytes():
        shutil.copyfile(source, target)
    return target


def ensure_runtime_ready(force: bool = False) -> dict[str, object]:
    with _RUNTIME_LOCK:
        storyboard_path = copy_storyboard_to_runtime()
        if runtime_artifacts_ready() and not force:
            return {
                "built": False,
                "source": "artifacts",
                "storyboard_path": str(storyboard_path) if storyboard_path else None,
            }

        pacote_dados = prepare_datasets(root=ROOT_DIR)
        artefatos_treinamento = train_temporal_model(pacote_dados.base_pares)
        save_training_artifacts(artefatos_treinamento)

        artefatos_analiticos = build_analytics(
            pacote_dados.base_analitica,
            pacote_dados.base_pares,
            artefatos_treinamento,
        )
        write_qna_report(artefatos_analiticos)
        write_painel_payload(artefatos_analiticos)
        storyboard_path = copy_storyboard_to_runtime()

        return {
            "built": True,
            "source": "pipeline",
            "storyboard_path": str(storyboard_path) if storyboard_path else None,
            "base_registros": len(pacote_dados.base_analitica),
            "pares_modelagem": len(pacote_dados.base_pares),
            "modelo_final": artefatos_treinamento.model_name,
        }
