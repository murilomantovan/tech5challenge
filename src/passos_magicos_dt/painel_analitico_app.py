from __future__ import annotations

import json

import pandas as pd

from .config import ANALYTICS_DIR, FEATURE_IMPORTANCE_NAME, MODEL_DIR, PAINEL_PAYLOAD_NAME
from .runtime import get_storyboard_source_path


def load_painel_payload() -> dict[str, object] | None:
    caminho = ANALYTICS_DIR / PAINEL_PAYLOAD_NAME
    if not caminho.exists():
        return None
    return json.loads(caminho.read_text(encoding="utf-8"))


def load_feature_importance() -> pd.DataFrame | None:
    caminho = MODEL_DIR / FEATURE_IMPORTANCE_NAME
    if not caminho.exists():
        return None
    return pd.read_csv(caminho)


def load_storyboard() -> list[dict[str, object]] | None:
    caminho = get_storyboard_source_path()
    if caminho is None:
        return None
    if not caminho.exists():
        return None
    return json.loads(caminho.read_text(encoding="utf-8"))
