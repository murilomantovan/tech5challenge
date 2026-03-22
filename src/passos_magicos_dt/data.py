from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ANALYTICAL_BASE_NAME,
    DEFAULT_CSV_NAME,
    DEFAULT_EXCEL_NAME,
    DEFAULT_SHEETS,
    INTERIM_DIR,
    LEGACY_DATA_DIR,
    PROCESSED_DIR,
    RAW_DIR,
)


NUMERIC_COLUMNS = (
    "ano_nasc",
    "idade",
    "ano_ingresso",
    "inde_22",
    "inde_23",
    "inde_24",
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
    "ian",
    "mat",
    "por",
    "ing",
    "fase_ideal_num",
    "media_academica",
    "media_comportamental",
    "delta_inde",
    "anos_no_programa",
)

CANONICAL_ALIASES = {
    "ra": ("ra",),
    "nome_exibicao": ("nome_anonimizado", "nome"),
    "fase": ("fase",),
    "turma": ("turma",),
    "ano_nasc": ("ano_nasc",),
    "data_nasc": ("data_de_nasc",),
    "idade": ("idade", "idade_22"),
    "genero": ("genero",),
    "ano_ingresso": ("ano_ingresso",),
    "instituicao": ("instituicao_de_ensino", "escola"),
    "pedra_20": ("pedra_20",),
    "pedra_21": ("pedra_21",),
    "pedra_22": ("pedra_22",),
    "pedra_23": ("pedra_2023", "pedra_23"),
    "pedra_24": ("pedra_2024",),
    "inde_22": ("inde_22",),
    "inde_23": ("inde_2023", "inde_23"),
    "inde_24": ("inde_2024",),
    "cg": ("cg",),
    "cf": ("cf",),
    "ct": ("ct",),
    "n_av": ("no_av", "n_av", "qtd_aval_2022"),
    "iaa": ("iaa",),
    "ieg": ("ieg",),
    "ips": ("ips",),
    "ipp": ("ipp",),
    "ida": ("ida",),
    "ipv": ("ipv",),
    "ian": ("ian",),
    "mat": ("mat", "matem", "nota_mat_2022"),
    "por": ("por", "portug", "nota_port_2022"),
    "ing": ("ing", "ingles", "nota_ing_2022", "nota_ing"),
    "fase_ideal": ("fase_ideal", "nivel_ideal_2021", "nivel_ideal_2022"),
    "defasagem": ("defasagem", "defas"),
}

CURRENT_COLUMNS_BY_YEAR = {
    2022: {"pedra_atual": "pedra_22", "inde_atual": "inde_22", "inde_anterior": None},
    2023: {"pedra_atual": "pedra_23", "inde_atual": "inde_23", "inde_anterior": "inde_22"},
    2024: {"pedra_atual": "pedra_24", "inde_atual": "inde_24", "inde_anterior": "inde_23"},
}


@dataclass
class DatasetBundle:
    base_analitica: pd.DataFrame
    base_pares: pd.DataFrame
    base_inferencia: pd.DataFrame
    caminho_excel: Path
    caminho_csv_legado: Path | None = None


def ensure_data_dirs() -> None:
    for directory in (INTERIM_DIR, PROCESSED_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def normalize_column_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def coerce_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(value)


def resolve_excel_path(explicit_path: Path | str | None = None, root: Path | str | None = None) -> Path:
    explicit = coerce_path(explicit_path)
    if explicit is not None:
        return explicit.resolve()

    search_root = (coerce_path(root) or Path.cwd()).resolve()
    candidates = (
        search_root / "data" / "raw" / DEFAULT_EXCEL_NAME,
        RAW_DIR / DEFAULT_EXCEL_NAME,
        search_root / DEFAULT_EXCEL_NAME,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Arquivo Excel nao encontrado. Procurei por: {', '.join(str(path) for path in candidates)}")


def resolve_legacy_csv_path(explicit_path: Path | str | None = None, root: Path | str | None = None) -> Path | None:
    explicit = coerce_path(explicit_path)
    if explicit is not None:
        resolved = explicit.resolve()
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"Arquivo CSV legado nao encontrado em: {resolved}")

    search_root = (coerce_path(root) or Path.cwd()).resolve()
    candidates = (
        search_root / "archive" / "legacy" / "data" / DEFAULT_CSV_NAME,
        search_root / "data" / "raw" / DEFAULT_CSV_NAME,
        LEGACY_DATA_DIR / DEFAULT_CSV_NAME,
        RAW_DIR / DEFAULT_CSV_NAME,
        search_root / DEFAULT_CSV_NAME,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def coerce_numeric(series: pd.Series) -> pd.Series:
    def _clean(value: object) -> object:
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        if text in {"", "nan", "None", "NULL", "INCLUIR"}:
            return np.nan
        return text.replace(",", ".")

    cleaned = series.map(_clean)
    return pd.to_numeric(cleaned, errors="coerce")


def clean_text(value: object) -> str | float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    return text if text else np.nan


def normalize_gender(value: object) -> str | float:
    text = clean_text(value)
    if pd.isna(text):
        return np.nan
    lowered = str(text).lower()
    mapping = {
        "menino": "Masculino",
        "m": "Masculino",
        "masculino": "Masculino",
        "menina": "Feminino",
        "f": "Feminino",
        "feminino": "Feminino",
    }
    return mapping.get(lowered, str(text).title())


def title_or_nan(value: object) -> str | float:
    text = clean_text(value)
    if pd.isna(text):
        return np.nan
    normalized = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return normalized.title()


def extract_year_from_sheet(sheet_name: str) -> int:
    match = re.search(r"(20\d{2})", sheet_name)
    if match is None:
        raise ValueError(f"Nao foi possivel identificar o ano da aba '{sheet_name}'.")
    return int(match.group(1))


def extract_phase_number(value: object) -> float:
    text = clean_text(value)
    if pd.isna(text):
        return np.nan
    lowered = str(text).lower()
    if "alfa" in lowered:
        return 0.0
    match = re.search(r"(\d+)", lowered)
    if match:
        return float(match.group(1))
    return np.nan


def categorize_ian(value: float) -> str:
    if pd.isna(value):
        return "Nao informado"
    if value <= 2.5:
        return "Defasagem severa"
    if value <= 5:
        return "Defasagem moderada"
    return "Adequado"


def select_first_available(frame: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series:
    for alias in aliases:
        if alias in frame.columns:
            return frame[alias]
    return pd.Series(np.nan, index=frame.index)


def load_raw_sheets(excel_path: Path, sheets: tuple[str, ...] = DEFAULT_SHEETS) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    with pd.ExcelFile(excel_path) as workbook:
        available = set(workbook.sheet_names)
        for sheet_name in sheets:
            if sheet_name not in available:
                continue
            frame = pd.read_excel(workbook, sheet_name=sheet_name)
            frame = frame.copy()
            frame.columns = [normalize_column_name(column) for column in frame.columns]
            frames[sheet_name] = frame
    if not frames:
        raise ValueError("Nenhuma aba do PEDE foi encontrada no arquivo informado.")
    return frames


def harmonize_sheet(sheet_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    year = extract_year_from_sheet(sheet_name)
    current_cols = CURRENT_COLUMNS_BY_YEAR[year]

    data: dict[str, pd.Series | int] = {
        "ano_referencia": pd.Series(year, index=frame.index),
        "origem_aba": pd.Series(sheet_name, index=frame.index),
    }
    for canonical_name, aliases in CANONICAL_ALIASES.items():
        data[canonical_name] = select_first_available(frame, aliases)

    base = pd.DataFrame(data)
    base["ra"] = base["ra"].map(clean_text)
    base["nome_exibicao"] = base["nome_exibicao"].map(clean_text)
    base["fase"] = base["fase"].map(title_or_nan)
    base["turma"] = base["turma"].map(clean_text)
    base["genero"] = base["genero"].map(normalize_gender)
    base["instituicao"] = base["instituicao"].map(clean_text)

    for column in ("pedra_20", "pedra_21", "pedra_22", "pedra_23", "pedra_24"):
        base[column] = base[column].map(title_or_nan)

    for column in ("ano_nasc", "idade", "ano_ingresso", "inde_22", "inde_23", "inde_24", "cg", "cf", "ct", "n_av", "iaa", "ieg", "ips", "ipp", "ida", "ipv", "ian", "mat", "por", "ing"):
        base[column] = coerce_numeric(base[column])

    base["idade"] = base["idade"].where(base["idade"].between(6, 30))
    base["fase_ideal_num"] = base["fase_ideal"].map(extract_phase_number)
    base["pedra_atual"] = base[current_cols["pedra_atual"]].map(title_or_nan)
    base["ciclo_programa"] = base["pedra_atual"].fillna("Nao informado")
    base["inde_atual"] = base[current_cols["inde_atual"]]
    base["inde_anterior"] = base[current_cols["inde_anterior"]] if current_cols["inde_anterior"] else np.nan
    base["delta_inde"] = base["inde_atual"] - base["inde_anterior"]
    base["media_academica"] = base[["mat", "por", "ing"]].mean(axis=1)
    base["media_comportamental"] = base[["iaa", "ieg", "ips", "ipp"]].mean(axis=1)
    base["risco_atual"] = np.where(base["ian"].notna(), (base["ian"] <= 5).astype(int), np.nan)
    base["categoria_ian"] = base["ian"].map(categorize_ian)
    base["anos_no_programa"] = np.where(
        base["ano_ingresso"].notna(),
        base["ano_referencia"] - base["ano_ingresso"],
        np.nan,
    )
    base["anos_no_programa"] = pd.Series(base["anos_no_programa"], index=base.index).where(lambda s: s >= 0)

    final_columns = [
        "ra",
        "nome_exibicao",
        "ano_referencia",
        "origem_aba",
        "fase",
        "turma",
        "ciclo_programa",
        "pedra_atual",
        "genero",
        "idade",
        "ano_nasc",
        "ano_ingresso",
        "anos_no_programa",
        "instituicao",
        "fase_ideal_num",
        "inde_22",
        "inde_23",
        "inde_24",
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
        "ian",
        "categoria_ian",
        "risco_atual",
        "mat",
        "por",
        "ing",
        "media_academica",
        "media_comportamental",
        "defasagem",
    ]
    return base[final_columns]


def build_analytical_base(excel_path: Path | str | None = None, root: Path | str | None = None) -> pd.DataFrame:
    ensure_data_dirs()
    resolved_excel = resolve_excel_path(excel_path, root)
    frames = load_raw_sheets(resolved_excel)
    analytical = pd.concat(
        [harmonize_sheet(sheet_name, frame) for sheet_name, frame in frames.items()],
        ignore_index=True,
        sort=False,
    )
    analytical = analytical.sort_values(["ano_referencia", "ra"]).reset_index(drop=True)
    for column in NUMERIC_COLUMNS:
        if column in analytical.columns:
            analytical[column] = pd.to_numeric(analytical[column], errors="coerce")
    return analytical


def build_pair_dataset(base_analitica: pd.DataFrame) -> pd.DataFrame:
    base = base_analitica.sort_values(["ra", "ano_referencia"]).copy()
    base["risco_proximo_ano"] = base.groupby("ra")["risco_atual"].shift(-1)
    base["ano_alvo"] = base.groupby("ra")["ano_referencia"].shift(-1)
    pairs = base[(base["ano_alvo"] == base["ano_referencia"] + 1) & base["risco_proximo_ano"].notna()].copy()
    pairs["ano_alvo"] = pairs["ano_alvo"].astype(int)
    pairs["risco_proximo_ano"] = pairs["risco_proximo_ano"].astype(int)
    return pairs.reset_index(drop=True)


def save_processed_data(base_analitica: pd.DataFrame, base_pares: pd.DataFrame) -> None:
    ensure_data_dirs()
    base_analitica.to_parquet(PROCESSED_DIR / ANALYTICAL_BASE_NAME, index=False)
    base_pares.to_parquet(PROCESSED_DIR / "base_modelagem_proximo_ano.parquet", index=False)
    base_analitica.to_csv(INTERIM_DIR / "base_analitica_preview.csv", index=False, encoding="utf-8")


def prepare_datasets(excel_path: Path | str | None = None, root: Path | str | None = None) -> DatasetBundle:
    resolved_excel = resolve_excel_path(excel_path, root)
    resolved_csv = resolve_legacy_csv_path(root=root)
    base_analitica = build_analytical_base(resolved_excel)
    base_pares = build_pair_dataset(base_analitica)
    base_inferencia = base_analitica[base_analitica["ano_referencia"] == base_analitica["ano_referencia"].max()].copy()
    save_processed_data(base_analitica, base_pares)
    return DatasetBundle(
        base_analitica=base_analitica,
        base_pares=base_pares,
        base_inferencia=base_inferencia,
        caminho_excel=resolved_excel,
        caminho_csv_legado=resolved_csv,
    )
