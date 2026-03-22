from __future__ import annotations

import unittest
from pathlib import Path

from src.passos_magicos_dt.data import build_analytical_base, build_pair_dataset, resolve_excel_path, resolve_legacy_csv_path
from src.passos_magicos_dt.modeling import FEATURE_COLUMNS, build_feature_frame, split_modeling_frames
from src.passos_magicos_dt.runtime import get_package_storyboard_path


class PipelineSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.excel_path = resolve_excel_path(root=Path("."))
        self.base = build_analytical_base(self.excel_path)
        self.pairs = build_pair_dataset(self.base)

    def test_analytical_base_has_expected_years(self) -> None:
        self.assertEqual(sorted(self.base["ano_referencia"].dropna().unique().tolist()), [2022, 2023, 2024])

    def test_pair_dataset_has_expected_years(self) -> None:
        self.assertGreater(len(self.pairs), 1000)
        self.assertEqual(sorted(self.pairs["ano_referencia"].unique().tolist()), [2022, 2023])
        self.assertTrue({"ano_alvo", "risco_proximo_ano"}.issubset(self.pairs.columns))

    def test_feature_frame_contains_all_columns(self) -> None:
        frame = build_feature_frame(self.pairs.head(5))
        self.assertEqual(frame.columns.tolist(), FEATURE_COLUMNS)

    def test_split_has_selection_and_holdout(self) -> None:
        selection, holdout, production = split_modeling_frames(self.pairs)
        self.assertGreater(len(selection), 0)
        self.assertGreater(len(holdout), 0)
        self.assertGreaterEqual(len(production), len(selection) + len(holdout))

    def test_path_resolution_accepts_string_root(self) -> None:
        excel_path = resolve_excel_path(root=".")
        legacy_csv = resolve_legacy_csv_path(root=".")
        self.assertTrue(excel_path.exists())
        self.assertIsNotNone(legacy_csv)
        self.assertTrue(legacy_csv.exists())

    def test_runtime_storyboard_source_exists(self) -> None:
        self.assertTrue(get_package_storyboard_path().exists())


if __name__ == "__main__":
    unittest.main()
