from __future__ import annotations

import unittest
from unittest.mock import patch

import app


class AppModelBundleLoadTests(unittest.TestCase):
    @patch("app.ensure_runtime_ready")
    @patch("app.load_model_bundle")
    @patch("app.preparar_runtime")
    def test_load_model_bundle_without_rebuild(
        self,
        preparar_runtime_mock,
        load_model_bundle_mock,
        ensure_runtime_ready_mock,
    ) -> None:
        model_tuple = ("modelo", {"selected_threshold": 0.5})
        load_model_bundle_mock.return_value = model_tuple

        result = app._load_or_rebuild_model_bundle()

        self.assertEqual(result, model_tuple)
        preparar_runtime_mock.assert_called_once_with()
        load_model_bundle_mock.assert_called_once_with(app.MODEL_DIR)
        ensure_runtime_ready_mock.assert_not_called()

    @patch("app.ensure_runtime_ready")
    @patch("app.load_model_bundle")
    @patch("app.preparar_runtime")
    def test_rebuild_when_versioned_bundle_load_fails(
        self,
        preparar_runtime_mock,
        load_model_bundle_mock,
        ensure_runtime_ready_mock,
    ) -> None:
        model_tuple = ("modelo_reconstruido", {"selected_threshold": 0.4})
        load_model_bundle_mock.side_effect = [RuntimeError("bundle incompatível"), model_tuple]

        result = app._load_or_rebuild_model_bundle()

        self.assertEqual(result, model_tuple)
        preparar_runtime_mock.assert_called_once_with()
        self.assertEqual(load_model_bundle_mock.call_count, 2)
        ensure_runtime_ready_mock.assert_called_once_with(force=True)


if __name__ == "__main__":
    unittest.main()
