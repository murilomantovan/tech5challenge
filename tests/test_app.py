from __future__ import annotations

import unittest
from unittest.mock import patch

import app


class AppModelBundleLoadTests(unittest.TestCase):
    @patch("app.ensure_runtime_ready")
    @patch("app.load_model_bundle")
    def test_carrega_bundle_sem_reconstrucao(
        self,
        load_model_bundle_mock,
        ensure_runtime_ready_mock,
    ) -> None:
        model_tuple = ("modelo", {"selected_threshold": 0.5}, {"built": False, "source": "bundle"})
        load_model_bundle_mock.return_value = model_tuple[:2]

        result = app._carregar_ou_reconstruir_bundle_modelo()

        self.assertEqual(result, model_tuple)
        load_model_bundle_mock.assert_called_once_with(app.MODEL_DIR)
        ensure_runtime_ready_mock.assert_not_called()

    @patch("app.ensure_runtime_ready")
    @patch("app.load_model_bundle")
    def test_reconstroi_bundle_quando_a_carga_inicial_falha(
        self,
        load_model_bundle_mock,
        ensure_runtime_ready_mock,
    ) -> None:
        status_runtime = {"built": True, "source": "pipeline"}
        model_tuple = ("modelo_reconstruido", {"selected_threshold": 0.4}, status_runtime)
        load_model_bundle_mock.side_effect = [RuntimeError("bundle incompatível"), model_tuple[:2]]
        ensure_runtime_ready_mock.return_value = status_runtime

        result = app._carregar_ou_reconstruir_bundle_modelo()

        self.assertEqual(result, model_tuple)
        self.assertEqual(load_model_bundle_mock.call_count, 2)
        ensure_runtime_ready_mock.assert_called_once_with(force=True)


if __name__ == "__main__":
    unittest.main()
