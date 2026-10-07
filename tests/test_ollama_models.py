import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('ollama_models', Path(__file__).resolve().parents[1] / 'scripts/ollama_models.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OllamaModelTests(unittest.TestCase):
    def test_existing_models_are_reused_and_missing_model_pulled_once(self):
        with patch.object(module, 'urlopen', return_value=io.BytesIO(
                b'{"status":"pulling manifest"}\n{"status":"success"}\n')) as request:
            module.ensure_models(['available', 'missing', 'missing'],
                                 {'models': [{'name': 'available'}]})
        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.args[0].full_url, 'http://127.0.0.1:11434/api/pull')
        self.assertEqual(request.call_args.args[0].data, b'{"model": "missing", "stream": true}')

    def test_api_error_and_incomplete_download_fail(self):
        for response in (b'{"error":"download failed"}\n', b'{"status":"pulling manifest"}\n', b''):
            with self.subTest(response=response):
                with patch.object(module, 'urlopen', return_value=io.BytesIO(response)):
                    with self.assertRaises(RuntimeError):
                        module.ensure_models(['missing'], {'models': []})

    def test_invalid_model_list_fails_before_network(self):
        with patch.object(module, 'urlopen') as request:
            for models in ([], [''], [None], 'model'):
                with self.assertRaises(ValueError):
                    module.ensure_models(models, {'models': []})
            request.assert_not_called()


if __name__ == '__main__':
    unittest.main()
