import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from server.model_assets import ModelAssetsError, model_asset_status, verify_model_assets


class ModelAssetTests(unittest.TestCase):
    def test_asset_manifest_and_hash_are_verified_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory)
            payload = b"local model fixture"
            digest = hashlib.sha256(payload).hexdigest()
            (model_dir / "model.bin").write_bytes(payload)
            (model_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "files": [
                            {
                                "path": "model.bin",
                                "size_bytes": len(payload),
                                "sha256": digest,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            expected = {"model.bin": (len(payload), digest)}

            status = model_asset_status(model_dir, expected)

            self.assertTrue(status["available"])
            self.assertEqual(verify_model_assets(model_dir, expected), model_dir)

    def test_missing_and_corrupt_assets_return_clear_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_dir = Path(temporary_directory)
            payload = b"model"
            digest = hashlib.sha256(payload).hexdigest()
            (model_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "files": [
                            {"path": "model.bin", "size_bytes": len(payload), "sha256": digest}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            expected = {"model.bin": (len(payload), digest)}

            missing = model_asset_status(model_dir, expected)
            self.assertFalse(missing["available"])
            self.assertIn("Missing model file: model.bin", missing["message"])
            with self.assertRaisesRegex(ModelAssetsError, "Missing model file"):
                verify_model_assets(model_dir, expected)

            (model_dir / "model.bin").write_bytes(b"other")
            with self.assertRaisesRegex(ModelAssetsError, "SHA-256 verification failed"):
                verify_model_assets(model_dir, expected)


if __name__ == "__main__":
    unittest.main()
