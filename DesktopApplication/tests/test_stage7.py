import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.launcher import _select_model_root
from server.config import DesktopSettings


class ModelRootSelectionTests(unittest.TestCase):
    def test_installed_profile_models_take_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            desktop_root = root / "app"
            profile_root = root / "profile" / "models"
            installed_manifest = profile_root / "VoxCPM2" / "manifest.json"
            development_manifest = desktop_root / "models" / "VoxCPM2" / "manifest.json"
            installed_manifest.parent.mkdir(parents=True)
            development_manifest.parent.mkdir(parents=True)
            installed_manifest.write_text("{}", encoding="utf-8")
            development_manifest.write_text("{}", encoding="utf-8")

            selected = _select_model_root(desktop_root, profile_root)

        self.assertEqual(selected, profile_root)

    def test_missing_pack_targets_profile_models_and_development_pack_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            desktop_root = root / "app"
            profile_root = root / "profile" / "models"

            self.assertEqual(
                _select_model_root(desktop_root, profile_root),
                profile_root,
            )

            development_manifest = desktop_root / "models" / "VoxCPM2" / "manifest.json"
            development_manifest.parent.mkdir(parents=True)
            development_manifest.write_text("{}", encoding="utf-8")

            self.assertEqual(
                _select_model_root(desktop_root, profile_root),
                desktop_root / "models",
            )

    def test_config_uses_installed_model_root_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_root = Path(temporary_directory) / "models"
            with patch.dict("os.environ", {"KALAPAK_TTS_MODELS": str(model_root)}):
                settings = DesktopSettings()

        self.assertEqual(settings.MODEL_DIR, model_root / "VoxCPM2")
        self.assertEqual(settings.VOICE_REFERENCE_DIR, model_root / "voices")


if __name__ == "__main__":
    unittest.main()
