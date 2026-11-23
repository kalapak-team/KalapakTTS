import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from server.gpu import MIN_FREE_VRAM_MIB
from server.main import CONTENT_SECURITY_POLICY, create_app

UI_DIR = Path(__file__).resolve().parents[1] / "ui"


def available_gpu_status() -> dict[str, object]:
    return {
        "available": True,
        "gpu_name": "Test GPU",
        "free_mib": 6141,
        "total_mib": 6141,
        "minimum_free_mib": MIN_FREE_VRAM_MIB,
        "safe_to_load_model": True,
        "warning": "Test GPU status.",
        "checked_at": "test",
    }


class DesktopPocTests(unittest.TestCase):
    def test_local_health_endpoint(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            application = create_app(Path(temporary_directory), available_gpu_status)
            with TestClient(application) as client:
                response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_khmer_proof_page_is_served_as_utf8(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            application = create_app(Path(temporary_directory), available_gpu_status)
            with TestClient(application) as client:
                response = client.get("/poc")

        self.assertEqual(response.status_code, 200)
        self.assertIn("សួស្តី! សូមសាកល្បងអក្សរខ្មែរ", response.text)
        self.assertIn("charset=utf-8", response.headers["content-type"])

    def test_static_export_and_bundled_font_are_served_from_root(self) -> None:
        fonts = list((UI_DIR / "fonts").glob("*.woff2"))
        self.assertTrue(fonts, "Build the desktop static UI before running this test.")

        with TemporaryDirectory() as temporary_directory:
            application = create_app(Path(temporary_directory), available_gpu_status)
            with TestClient(application) as client:
                page = client.get("/")
                font = client.get(f"/fonts/{fonts[0].name}")

        self.assertEqual(page.status_code, 200)
        self.assertIn("KalapakTTS", page.text)
        self.assertEqual(font.status_code, 200)

    def test_local_api_sets_external_blocking_csp(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            application = create_app(Path(temporary_directory), available_gpu_status)
            with TestClient(application) as client:
                response = client.get("/")

        csp = response.headers["content-security-policy"]
        self.assertIn("default-src 'self'", csp)
        self.assertIn("connect-src 'self'", csp)
        self.assertIn("font-src 'self'", csp)
        self.assertNotIn("https:", csp)
