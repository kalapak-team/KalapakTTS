import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from server.main import create_app


EXPECTED_HOST = "127.0.0.1:49123"
EXPECTED_ORIGIN = f"http://{EXPECTED_HOST}"
API_TOKEN = "test-token-for-desktop-security-checks-1234567890"


def gpu_status() -> dict[str, object]:
    return {
        "available": False,
        "cuda_available": False,
        "gpu_name": None,
        "free_mib": None,
        "total_mib": None,
        "minimum_free_mib": None,
        "safe_to_load_model": False,
        "warning": "GPU unavailable in API security tests.",
        "checked_at": "test",
    }


class DesktopApiSecurityTests(unittest.TestCase):
    def test_api_requires_token_and_rejects_foreign_host_or_origin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application = create_app(
                Path(temporary_directory),
                gpu_status,
                api_token=API_TOKEN,
                expected_host=EXPECTED_HOST,
            )
            with TestClient(application) as client:
                no_token = client.get(
                    "/api/storage/status",
                    headers={"Host": EXPECTED_HOST},
                )
                invalid_token = client.get(
                    "/api/storage/status",
                    headers={
                        "Host": EXPECTED_HOST,
                        "X-Desktop-Token": "invalid",
                    },
                )
                wrong_host = client.get(
                    "/api/storage/status",
                    headers={
                        "Host": "attacker.example",
                        "X-Desktop-Token": API_TOKEN,
                        "Origin": EXPECTED_ORIGIN,
                    },
                )
                wrong_origin = client.get(
                    "/api/storage/status",
                    headers={
                        "Host": EXPECTED_HOST,
                        "X-Desktop-Token": API_TOKEN,
                        "Origin": "http://attacker.example",
                    },
                )
                cross_site = client.get(
                    "/api/storage/status",
                    headers={
                        "Host": EXPECTED_HOST,
                        "X-Desktop-Token": API_TOKEN,
                        "Sec-Fetch-Site": "cross-site",
                    },
                )
                authorized = client.get(
                    "/api/storage/status",
                    headers={
                        "Host": EXPECTED_HOST,
                        "X-Desktop-Token": API_TOKEN,
                        "Origin": EXPECTED_ORIGIN,
                    },
                )

        self.assertEqual(no_token.status_code, 401)
        self.assertEqual(invalid_token.status_code, 401)
        self.assertEqual(wrong_host.status_code, 403)
        self.assertEqual(wrong_origin.status_code, 403)
        self.assertEqual(cross_site.status_code, 403)
        self.assertEqual(authorized.status_code, 200, authorized.text)
        self.assertEqual(authorized.headers["cache-control"], "no-store")
        self.assertEqual(authorized.headers["cross-origin-resource-policy"], "same-origin")

    def test_health_endpoint_requires_token_and_matching_host(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application = create_app(
                Path(temporary_directory),
                gpu_status,
                api_token=API_TOKEN,
                expected_host=EXPECTED_HOST,
            )
            with TestClient(application) as client:
                denied = client.get("/health", headers={"Host": EXPECTED_HOST})
                authorized = client.get(
                    "/health",
                    headers={
                        "Host": EXPECTED_HOST,
                        "X-Desktop-Token": API_TOKEN,
                    },
                )

        self.assertEqual(denied.status_code, 401)
        self.assertEqual(authorized.status_code, 200)

    def test_secure_mode_rejects_short_or_hostless_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "long token"):
            create_app(api_token="short", expected_host=EXPECTED_HOST)
        with self.assertRaisesRegex(ValueError, "long token"):
            create_app(api_token=API_TOKEN)


if __name__ == "__main__":
    unittest.main()
