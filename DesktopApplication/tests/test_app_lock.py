import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from server.main import create_app


EXPECTED_HOST = "127.0.0.1:49124"
API_TOKEN = "test-token-for-app-lock-tests-123456789012345"
PASSWORD = "local-pin-123"


def gpu_status() -> dict[str, object]:
    return {
        "available": False,
        "cuda_available": False,
        "gpu_name": None,
        "free_mib": None,
        "total_mib": None,
        "minimum_free_mib": None,
        "safe_to_load_model": False,
        "warning": "GPU unavailable in app-lock tests.",
        "checked_at": "test",
    }


def secure_headers() -> dict[str, str]:
    return {
        "Host": EXPECTED_HOST,
        "Origin": f"http://{EXPECTED_HOST}",
        "X-Desktop-Token": API_TOKEN,
    }


def new_app(root: Path):
    return create_app(
        root,
        gpu_status,
        api_token=API_TOKEN,
        expected_host=EXPECTED_HOST,
    )


class AppLockTests(unittest.TestCase):
    def test_lock_persists_as_argon2id_and_reset_preserves_local_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with TestClient(new_app(root)) as client:
                store = client.app.state.local_store
                store.save_draft("current", "Keep this draft", {})
                store.save_generation(
                    "keep-history",
                    "Keep this speech",
                    "vanna",
                    "natural",
                    4,
                )
                setup = client.post(
                    "/api/security/setup",
                    headers=secure_headers(),
                    json={"password": PASSWORD, "confirmation": PASSWORD},
                )
                password_hash = store.get_setting("app_lock.password_hash")
                locked = client.post("/api/security/lock", headers=secure_headers())
                protected = client.get(
                    "/api/storage/drafts/current",
                    headers=secure_headers(),
                )

            self.assertEqual(setup.status_code, 200, setup.text)
            self.assertTrue(password_hash.startswith("$argon2id$"))
            self.assertNotIn(PASSWORD, password_hash)
            self.assertEqual(locked.status_code, 200, locked.text)
            self.assertEqual(protected.status_code, 423)

            with TestClient(new_app(root)) as reopened:
                status = reopened.get("/api/security/status", headers=secure_headers())
                denied = reopened.get("/api/tts/generations", headers=secure_headers())
                wrong_password = reopened.post(
                    "/api/security/unlock",
                    headers=secure_headers(),
                    json={"password": "incorrect-pin"},
                )
                unlocked = reopened.post(
                    "/api/security/unlock",
                    headers=secure_headers(),
                    json={"password": PASSWORD},
                )
                locked_again = reopened.post("/api/security/lock", headers=secure_headers())
                reset = reopened.post("/api/security/reset", headers=secure_headers())
                draft = reopened.get(
                    "/api/storage/drafts/current",
                    headers=secure_headers(),
                )
                history = reopened.get(
                    "/api/tts/generations",
                    headers=secure_headers(),
                )

            self.assertTrue(status.json()["enabled"])
            self.assertTrue(status.json()["locked"])
            self.assertEqual(denied.status_code, 423)
            self.assertEqual(wrong_password.status_code, 401)
            self.assertEqual(unlocked.status_code, 200)
            self.assertEqual(locked_again.status_code, 200)
            self.assertEqual(reset.status_code, 200, reset.text)
            self.assertIn("were not deleted", reset.json()["message"])
            self.assertEqual(draft.json()["text"], "Keep this draft")
            self.assertEqual(history.json()["items"][0]["id"], "keep-history")

    def test_repeated_wrong_passwords_lock_out_and_survive_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with TestClient(new_app(root)) as client:
                setup = client.post(
                    "/api/security/setup",
                    headers=secure_headers(),
                    json={"password": PASSWORD, "confirmation": PASSWORD},
                )
                self.assertEqual(setup.status_code, 200, setup.text)
                client.post("/api/security/lock", headers=secure_headers())
                attempts = [
                    client.post(
                        "/api/security/unlock",
                        headers=secure_headers(),
                        json={"password": "wrong-pin"},
                    )
                    for _ in range(5)
                ]

            self.assertEqual([response.status_code for response in attempts], [401] * 4 + [429])
            self.assertEqual(attempts[-1].headers["retry-after"], "60")

            with TestClient(new_app(root)) as reopened:
                status = reopened.get("/api/security/status", headers=secure_headers())
                still_locked_out = reopened.post(
                    "/api/security/unlock",
                    headers=secure_headers(),
                    json={"password": PASSWORD},
                )
                reset = reopened.post("/api/security/reset", headers=secure_headers())

            self.assertGreater(status.json()["lockout_remaining_seconds"], 0)
            self.assertEqual(still_locked_out.status_code, 429)
            self.assertEqual(reset.status_code, 200)

    def test_setup_requires_matching_passwords_and_disable_verifies_password(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with TestClient(new_app(Path(temporary_directory))) as client:
                mismatch = client.post(
                    "/api/security/setup",
                    headers=secure_headers(),
                    json={"password": PASSWORD, "confirmation": "different-pin"},
                )
                setup = client.post(
                    "/api/security/setup",
                    headers=secure_headers(),
                    json={"password": PASSWORD, "confirmation": PASSWORD},
                )
                bad_disable = client.post(
                    "/api/security/disable",
                    headers=secure_headers(),
                    json={"password": "incorrect-pin"},
                )
                disabled = client.post(
                    "/api/security/disable",
                    headers=secure_headers(),
                    json={"password": PASSWORD},
                )

            self.assertEqual(mismatch.status_code, 422)
            self.assertEqual(setup.status_code, 200)
            self.assertEqual(bad_disable.status_code, 401)
            self.assertEqual(disabled.status_code, 200)
            self.assertFalse(disabled.json()["enabled"])


if __name__ == "__main__":
    unittest.main()
