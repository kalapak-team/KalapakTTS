import json
import uuid

from tests.api_test_support import (
    ApiTestCase,
    api_request,
    persisted_voice,
)


class VoiceLibraryFlowTests(ApiTestCase):
    def test_public_voice_library_is_available_without_authentication(self):
        status_code, _, body = api_request("GET", "/api/voices/")
        self.assertEqual(status_code, 200, body.decode("utf-8", errors="replace"))
        voices = json.loads(body)
        self.assertGreaterEqual(len(voices), 8)
        vanna = next(voice for voice in voices if voice["name"] == "វណ្ណា")
        self.assertTrue(vanna["has_reference_audio"])
        vanna2 = next(voice for voice in voices if voice["name"] == "វណ្ណា២")
        self.assertTrue(vanna2["has_reference_audio"])
        vanna3 = next(voice for voice in voices if voice["name"] == "វណ្ណា៣")
        self.assertTrue(vanna3["has_reference_audio"])
        vanna4 = next(voice for voice in voices if voice["name"] == "វណ្ណា៤")
        self.assertTrue(vanna4["has_reference_audio"])

    def test_authenticated_voice_creation_and_persistence(self):
        status_code, _, _ = api_request("GET", "/api/voices/")
        self.assertEqual(status_code, 200)

        _, token = self.create_and_login_user()
        name = f"Khmer test voice {uuid.uuid4().hex[:8]}"
        status_code, _, body = api_request(
            "POST",
            "/api/voices/",
            token=token,
            json_body={
                "name": name,
                "description": "Temporary integration test voice",
                "control_instruction": "Warm Khmer voice",
                "is_public": False,
            },
        )
        self.assertEqual(status_code, 201, body.decode("utf-8", errors="replace"))
        voice = json.loads(body)

        status_code, _, body = api_request("GET", "/api/voices/", token=token)
        self.assertEqual(status_code, 200, body.decode("utf-8", errors="replace"))
        self.assertIn(voice["id"], {entry["id"] for entry in json.loads(body)})
        self.assertTrue(persisted_voice(voice["id"]))

        self.assertEqual(
            api_request("DELETE", f"/api/voices/{voice['id']}", token=token)[0],
            204,
        )
