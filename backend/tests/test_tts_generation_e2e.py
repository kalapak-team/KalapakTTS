import io
import json
import wave

from tests.api_test_support import ApiTestCase, api_request


class SpeechGenerationFlowTests(ApiTestCase):
    def test_vanna_reference_voice_generates_valid_wav(self):
        _, token = self.create_and_login_user()
        status_code, _, body = api_request("GET", "/api/voices/", token=token)
        self.assertEqual(status_code, 200, body.decode("utf-8", errors="replace"))
        vanna = next(
            voice for voice in json.loads(body) if voice["name"] == "វណ្ណា"
        )
        self.assertTrue(vanna["has_reference_audio"])

        status_code, _, body = api_request(
            "POST",
            "/api/tts/generate",
            token=token,
            json_body={
                "text": "សួស្តី នេះជាសំឡេង វណ្ណា",
                "voice_preset_id": vanna["id"],
                "inference_timesteps": 5,
                "speed": 0.8,
                "style": "news_anchor",
            },
            timeout=600,
        )
        self.assertEqual(status_code, 201, body.decode("utf-8", errors="replace"))
        generation = json.loads(body)
        status_code, headers, audio = api_request(
            "GET", generation["audio_url"], token=token
        )
        self.assertEqual(status_code, 200)
        self.assertIn("audio/wav", headers.get("Content-Type", ""))
        with wave.open(io.BytesIO(audio), "rb") as wav:
            self.assertGreater(wav.getnframes(), 0)

        self.assertEqual(
            api_request(
                "DELETE",
                f"/api/tts/history/{generation['id']}",
                token=token,
            )[0],
            204,
        )

    def test_khmer_generation_wav_history_and_authentication(self):
        status_code, _, _ = api_request(
            "POST", "/api/tts/generate", json_body={"text": "សួស្តី"}
        )
        self.assertEqual(status_code, 401)

        _, token = self.create_and_login_user()
        text = "សួស្តី នេះជាការសាកល្បង"
        status_code, _, body = api_request(
            "POST",
            "/api/tts/generate",
            token=token,
            json_body={
                "text": text,
                "inference_timesteps": 5,
                "speed": 0.8,
                "style": "news_anchor",
            },
            timeout=600,
        )
        self.assertEqual(status_code, 201, body.decode("utf-8", errors="replace"))
        generation = json.loads(body)

        status_code, headers, audio = api_request(
            "GET", generation["audio_url"], token=token
        )
        self.assertEqual(status_code, 200)
        self.assertGreater(len(audio), 44)
        self.assertIn("audio/wav", headers.get("Content-Type", ""))
        with wave.open(io.BytesIO(audio), "rb") as wav:
            self.assertGreater(wav.getnframes(), 0)
            self.assertGreater(wav.getframerate(), 0)

        status_code, _, body = api_request(
            "GET", "/api/tts/history?limit=20", token=token
        )
        self.assertEqual(status_code, 200, body.decode("utf-8", errors="replace"))
        history = json.loads(body)
        saved = next(item for item in history["items"] if item["id"] == generation["id"])
        self.assertEqual(saved["text_input"], text)
        self.assertGreater(saved["audio_duration"], 0)
        self.assertAlmostEqual(saved["speed"], 0.8)
        self.assertEqual(saved["style"], "news_anchor")

        self.assertEqual(
            api_request(
                "DELETE",
                f"/api/tts/history/{generation['id']}",
                token=token,
            )[0],
            204,
        )
