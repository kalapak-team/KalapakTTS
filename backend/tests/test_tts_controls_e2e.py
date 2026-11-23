import io
import json
import wave

from tests.api_test_support import ApiTestCase, api_request


TEXT = "សួស្តី នេះជាការសាកល្បង"
STYLE_INSTRUCTIONS = {
    "news_anchor": (
        "Speak like a clear, formal, steady news presenter. "
        "និយាយបែបអ្នកអានព័ត៌មាន ឱ្យច្បាស់ ផ្លូវការ និងមានល្បឿនថេរ។"
    ),
    "vlogger": (
        "Use an energetic, friendly, expressive content-creator tone. "
        "និយាយបែបអ្នកបង្កើតមាតិកា ឱ្យស្វាហាប់ រួសរាយ និងរស់រវើក។"
    ),
    "storyteller": (
        "Use a warm storytelling voice with expressive phrasing and pauses. "
        "និយាយបែបអ្នកនិទានរឿង ឱ្យកក់ក្តៅ និងមានចន្លោះឃ្លាសមរម្យ។"
    ),
}


class SpeechControlEndToEndTests(ApiTestCase):
    def delete_generation(self, token, generation_id):
        status_code, _, body = api_request(
            "DELETE", f"/api/tts/history/{generation_id}", token=token
        )
        self.assertEqual(status_code, 204, body.decode("utf-8", errors="replace"))

    def test_speed_scaling_and_style_samples_generate_persisted_audio(self):
        _, token = self.create_and_login_user()
        status_code, _, body = api_request("GET", "/api/voices/", token=token)
        self.assertEqual(status_code, 200, body.decode("utf-8", errors="replace"))
        default_voice = next(
            voice for voice in json.loads(body) if voice["name"] == "មាលី"
        )

        samples = [
            {
                "speed": speed,
                "style": "natural",
                "control_instruction": "",
                "pace_mode": "custom",
                "seed": 20261119,
            }
            for speed in (0.8, 1.0, 1.2)
        ]
        samples.extend(
            {
                "speed": 1.0,
                "style": style,
                "control_instruction": instruction,
                "pace_mode": "custom",
                "seed": 20261119,
            }
            for style, instruction in STYLE_INSTRUCTIONS.items()
        )
        generated = []
        for sample in samples:
            with self.subTest(speed=sample["speed"], style=sample["style"]):
                status_code, _, response_body = api_request(
                    "POST",
                    "/api/tts/generate",
                    token=token,
                    json_body={
                        "text": TEXT,
                        "voice_preset_id": default_voice["id"],
                        "inference_timesteps": 5,
                        **sample,
                    },
                    timeout=600,
                )
                self.assertEqual(
                    status_code,
                    201,
                    response_body.decode("utf-8", errors="replace"),
                )
                generation = json.loads(response_body)
                self.addCleanup(self.delete_generation, token, generation["id"])
                status_code, headers, audio = api_request(
                    "GET", generation["audio_url"], token=token
                )
                self.assertEqual(status_code, 200)
                self.assertIn("audio/wav", headers.get("Content-Type", ""))
                with wave.open(io.BytesIO(audio), "rb") as wav:
                    duration = wav.getnframes() / wav.getframerate()
                self.assertGreater(duration, 0)
                generated.append((generation["id"], sample, duration))

        status_code, _, history_body = api_request(
            "GET", "/api/tts/history?limit=20", token=token
        )
        self.assertEqual(status_code, 200)
        history_by_id = {
            item["id"]: item for item in json.loads(history_body)["items"]
        }
        for generation_id, sample, duration in generated:
            item = history_by_id[generation_id]
            self.assertAlmostEqual(item["speed"], sample["speed"])
            self.assertEqual(item["style"], sample["style"])
            self.assertAlmostEqual(item["audio_duration"], duration, places=2)

        speed_durations = [
            f'{sample["speed"]:.1f}x={duration:.2f}s'
            for _, sample, duration in generated[:3]
        ]
        self.assertGreater(generated[0][2], generated[1][2])
        self.assertGreater(generated[1][2], generated[2][2])
        style_durations = [
            f'{sample["style"]}={duration:.2f}s'
            for _, sample, duration in generated[3:]
        ]
        print("Measured speed samples: " + ", ".join(speed_durations))
        print("Measured style samples: " + ", ".join(style_durations))
