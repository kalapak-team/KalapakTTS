import asyncio
import json
import uuid

from app.database import AsyncSessionLocal, engine
from app.models.generation import Generation
from tests.api_test_support import ApiTestCase, api_request


class HistoryTitleFlowTests(ApiTestCase):
    def test_title_update_is_authenticated_and_persisted(self):
        user, token = self.create_and_login_user()
        generation_id = uuid.uuid4()

        async def create_generation():
            async with AsyncSessionLocal() as session:
                session.add(
                    Generation(
                        id=generation_id,
                        user_id=uuid.UUID(user["id"]),
                        text_input="សួស្តី",
                    )
                )
                await session.commit()
            await engine.dispose()

        asyncio.run(create_generation())

        self.assertEqual(
            api_request(
                "PATCH",
                f"/api/tts/history/{generation_id}",
                json_body={"title": "   "},
                token=token,
            )[0],
            422,
        )
        self.assertEqual(
            api_request(
                "PATCH",
                f"/api/tts/history/{generation_id}",
                json_body={"title": "Khmer welcome"},
            )[0],
            401,
        )

        status_code, _, body = api_request(
            "PATCH",
            f"/api/tts/history/{generation_id}",
            json_body={"title": "  Khmer welcome  "},
            token=token,
        )
        self.assertEqual(status_code, 200, body.decode("utf-8", errors="replace"))
        self.assertEqual(json.loads(body)["title"], "Khmer welcome")

        status_code, _, body = api_request(
            "GET",
            "/api/tts/history?limit=20",
            token=token,
        )
        self.assertEqual(status_code, 200, body.decode("utf-8", errors="replace"))
        saved = next(
            item for item in json.loads(body)["items"] if item["id"] == str(generation_id)
        )
        self.assertEqual(saved["title"], "Khmer welcome")
