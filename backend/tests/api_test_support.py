import asyncio
import json
import os
import unittest
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from sqlalchemy import delete, select

from app.database import AsyncSessionLocal, engine
from app.models.generation import Generation
from app.models.user import User
from app.models.voice import Voice


API_URL = os.environ.get("KALAPAK_API_URL", "http://127.0.0.1:8000")


def api_request(method, path, token=None, json_body=None, form_body=None, timeout=30):
    headers = {}
    body = None
    if json_body is not None:
        body = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif form_body is not None:
        body = urlencode(form_body).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(f"{API_URL}{path}", data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, response.headers, response.read()
    except HTTPError as error:
        return error.code, error.headers, error.read()
    except URLError as error:
        raise AssertionError(f"API request failed for {path}: {error}") from error


def remove_test_user(email):
    async def remove():
        try:
            async with AsyncSessionLocal() as session:
                user = await session.scalar(select(User).where(User.email == email))
                if user is None:
                    return

                generations = (
                    await session.scalars(
                        select(Generation).where(Generation.user_id == user.id)
                    )
                ).all()
                for generation in generations:
                    if generation.audio_file_path:
                        Path(generation.audio_file_path).unlink(missing_ok=True)

                await session.execute(
                    delete(Generation).where(Generation.user_id == user.id)
                )
                await session.execute(delete(Voice).where(Voice.owner_id == user.id))
                await session.execute(delete(User).where(User.id == user.id))
                await session.commit()
        finally:
            await engine.dispose()

    asyncio.run(remove())


def persisted_user(user_id):
    async def check():
        try:
            async with AsyncSessionLocal() as session:
                return await session.get(User, uuid.UUID(user_id)) is not None
        finally:
            await engine.dispose()

    return asyncio.run(check())


def persisted_voice(voice_id):
    async def check():
        try:
            async with AsyncSessionLocal() as session:
                return await session.get(Voice, uuid.UUID(voice_id)) is not None
        finally:
            await engine.dispose()

    return asyncio.run(check())


class ApiTestCase(unittest.TestCase):
    def create_and_login_user(self):
        suffix = uuid.uuid4().hex[:12]
        self.email = f"tts-test-{suffix}@example.com"
        self.username = f"tts_test_{suffix}"
        self.password = f"Kalapak-{uuid.uuid4().hex}"
        status_code, _, body = api_request(
            "POST",
            "/api/auth/register",
            json_body={
                "username": self.username,
                "email": self.email,
                "password": self.password,
                "full_name": "Kalapak E2E",
            },
        )
        self.assertEqual(status_code, 201, body.decode("utf-8", errors="replace"))
        user = json.loads(body)
        self.addCleanup(remove_test_user, self.email)

        status_code, _, body = api_request(
            "POST",
            "/api/auth/login",
            form_body={"username": self.email, "password": self.password},
        )
        self.assertEqual(status_code, 200, body.decode("utf-8", errors="replace"))
        token = json.loads(body)["access_token"]
        return user, token
