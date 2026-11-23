import json

from tests.api_test_support import ApiTestCase, api_request, persisted_user


class AuthenticationFlowTests(ApiTestCase):
    def test_registration_login_and_persisted_identity(self):
        user, token = self.create_and_login_user()

        status_code, _, body = api_request(
            "POST",
            "/api/auth/login",
            form_body={"username": self.email, "password": "incorrect-password"},
        )
        self.assertEqual(status_code, 401)

        status_code, _, body = api_request("GET", "/api/auth/me", token=token)
        self.assertEqual(status_code, 200, body.decode("utf-8", errors="replace"))
        self.assertEqual(json.loads(body)["id"], user["id"])
        self.assertTrue(persisted_user(user["id"]))
