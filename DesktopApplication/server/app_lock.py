from __future__ import annotations

from argon2 import PasswordHasher, Type
from argon2.exceptions import VerificationError, VerifyMismatchError


MIN_PASSWORD_LENGTH = 6
MAX_PASSWORD_LENGTH = 128
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 60

_PASSWORD_HASHER = PasswordHasher(
    time_cost=3,
    memory_cost=65_536,
    parallelism=2,
    type=Type.ID,
)


def validate_password(password: str) -> None:
    if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
        raise ValueError(
            f"App-lock PIN/password must contain {MIN_PASSWORD_LENGTH} to "
            f"{MAX_PASSWORD_LENGTH} characters."
        )


def hash_password(password: str) -> str:
    validate_password(password)
    return _PASSWORD_HASHER.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _PASSWORD_HASHER.verify(password_hash, password)
    except (VerificationError, VerifyMismatchError):
        return False
