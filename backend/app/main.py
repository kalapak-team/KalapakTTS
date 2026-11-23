"""
KalapakTTS FastAPI application entry point.

Startup lifecycle:
  1. Create all DB tables (idempotent via SQLAlchemy metadata)
  2. Seed admin user
  3. Seed built-in Khmer voices
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, text

from app.config import settings
from app.database import engine, AsyncSessionLocal, Base
from app.models import User, Voice, Generation  # noqa: F401 — registers metadata
from app.core.security import get_password_hash
from app.core.tts_engine import tts_engine
from app.api import auth, tts, voices, admin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
VANNA_REFERENCE_AUDIO = Path(__file__).resolve().parents[1] / "voice_references" / "vanna.wav"
VANNA2_REFERENCE_AUDIO = Path(__file__).resolve().parents[2] / "VoiceClone" / "Vanna2.wav"
VANNA3_REFERENCE_AUDIO = Path(__file__).resolve().parents[2] / "VoiceClone" / "Vanna3.wav"
VANNA4_REFERENCE_AUDIO = Path(__file__).resolve().parents[2] / "VoiceClone" / "Vanna4.wav"

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

SEED_VOICES = [
    {
        "name": "មាលី",
        "description": "Young female voice, friendly and clear",
        "control_instruction": "A young Khmer woman, warm and friendly voice, clear pronunciation",
    },
    {
        "name": "ចន្ទ",
        "description": "Male voice, authoritative and deep",
        "control_instruction": "A middle-aged Khmer man, authoritative deep voice",
    },
    {
        "name": "ស្រីពៅ",
        "description": "Female narrator, warm and calm",
        "control_instruction": "A warm Khmer female narrator, calm and soothing",
    },
    {
        "name": "បុណ្ណ",
        "description": "Young male, energetic",
        "control_instruction": "A young Khmer man, energetic and enthusiastic voice",
    },
    {
        "name": "វណ្ណា",
        "description": "Voice cloned from the provided reference recording",
        "control_instruction": "",
        "reference_audio_path": str(VANNA_REFERENCE_AUDIO),
    },
    {
        "name": "វណ្ណា២",
        "description": "Voice cloned from the Vanna2 reference recording",
        "control_instruction": "",
        "reference_audio_path": str(VANNA2_REFERENCE_AUDIO),
    },
    {
        "name": "វណ្ណា៣",
        "description": "Voice cloned from the Vanna3 reference recording",
        "control_instruction": "",
        "reference_audio_path": str(VANNA3_REFERENCE_AUDIO),
    },
    {
        "name": "វណ្ណា៤",
        "description": "Voice cloned from the Vanna4 reference recording",
        "control_instruction": "",
        "reference_audio_path": str(VANNA4_REFERENCE_AUDIO),
    },
]


async def seed_database():
    async with AsyncSessionLocal() as db:
        if settings.ADMIN_PASSWORD.strip():
            result = await db.execute(select(User).where(User.email == settings.ADMIN_EMAIL))
            if not result.scalar_one_or_none():
                admin_user = User(
                    username=settings.ADMIN_USERNAME,
                    email=settings.ADMIN_EMAIL,
                    hashed_password=get_password_hash(settings.ADMIN_PASSWORD),
                    full_name="KalapakTTS Administrator",
                    role="admin",
                    is_active=True,
                )
                db.add(admin_user)
                await db.flush()
                logger.info("Admin user created: %s", settings.ADMIN_EMAIL)
        else:
            logger.warning("Admin seed skipped; configure ADMIN_PASSWORD to create the initial admin account.")

        # Seed voices (match by name, is_public=True, owner_id=None)
        for v_data in SEED_VOICES:
            result = await db.execute(
                select(Voice).where(Voice.name == v_data["name"], Voice.is_public == True)
            )
            voice = result.scalar_one_or_none()
            if not voice:
                voice = Voice(
                    name=v_data["name"],
                    description=v_data["description"],
                    control_instruction=v_data["control_instruction"],
                    reference_audio_path=v_data.get("reference_audio_path"),
                    language="km",
                    is_public=True,
                    owner_id=None,
                )
                db.add(voice)
                logger.info("Seeded voice: %s", v_data["name"])
            elif "reference_audio_path" in v_data:
                voice.description = v_data["description"]
                voice.control_instruction = v_data["control_instruction"]
                voice.reference_audio_path = v_data["reference_audio_path"]

        await db.commit()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.SECRET_KEY.strip():
        raise RuntimeError("SECRET_KEY must be configured before starting the API.")
    tts_engine.log_runtime_info()
    if not VANNA_REFERENCE_AUDIO.is_file():
        raise RuntimeError("The reference audio for the built-in Khmer voice វណ្ណា is missing.")
    if not VANNA2_REFERENCE_AUDIO.is_file():
        raise RuntimeError("The reference audio for the built-in Khmer voice វណ្ណា២ is missing.")
    if not VANNA3_REFERENCE_AUDIO.is_file():
        raise RuntimeError("The reference audio for the built-in Khmer voice វណ្ណា៣ is missing.")
    if not VANNA4_REFERENCE_AUDIO.is_file():
        raise RuntimeError("The reference audio for the built-in Khmer voice វណ្ណា៤ is missing.")

    # Ensure audio dir exists
    Path(settings.AUDIO_FILES_DIR).mkdir(parents=True, exist_ok=True)

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if engine.dialect.name == "postgresql":
            await conn.execute(
                text(
                    "ALTER TABLE voices "
                    "ADD COLUMN IF NOT EXISTS reference_audio_path VARCHAR(1000)"
                )
            )
            await conn.execute(
                text(
                    "ALTER TABLE generations "
                    "ADD COLUMN IF NOT EXISTS speed FLOAT NOT NULL DEFAULT 1.0, "
                    "ADD COLUMN IF NOT EXISTS pace_mode VARCHAR(20) NOT NULL DEFAULT 'custom', "
                    "ADD COLUMN IF NOT EXISTS target_cps FLOAT, "
                    "ADD COLUMN IF NOT EXISTS pronunciation_dictionary BOOLEAN NOT NULL DEFAULT FALSE, "
                    "ADD COLUMN IF NOT EXISTS noise_reduction BOOLEAN NOT NULL DEFAULT FALSE, "
                    "ADD COLUMN IF NOT EXISTS style VARCHAR(50), "
                    "ADD COLUMN IF NOT EXISTS title VARCHAR(100) NOT NULL DEFAULT 'Untitled'"
                )
            )
            await conn.execute(
                text("ALTER TABLE generations ALTER COLUMN target_cps DROP NOT NULL")
            )
            await conn.execute(
                text("ALTER TABLE generations ALTER COLUMN inference_timesteps SET DEFAULT 10")
            )
    logger.info("Database tables ensured.")

    # Seed
    await seed_database()
    logger.info("Database seeding complete.")

    if settings.WARM_MODEL_ON_STARTUP:
        logger.info("Warming the VoxCPM2 model during API startup.")
        await tts_engine.load_model()

    yield

    # Shutdown
    await engine.dispose()
    logger.info("Database engine disposed.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="KalapakTTS API",
    description="Khmer Text-to-Speech API powered by VoxCPM2",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1):(3000|3001|3100|3101)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth.router)
app.include_router(tts.router)
app.include_router(voices.router)
app.include_router(admin.router)


@app.get("/", tags=["health"])
async def root():
    return {"message": "KalapakTTS API is running", "version": "1.0.0"}


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}
