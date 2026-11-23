FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ffmpeg libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY VoxCPM/ ./VoxCPM/
COPY backend/requirements.txt ./backend/requirements.txt

RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.5.0" "torchaudio>=2.5.0"

RUN python -m pip install --no-cache-dir -r ./backend/requirements.txt

RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VOXCPM=0.0.0 \
    python -m pip install --no-cache-dir ./VoxCPM

COPY backend/app/ ./backend/app/
COPY backend/download_voxcpm_model.py ./backend/download_voxcpm_model.py
COPY backend/voice_references/ ./backend/voice_references/

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /models /data/audio_files \
    && chown -R app:app /app/backend /models /data/audio_files

WORKDIR /app/backend
ENV HF_HOME=/models \
    AUDIO_FILES_DIR=/data/audio_files

USER app
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
