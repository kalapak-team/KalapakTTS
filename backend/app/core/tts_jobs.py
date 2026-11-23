import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from app.config import settings


JobStatus = Literal["queued", "running", "completed", "failed"]
generation_slots = asyncio.Semaphore(settings.TTS_MAX_CONCURRENT_JOBS)


@dataclass
class GenerationJob:
    job_id: UUID
    user_id: UUID
    total_chunks: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: JobStatus = "queued"
    current_chunk: int = 0
    message: str = "Waiting to start"
    queued_position: int | None = None
    queue_wait_seconds: float | None = None
    first_audio_seconds: float | None = None
    generation_seconds: float | None = None
    pace_retry_count: int = 0
    generation: dict[str, Any] | None = None
    error: str | None = None
    task: asyncio.Task[None] | None = field(default=None, repr=False)

    def as_response(self) -> dict[str, Any]:
        if self.status == "queued":
            queued_jobs = sorted(
                (job for job in _jobs.values() if job.status == "queued"),
                key=lambda job: job.created_at,
            )
            self.queued_position = next(
                index for index, job in enumerate(queued_jobs, start=1)
                if job.job_id == self.job_id
            )
        else:
            self.queued_position = None
        return {
            "job_id": self.job_id,
            "status": self.status,
            "current_chunk": self.current_chunk,
            "total_chunks": self.total_chunks,
            "message": self.message,
            "queued_position": self.queued_position,
            "queue_wait_seconds": self.queue_wait_seconds,
            "first_audio_seconds": self.first_audio_seconds,
            "generation_seconds": self.generation_seconds,
            "pace_retry_count": self.pace_retry_count,
            "generation": self.generation,
            "error": self.error,
        }


_jobs: dict[UUID, GenerationJob] = {}


def create_job(user_id: UUID, total_chunks: int) -> GenerationJob:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=2)
    expired = [
        job_id
        for job_id, job in _jobs.items()
        if job.status in ("completed", "failed") and job.created_at < cutoff
    ]
    for job_id in expired:
        _jobs.pop(job_id, None)

    job = GenerationJob(job_id=uuid4(), user_id=user_id, total_chunks=total_chunks)
    _jobs[job.job_id] = job
    return job


def get_job(job_id: UUID, user_id: UUID) -> GenerationJob | None:
    job = _jobs.get(job_id)
    if job is None or job.user_id != user_id:
        return None
    return job
