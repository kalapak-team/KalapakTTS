import concurrent.futures
import hashlib
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / "model_cache"))

from huggingface_hub import hf_hub_url, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.utils import get_session

REPOSITORY = "openbmb/VoxCPM2"
REVISION = "7936080b41163874f38610f46f438e2c0757f9fd"
EXPECTED_FILES = {
    "audiovae.pth": (376_951_122, "94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1"),
    "model.safetensors": (4_580_080_592, "f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d"),
}
CHUNK_SIZE = 64 * 1024**2
MAX_WORKERS = 8
MAX_ATTEMPTS = 5


class RedactSignedUrl(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = re.sub(r"https://us\.aws\.cdn\.hf\.co/\S+", "[signed URL redacted]", message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def configure_logging() -> Path:
    log_path = Path(__file__).resolve().parent / "voxcpm-download.log"
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    redaction_filter = RedactSignedUrl()
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    for handler in handlers:
        handler.addFilter(redaction_filter)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )
    for logger_name in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "huggingface_hub.utils._http",
    ):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
        logger.disabled = True
    return log_path


def verify_file(path: Path, expected_size: int, expected_sha256: str) -> None:
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(f"{path.name} has {actual_size} bytes; expected {expected_size}")

    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError(f"{path.name} SHA-256 checksum does not match Hugging Face")


def cached_downloaded_bytes(cache_dir: Path) -> int:
    downloaded = 0
    for expected_size, blob_name in EXPECTED_FILES.values():
        blob = cache_dir / "blobs" / blob_name
        if blob.is_file() and blob.stat().st_size == expected_size:
            downloaded += expected_size
            continue
        part_files = (cache_dir / "blobs").glob(f"{blob_name}.part-*")
        downloaded += sum(path.stat().st_size for path in part_files if path.is_file())
    return min(downloaded, sum(size for size, _ in EXPECTED_FILES.values()))


def report_progress(cache_dir: Path, previous_report: int) -> int:
    total_bytes = sum(size for size, _ in EXPECTED_FILES.values())
    downloaded_bytes = cached_downloaded_bytes(cache_dir)
    percent = downloaded_bytes * 100 // total_bytes
    if percent >= previous_report + 2 or percent == 100:
        logging.info(
            "Model files: %d%% (%0.2f / %0.2f GiB)",
            percent,
            downloaded_bytes / 1024**3,
            total_bytes / 1024**3,
        )
        return percent
    return previous_report


def download_part(
    filename: str,
    part_path: Path,
    start: int,
    end: int,
    total_size: int,
) -> None:
    part_size = end - start + 1
    if part_path.is_file() and part_path.stat().st_size == part_size:
        return
    if part_path.is_file() and part_path.stat().st_size > part_size:
        part_path.unlink()

    url = hf_hub_url(REPOSITORY, filename, revision=REVISION)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        offset = part_path.stat().st_size if part_path.is_file() else 0
        if offset == part_size:
            return
        request_start = start + offset
        try:
            with get_session().stream(
                "GET",
                url,
                headers={"Range": f"bytes={request_start}-{end}"},
                timeout=(30, 180),
            ) as response:
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Range request for {filename} returned HTTP {response.status_code}"
                    )
                expected_range = f"bytes {request_start}-{end}/{total_size}"
                if response.headers.get("content-range") != expected_range:
                    raise RuntimeError(
                        f"Unexpected Content-Range for {filename}: "
                        f"{response.headers.get('content-range')}"
                    )
                with part_path.open("ab") as output:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        output.write(chunk)
                        if part_path.stat().st_size > part_size:
                            raise RuntimeError(f"Range response exceeded its size for {filename}")

            if part_path.stat().st_size == part_size:
                return
            raise RuntimeError(
                f"Range for {filename} ended at {part_path.stat().st_size} of {part_size} bytes"
            )
        except Exception:
            logging.exception(
                "Download chunk failed for %s (attempt %d/%d).",
                filename,
                attempt,
                MAX_ATTEMPTS,
            )
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(min(2**attempt, 30))


def download_file(cache_dir: Path, filename: str, expected_size: int, expected_hash: str) -> Path:
    blobs_dir = cache_dir / "blobs"
    blob = blobs_dir / expected_hash
    if blob.is_file():
        if blob.stat().st_size == expected_size:
            logging.info("Verifying cached model file: %s.", filename)
            verify_file(blob, expected_size, expected_hash)
            return blob
        if blob.stat().st_size != 0:
            raise RuntimeError(
                f"Cached {filename} is {blob.stat().st_size} bytes; expected {expected_size}. "
                "Refusing to overwrite an unverified cache file."
            )
        blob.unlink()
        logging.warning("Removed the confirmed zero-byte cache blob for %s.", filename)

    stale_parts = blobs_dir.glob(f"{expected_hash}.*.incomplete")
    for stale_part in stale_parts:
        if stale_part.is_file():
            stale_part.unlink()
            logging.warning("Removed an abandoned partial cache file for %s.", filename)

    chunk_count = (expected_size + CHUNK_SIZE - 1) // CHUNK_SIZE
    logging.info(
        "Downloading %s in %d resumable ranges (%d workers).",
        filename,
        chunk_count,
        MAX_WORKERS,
    )
    last_progress = -2
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for index in range(chunk_count):
            start = index * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, expected_size) - 1
            part_path = blobs_dir / f"{expected_hash}.part-{index:04d}"
            futures.append(
                executor.submit(
                    download_part,
                    filename,
                    part_path,
                    start,
                    end,
                    expected_size,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            future.result()
            last_progress = report_progress(cache_dir, last_progress)

    assembling = blobs_dir / f"{expected_hash}.assembling"
    try:
        with assembling.open("wb") as output:
            for index in range(chunk_count):
                part_path = blobs_dir / f"{expected_hash}.part-{index:04d}"
                with part_path.open("rb") as part:
                    shutil.copyfileobj(part, output, length=8 * 1024**2)
        verify_file(assembling, expected_size, expected_hash)
        os.replace(assembling, blob)
    except Exception:
        assembling.unlink(missing_ok=True)
        raise

    for index in range(chunk_count):
        (blobs_dir / f"{expected_hash}.part-{index:04d}").unlink(missing_ok=True)
    logging.info("Verified %s size and SHA-256.", filename)
    return blob


def main() -> int:
    log_path = configure_logging()
    cache_dir = Path(HF_HUB_CACHE) / "models--openbmb--VoxCPM2"
    cache_root = cache_dir.parent
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir.joinpath("blobs").mkdir(parents=True, exist_ok=True)

    total_size = sum(size for size, _ in EXPECTED_FILES.values())
    free_bytes = shutil.disk_usage(cache_root).free
    required_bytes = total_size + 10 * 1024**3
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Need 10 GiB plus model files; only {free_bytes / 1024**3:.1f} GiB is free."
        )

    logging.info("Download log: %s", log_path)
    logging.info(
        "Disk available: %.1f GiB; model files total: %.2f GiB",
        free_bytes / 1024**3,
        total_size / 1024**3,
    )
    logging.info("Downloading %s at revision %s.", REPOSITORY, REVISION)

    for filename, (expected_size, expected_hash) in EXPECTED_FILES.items():
        download_file(cache_dir, filename, expected_size, expected_hash)

    model_path = Path(
        snapshot_download(
            repo_id=REPOSITORY,
            revision=REVISION,
            max_workers=4,
            etag_timeout=60,
        )
    )
    for filename, (expected_size, expected_hash) in EXPECTED_FILES.items():
        model_file = model_path / filename
        if not model_file.is_file():
            raise RuntimeError(f"Required model file is missing: {model_file}")
        verify_file(model_file, expected_size, expected_hash)

    logging.info("VoxCPM2 snapshot download and integrity checks succeeded.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("VoxCPM2 download did not complete successfully.")
        raise
