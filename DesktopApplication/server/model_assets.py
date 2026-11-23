from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from server.config import settings


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = str(settings.MODEL_DIR.parent / "hf-home")
os.environ["HF_HUB_CACHE"] = str(settings.MODEL_DIR.parent / "hf-cache")
os.environ["TRANSFORMERS_CACHE"] = str(settings.MODEL_DIR.parent / "transformers-cache")


EXPECTED_MODEL_FILES = {
    ".gitattributes": (1519, "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361"),
    "audiovae.pth": (376_951_122, "94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1"),
    "config.json": (4336, "405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c"),
    "model.safetensors": (
        4_580_080_592,
        "f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d",
    ),
    "README.md": (7903, "feb850b390aa5051b81cbdc73b048361273552f003dd386b4da9c39ec9b6f055"),
    "special_tokens_map.json": (
        1632,
        "068594063e37662c02b21acf42ebb334ef6a74fb810e68a2368f88f08351de76",
    ),
    "tokenization_voxcpm2.py": (
        2895,
        "84489ea32b6ee0cae22ed5480cacb6df85c46624c3119be9a2021c3649a12729",
    ),
    "tokenizer.json": (
        3_676_772,
        "f8984687e4a92a3503d521396d454b7d68e9fdaab2a0288eb3536c7c1aa4bc20",
    ),
    "tokenizer_config.json": (
        5059,
        "e78a3ebb48a0b9437efd1823b6b726c823da89e49dd8bcc90c02419d9baa772b",
    ),
}


class ModelAssetsError(RuntimeError):
    pass


def model_asset_status(
    model_dir: Path | None = None,
    expected_files: dict[str, tuple[int, str]] | None = None,
) -> dict[str, Any]:
    directory = Path(model_dir or settings.MODEL_DIR)
    expected = expected_files or EXPECTED_MODEL_FILES
    if not directory.is_dir():
        return {
            "available": False,
            "directory": str(directory),
            "message": f"VoxCPM2 model files are missing from {directory}. Copy the separate model pack here.",
        }

    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return {
            "available": False,
            "directory": str(directory),
            "message": f"The VoxCPM2 SHA-256 manifest is missing from {directory}.",
        }

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "directory": str(directory),
            "message": f"The VoxCPM2 SHA-256 manifest could not be read: {exc}",
        }
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        return {
            "available": False,
            "directory": str(directory),
            "message": "The VoxCPM2 SHA-256 manifest has an invalid structure.",
        }

    manifest_files = {
        entry.get("path"): entry
        for entry in manifest.get("files", [])
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    problems: list[str] = []
    if set(manifest_files) != set(expected):
        problems.append("The model manifest does not match the expected VoxCPM2 file list.")

    for filename, (expected_size, expected_hash) in expected.items():
        path = directory / filename
        entry = manifest_files.get(filename)
        if not path.is_file():
            problems.append(f"Missing model file: {filename}.")
            continue
        if path.stat().st_size != expected_size:
            problems.append(f"Unexpected size for {filename}.")
        if entry is None or entry.get("size_bytes") != expected_size or entry.get("sha256") != expected_hash:
            problems.append(f"Manifest entry is missing or incorrect for {filename}.")

    return {
        "available": not problems,
        "directory": str(directory),
        "message": "; ".join(problems) if problems else None,
        "total_bytes": sum(size for size, _ in expected.values()),
    }


def verify_model_assets(
    model_dir: Path | None = None,
    expected_files: dict[str, tuple[int, str]] | None = None,
) -> Path:
    directory = Path(model_dir or settings.MODEL_DIR)
    expected = expected_files or EXPECTED_MODEL_FILES
    status = model_asset_status(directory, expected)
    if not status["available"]:
        raise ModelAssetsError(status["message"])

    for filename, (_, expected_hash) in expected.items():
        digest = hashlib.sha256()
        try:
            with (directory / filename).open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise ModelAssetsError(f"Could not verify {filename}: {exc}") from exc
        if digest.hexdigest() != expected_hash:
            raise ModelAssetsError(f"SHA-256 verification failed for VoxCPM2 file {filename}.")
    return directory
