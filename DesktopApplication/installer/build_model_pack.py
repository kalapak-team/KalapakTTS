from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

DESKTOP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DESKTOP_ROOT))

from server.model_assets import EXPECTED_MODEL_FILES, model_asset_status  # noqa: E402


MODEL_ROOT = DESKTOP_ROOT / "models"
PACK_OUTPUT = DESKTOP_ROOT / "build" / "KalapakTTS-model-pack"
VANNA_SHA256 = "a97a322277d2ab8ad0dc7346f57a499392bf5b5eb4daf6bd6049ecab611909dd"
PACK_FILES = (
    DESKTOP_ROOT / "installer" / "install_model_pack.ps1",
    DESKTOP_ROOT / "installer" / "MODEL-PACK-README.txt",
    DESKTOP_ROOT / "installer" / "MODEL-LICENSE-APACHE-2.0.txt",
    DESKTOP_ROOT / "installer" / "VANNA-VOICE-NOTICE.txt",
)


def _copy_and_hash(source: Path, destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as source_stream, destination.open("wb") as target_stream:
        for block in iter(lambda: source_stream.read(8 * 1024 * 1024), b""):
            target_stream.write(block)
            digest.update(block)
            size += len(block)
    shutil.copystat(source, destination)
    return size, digest.hexdigest()


def build_model_pack(
    model_root: Path = MODEL_ROOT,
    output: Path = PACK_OUTPUT,
) -> dict[str, int]:
    model_root = model_root.resolve()
    output = output.resolve()
    model_dir = model_root / "VoxCPM2"
    status = model_asset_status(model_dir)
    if not status["available"]:
        raise RuntimeError(f"VoxCPM2 assets are not ready: {status['message']}")
    if output.exists():
        raise FileExistsError(f"Model-pack output already exists: {output}")

    expected_paths = {
        *(f"VoxCPM2/{filename}" for filename in EXPECTED_MODEL_FILES),
        "VoxCPM2/manifest.json",
        "voices/vanna.wav",
    }
    source_paths = {
        path.relative_to(model_root).as_posix()
        for path in model_root.rglob("*")
        if path.is_file()
    }
    if source_paths != expected_paths:
        raise RuntimeError(
            "The local model directory contains unexpected or missing files; "
            "refusing to build an incomplete or expanded pack."
        )

    output.mkdir(parents=True)
    entries: list[dict[str, object]] = []
    try:
        for relative_path in sorted(expected_paths):
            source = model_root / Path(relative_path)
            destination = output / Path(relative_path)
            if source.is_symlink():
                raise RuntimeError(f"Symbolic links are not allowed in model packs: {relative_path}")
            size, digest = _copy_and_hash(source, destination)
            expected = EXPECTED_MODEL_FILES.get(Path(relative_path).name)
            if relative_path.startswith("VoxCPM2/") and Path(relative_path).name in EXPECTED_MODEL_FILES:
                if expected != (size, digest):
                    raise RuntimeError(f"Model file verification failed: {relative_path}")
            if relative_path == "voices/vanna.wav" and digest != VANNA_SHA256:
                raise RuntimeError("The Vanna preset reference audio SHA-256 did not match.")
            entries.append(
                {"path": relative_path, "size_bytes": size, "sha256": digest}
            )

        manifest = {
            "schema_version": 1,
            "model_repository": "openbmb/VoxCPM2",
            "model_snapshot": "7936080b41163874f38610f46f438e2c0757f9fd",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "files": entries,
        }
        manifest_path = output / "manifest.json"
        manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
        manifest_path.write_bytes(manifest_bytes)
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        (output / "manifest.sha256").write_text(
            f"{manifest_hash}  manifest.json\n",
            encoding="ascii",
        )
        for support_file in PACK_FILES:
            shutil.copy2(support_file, output / support_file.name)
    except Exception:
        shutil.rmtree(output)
        raise

    payload_bytes = sum(int(entry["size_bytes"]) for entry in entries)
    pack_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    return {"payload_bytes": payload_bytes, "pack_bytes": pack_bytes}


if __name__ == "__main__":
    sizes = build_model_pack()
    print(f"Verified model payload: {sizes['payload_bytes']} bytes")
    print(f"Model pack folder: {PACK_OUTPUT} ({sizes['pack_bytes']} bytes)")
