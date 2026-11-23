# KalapakTTS Desktop

This folder is the isolated Windows desktop-app work area. It does not modify the web application in `backend/`, `frontend/`, `admin/`, or `compose.yaml`.

## Stage 0 proof of concept

The proof of concept uses Python `pywebview` with the installed Microsoft Edge WebView2 runtime. Its local FastAPI server binds to loopback on a dynamically selected port and exposes `/health`. The POC window checks that endpoint and verifies that Khmer text is present in the document.

The Next.js UI export is built from a temporary copy of `frontend/`; the build script changes only that copy. The resulting static files are written to `ui/`. This proves exportability, not that every web feature is desktop-ready. Authentication removal, SQLite persistence, TTS integration, packaging, and offline verification belong to later stages.

### Stage 0 results

- **MEASURED — static export:** Next.js 16.4.6 exported successfully to 55 files totaling 3,506,309 bytes (3.3 MiB). The home page, not-found page, icon, Next.js chunks, and six local WOFF2 font files were generated. No API routes or middleware were found. `next/image` works in export mode with `images.unoptimized`.
- **MEASURED — desktop-only build adjustments:** The isolated copy enables `output: "export"`, uses the WebView's origin for the API, disables the copied PWA service-worker registration, and turns off cache components/partial prefetching because Next.js rejects PPR in export mode. The build uses Webpack: Turbopack could not resolve dependencies through the external `node_modules` junction. No web-app source or configuration was edited.
- **MEASURED — pywebview proof:** Python 3.13, pywebview 6.2.1, and WebView2 157.0.4341.60 opened the window. Khmer rendered correctly, and the window displayed a successful local `/health` response. The UTF-8 result is saved in the ignored `build/pywebview-check.json`.
- **MEASURED — NVIDIA impact:** With the existing model already loaded, `nvidia-smi` measured 4,905/6,141 MiB used (1,016 MiB free) both before and while the window was open. WebView2 launched with `--disable-gpu`; its renderer used software WARP. No NVIDIA memory increase was measurable.
- **MEASURED — Vanna reference:** Only `backend/voice_references/vanna.wav` was copied to `models/voices/vanna.wav` (958,564 bytes); its SHA-256 matched the source.
- **MEASURED — tests:** Python syntax compilation succeeded and all three Stage 0 unit tests passed, including serving the exported page and a bundled font from the local API.
- **NOT CONFIRMED — desktop engine path at Stage 0:** The initial copied UI bundle contained browser WebGPU/ONNX code and a Hugging Face URL. The POC did not invoke it; Stage 1 replaces that build input with a desktop-only UI.
- **NOT CONFIRMED — offline operation:** No clean offline build, network-isolation, or outbound-traffic test has been performed. The exported UI is a static build of the web frontend, not yet the final desktop editor; VoxCPM2 is not loaded by this POC and speech generation, history, SQLite, packaging, and installer behavior remain unimplemented.

### Stage 1 — local storage and auth-free desktop UI

- **MEASURED — isolated UI export:** The desktop build exports the replacement UI from `ui-source/` and bundles its fonts locally. The current build produced 39 files totaling 1,177,964 bytes (1.12 MiB). It omits the web frontend's sign-in/session UI and browser WebGPU/ONNX engine. A test scans exported HTML resource attributes, CSS resource URLs, and JavaScript network-call sites for external HTTP(S) requests; it also checks that auth/browser-engine markers are absent.
- **MEASURED — local storage:** The desktop API stores its profile under `%APPDATA%\KalapakTTS` (falling back to the current user's `AppData\Roaming` folder when `APPDATA` is unavailable), with `db/`, `audio/`, `drafts/`, `logs/`, and `settings/` directories. SQLite schema version 1 uses WAL mode, `synchronous=FULL`, foreign-key enforcement, and transaction-wrapped writes. Startup runs `PRAGMA integrity_check`; an existing database is backed up before migration.
- **MEASURED — recovery path:** Corruption puts the UI into recovery-required state, blocks draft/settings operations with a clear error, and offers restoration from the most recent validated SQLite backup. The test suite exercises both the storage layer and the local API recovery flow, including a Windows-safe restore.
- **MEASURED — local profile and CSP:** The desktop UI opens directly into a local editor without sign-in or registration. The API sets a Content Security Policy that restricts resources and connections to the local origin (with inline Next.js hydration/style allowances); external scripts, fonts, and network connections are not permitted.
- **MEASURED — backend isolation:** The desktop server imports from `DesktopApplication/server/` only. Tests inspect its imports and block imports of the repository's `backend` package while importing the desktop app.
- **MEASURED — validation:** All 14 desktop unit tests pass, covering the UI export, external-request scan, CSP, backend import isolation, SQLite persistence/migration/recovery, and the startup VRAM guard.
- **ESTIMATED — VRAM guard:** Startup and current NVIDIA free memory are checked and the UI shows an explicit warning. The current guard threshold is 1,536 MiB, selected above the Stage 0 measurement of 1,016 MiB free while the web TTS model was loaded. This threshold has **not** been validated against loading VoxCPM2; speech generation remains disabled and no desktop model is loaded in this stage. Free-memory readings cannot identify which process owns a loaded model.
- **NOT CONFIRMED — desktop TTS:** The desktop server does not yet load VoxCPM2 or generate speech. No code has been copied from `backend/` during Stage 1.
- **NOT CONFIRMED — offline operation:** The external-request test and CSP are not a network-isolation proof. No firewall-blocked launch or clean-machine offline test has been performed.

#### VoxCPM server-code copy/adaptation plan

No backend files are copied into the desktop server yet. When desktop TTS integration begins, copy/adapt exactly these core source files into the corresponding desktop locations:

| Backend source path | Desktop copy/adaptation path |
| --- | --- |
| `backend/app/core/tts_engine.py` | `DesktopApplication/server/tts_engine.py` |
| `backend/app/core/ffmpeg_audio.py` | `DesktopApplication/server/ffmpeg_audio.py` |
| `backend/app/core/tts_chunking.py` | `DesktopApplication/server/tts_chunking.py` |
| `backend/app/core/audio_diagnostics.py` | `DesktopApplication/server/audio_diagnostics.py` |
| `backend/app/core/noise_reduction.py` | `DesktopApplication/server/noise_reduction.py` |
| `backend/app/core/tts_jobs.py` | `DesktopApplication/server/tts_jobs.py` |

This is the complete planned copy list, not a runtime dependency. Desktop configuration, SQLite persistence, and local API routes remain desktop-owned; do not copy auth, PostgreSQL, or web API modules. For each sync, record the source Git commit and SHA-256 values, compare upstream changes against the desktop-adapted copies, keep desktop-specific edits separate, and run desktop tests with repository-backend imports blocked. The isolation test must continue to pass.

### Developer setup

From `DesktopApplication/`, create a development-only virtual environment and install the POC dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Build the static UI using the repository's existing frontend dependencies:

```powershell
.\app\build_ui.ps1
```

Start the POC shell:

```powershell
python .\app\launcher.py
```

Run its local API check:

```powershell
python -m unittest discover -s tests -v
```

The development environment depends on this checkout and the machine's installed Python. It is not the final portable runtime or installer.

## Local assets

`models/voices/vanna.wav` is a local reference asset and is intentionally excluded from Git. Model weights, generated audio, user profiles, and packaging outputs are also excluded. Do not copy the dataset or the web application's PostgreSQL data into this folder.

## Offline status

**Offline operation is not yet proven.** This POC does not load VoxCPM2 or generate speech. Do not use it as the completed desktop application.
