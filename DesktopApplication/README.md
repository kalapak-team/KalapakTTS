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

- **MEASURED — isolated UI export:** The desktop build exports the replacement UI from `ui-source/` and bundles its fonts locally. The current build produced 39 files totaling 1,186,482 bytes (1.13 MiB). It omits the web frontend's sign-in/session UI and browser WebGPU/ONNX engine. A test scans exported HTML resource attributes, CSS resource URLs, and JavaScript network-call sites for external HTTP(S) requests; it also checks that auth/browser-engine markers are absent.
- **MEASURED — local storage:** The desktop API stores its profile under `%APPDATA%\KalapakTTS` (falling back to the current user's `AppData\Roaming` folder when `APPDATA` is unavailable), with `db/`, `audio/`, `drafts/`, `logs/`, and `settings/` directories. SQLite schema version 1 uses WAL mode, `synchronous=FULL`, foreign-key enforcement, and transaction-wrapped writes. Startup runs `PRAGMA integrity_check`; an existing database is backed up before migration.
- **MEASURED — recovery path:** Corruption puts the UI into recovery-required state, blocks draft/settings operations with a clear error, and offers restoration from the most recent validated SQLite backup. The test suite exercises both the storage layer and the local API recovery flow, including a Windows-safe restore.
- **MEASURED — local profile and CSP:** The desktop UI opens directly into a local editor without sign-in or registration. The API sets a Content Security Policy that restricts resources and connections to the local origin (with inline Next.js hydration/style allowances); external scripts, fonts, and network connections are not permitted.
- **MEASURED — backend isolation:** The desktop server imports from `DesktopApplication/server/` only. Tests inspect its imports and block imports of the repository's `backend` package while importing the desktop app.
- **MEASURED — validation:** The Stage 1 snapshot passed 14 tests. The current combined desktop suite has 27 tests, covering the UI export, external-request scan, CSP, backend import isolation, SQLite persistence/migration/recovery, GPU calibration and engine behavior.
- **HISTORICAL / SUPERSEDED — Stage 1 VRAM guard:** The initial 1,536 MiB threshold was an estimate and was not a model-load measurement. Stage 5 replaced it with a measured peak-reserved requirement and explicit context headroom below.
- **HISTORICAL — desktop TTS:** Speech generation was not implemented in the Stage 1 snapshot. Stage 5 results below supersede that status.
- **NOT CONFIRMED — offline operation:** The external-request test and CSP are not a network-isolation proof. No firewall-blocked launch or clean-machine offline test has been performed.

#### VoxCPM server-code copy/adaptation record

The desktop server vendors/adapts the following six core files from the web backend. Hashes are SHA-256 of the current desktop copies; source hashes are included to make future sync reviews reproducible. Source revision: `7148be87d57c148f34d062b5ace23a7382b4e58d`.

| Backend source path | Desktop path | Source SHA-256 | Desktop SHA-256 |
| --- | --- | --- | --- |
| `backend/app/core/tts_engine.py` | `server/tts_engine.py` | `3806bf00847bc9d5a361144cdc60ec5001b0c829f91a7910360c9ad71b2c709d` | `e28bcccee2e23ea15aae7224f5f3ed3a5030dfb3d34c66a5bbdeebac6c73f274` |
| `backend/app/core/ffmpeg_audio.py` | `server/ffmpeg_audio.py` | `2b23bc9c82ad145e9c0b7283a5a9a8ab70a813d7a7ed303ac362e31d98a4177d` | `de0f75f26b9a08fc35fdc88740885b186c82898d37a68f381e69b9c6c24e51cf` |
| `backend/app/core/tts_chunking.py` | `server/tts_chunking.py` | `48eb4b4748cc105a603729979f7fe69730888dbe05a33a133f644c3277b7cbf1` | `aa26193a2569ba7910d8f45cd4bff0e8e505e866cb23699607e57d18c353d745` |
| `backend/app/core/audio_diagnostics.py` | `server/audio_diagnostics.py` | `4040a83af187b92129284b1a074edea6013b91acd4b1b41a2c843911957e3edf` | `4040a83af187b92129284b1a074edea6013b91acd4b1b41a2c843911957e3edf` |
| `backend/app/core/noise_reduction.py` | `server/noise_reduction.py` | `6ec37c353fd0b780a3589ffc2e3bfdee3f60855f007ca2eac06d3f952c4147d2` | `43b5a6bd93d9cdc89651a10681b618daac387bb1828268c1a63ad4f87191219a` |
| `backend/app/core/tts_jobs.py` | `server/tts_jobs.py` | `988f9521e10f91781d330a5fa997a465aab43ccd6a85dd7205c67ce2fa848e9e` | `bfb9fb71964369630d0bcb5831a7dd8bdfbd535450bd0e9fc94c3b5721b9d773` |

Five copies are adapted (only `audio_diagnostics.py` is byte-identical). To keep them in sync, compare these files against the listed backend paths at each source-revision change, review differences rather than overwriting desktop changes, update both hashes and the source revision here, and run the desktop suite. Desktop configuration, SQLite persistence, and API routes remain local; no backend module is imported at runtime. The isolation test blocks the repository `backend` package.

### Stage 2 — history, drafts and local media

- **MEASURED — SQLite migration:** Schema version 3 adds history indexes and upgrades existing profiles with a pre-migration backup. History metadata, settings, and audio paths remain in local storage.
- **MEASURED — history operations:** The API/UI provide text/title search, date/voice/pace/steps filters, favorites, rename, confirmed deletion, duplicate (including a separate audio file), playback, WAV/SRT/VTT downloads, and Open / Regenerate. Missing audio is identified and the original text/settings can be sent back to the editor for regeneration.
- **MEASURED — draft and preferences:** Text and voice/pace/steps/speed settings autosave after a short idle delay and on page hide; the last draft is restored at startup. Khmer is the current fixed engine language and is stored in the draft settings. The pywebview launcher restores and persists window dimensions using its verified resize event, with safe minimum/maximum bounds.
- **MEASURED — media management:** Storage status reports history count and WAV storage size. Users can delete history/audio before a selected date. Changing the media folder copies files first, updates database paths transactionally, and removes old copies only after the update; path collisions with different file contents stop the move without changing history. The configured folder persists across app restart.
- **MEASURED — validation:** Static UI build and TypeScript checks succeeded; the combined desktop suite has 35 passing tests, including filtering/actions, cleanup, safe media moves, migration, and window-size persistence.
- **NOT CONFIRMED — desktop interaction:** The Python/API/unit tests and static export validate the behavior, but the history panels, debounce autosave, pywebview resize event, and page-hide save have not been manually exercised in a live window. Crash/reboot persistence and large-media/low-disk behavior remain for Stage 8 validation.

### Stage 5 — portable VoxCPM2 runtime and GPU measurements

- **MEASURED — independent runtime:** The app launches with `DesktopApplication/runtime/python/python.exe` (Python 3.12.10), Torch 2.11.0+cu128, CUDA 12.8 and BF16 support on the RTX 4050. `PYTHONPATH`/`PYTHONHOME` and user-site imports are disabled by the launcher; model paths resolve only under `DesktopApplication/models`. HF offline environment flags are set before importing the server. No import from `backend/` is needed.
- **MEASURED — size:** `runtime/` is 5,494,473,183 bytes (5.117 GiB); `models/` is 4,961,691,976 bytes (4.621 GiB); exported `ui/` is 1,186,482 bytes (1.13 MiB). The VoxCPM2 weights are `model.safetensors` (4,580,080,592 bytes, SHA-256 `f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d`) and `audiovae.pth` (376,951,122 bytes, SHA-256 `94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1`). The local manifest verifies model assets before load. Torch is the largest runtime dependency (about 4.12 GiB); the bundled FFmpeg binary is a 222 MB GPL build, so distribution needs an accompanying-license/source-compliance decision.
- **MEASURED — CUDA calibration before model load:** On this Windows/WDDM session, Torch reported 5,080 MiB free both before and after context initialization (Torch delta 0 MiB). `nvidia-smi` changed from 5,920 MiB to 5,848 MiB free (measured context overhead 72 MiB). The separate `nvidia-smi`/Torch discrepancy was 840 MiB at startup and 768 MiB on subsequent reads; it is not mislabelled as context overhead.
- **MEASURED — model memory and guard:** With the desktop window open and the web TTS/KamVox not running, the longest fixture reached `torch.cuda.max_memory_reserved()` of 5,551,161,344 bytes (5,294 MiB). The app guard is set to that measured model peak plus 72 MiB measured context overhead: 5,366 MiB free by `nvidia-smi`. Idle after model load: 5,209 MiB used / 712 MiB free. During the long fixture: 5,395 MiB peak used / 526 MiB minimum free. The earlier web-server number of 5,395 MiB used is consistent as a whole-GPU peak-used reading, not a free-memory requirement. The UI now labels the `nvidia-smi`/Torch gap and reports GPU free memory from `nvidia-smi`.
- **MEASURED — GPU run:** Desktop startup to API-ready 3.79 s; model-load request 18.21 s (engine-reported load 12.32 s). The 49-character short sample: first audio 11.12 s, total 19.93 s, 3.81 s audio, RTF 5.23. The 1,094-character `tests/fixtures/khmer_long_sample.txt`: first audio 5.96 s, total 141.74 s, 76.54 s audio, RTF 1.85. Peak process RAM was 15.53 GB; process working set after the long generation was 10.55 GB. WAV, SRT and VTT export endpoints returned files and the WAV SHA-256 matched the saved generation.
- **MEASURED — user benchmark comparison:** User-supplied web GPU benchmark at 4 steps: 191 s total, 7.8 s to first audio, RTF 2.6. Desktop long fixture at the same 4 steps was 141.74 s total (about 49.3 s faster), 5.96 s to first audio, RTF 1.85. These are separate runs and fixtures/environment may differ; treat the comparison as indicative, not a controlled A/B.
- **MEASURED — CPU fallback path:** Explicit CPU model load took 17.58 s and the short sample took 50.88 s total, first audio at 17.68 s, RTF 12.98. Peak process RAM was 16.43 GB. No NVIDIA model allocation was observed during CPU generation. The app displays and persists a clear fallback reason. The actual CUDA-OOM retry is covered by a simulated unit test; a real OOM was not forced.
- **MEASURED — features and tests:** Local VoxCPM2 streaming generation uses the existing chunker, Vanna reference preset, Khmer pronunciation dictionary, Relaxed/Natural/Brisk CPS targets, 4/8/12 step presets, pitch-preserving speed, single-job queue and WAV/SRT/VTT exports. CUDA load and generation OOM handlers retry on CPU; the UI receives a fallback event. The static export and external-runtime-URL check pass, and all 28 desktop unit tests pass.
- **ESTIMATED — runtime portability:** This is a portable runtime inside the project folder, not an installer yet. Size and successful launch are measured on this PC, but running after renaming/removing the repository and from a clean install has not been tested.
- **NOT CONFIRMED — offline proof and other Stage 5 cases:** No firewall-blocked/network-disconnected test has been done. No clean-machine runtime test, below-threshold GPU load, real CUDA OOM, GPU-busy CPU fallback, crash recovery, reboot persistence, or user-performed audio listening review has been completed. Do not claim 100% offline.
- **NOT CONFIRMED — RAM and feature edge cases:** Windows process working-set telemetry is measured; total system peak RAM attributable only to this app is not isolated. Long-form CPU generation was not run.

Benchmark records are generated locally at `build/stage5-benchmark.json` and `build/stage5-cpu-benchmark.json` and are ignored build artifacts, not committed. Re-run the harness from a desktop API session with the UI open:

```powershell
.\runtime\python\python.exe .\tests\run_stage5_benchmark.py --base-url http://127.0.0.1:<port> --device cuda
```

Keep the web TTS server and KamVox stopped while measuring. The harness limits model loading to 180 seconds and GPU generation to 420 seconds (CPU: 600/900 seconds). Use `--try-gpu-anyway` only for an explicitly controlled below-threshold GPU experiment.

### Developer setup

From `DesktopApplication/`, use the prepared app-local runtime to run the desktop app:

```powershell
.\app\run_desktop.ps1
```

Build the static UI using the repository's existing frontend dependencies (this build step uses the web frontend's installed Node dependencies):

```powershell
.\app\build_ui.ps1
```

Run the desktop tests:

```powershell
.\runtime\python\python.exe -m unittest discover -s tests -v
```

The portable runtime is included only in the local working tree and ignored by Git. Installer generation and clean-machine installation are still pending.

## Local assets

`models/voices/vanna.wav` is a local reference asset and is intentionally excluded from Git. Model weights, generated audio, user profiles, and packaging outputs are also excluded. Do not copy the dataset or the web application's PostgreSQL data into this folder.

## Offline status

**100% offline operation is not proven.** TTS runs locally from the app-local runtime and model, and the exported UI has no external runtime URLs; however, no firewall-blocked, network-disconnected, clean-install test has been performed. Stage 8 is still required before claiming offline operation.
