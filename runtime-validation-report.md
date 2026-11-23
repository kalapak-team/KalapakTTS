# Runtime Validation Report

**Generated**: 2026-11-22T16:30:03+07:00  
**Target**: `<repo>`

## Integrated PyTorch server-engine validation

**Updated**: 2026-11-22T16:30:03+07:00  
**Overall**: MEASURED GPU/CPU generation, startup, and browser E2E passed. The FastAPI process is currently running on the RTX 4050 in `DEVICE=auto` mode. The ONNX/WebGPU full-generation track remains paused. A real CUDA OOM and optimization speedups remain NOT CONFIRMED.

| Area | Result | Evidence / limits |
|---|---|---|
| Startup / selected engine | MEASURED | Startup took 16.69 s to load the warm model. `torch 2.11.0+cu128`, CUDA 12.8, RTX 4050 Laptop GPU, BF16, `AudioVAE=cpu`; `DEVICE=auto`, `FORCE_CUDA=false`, `MIN_FREE_VRAM_GIB=4.5` selected CUDA. Startup log: [`api-auto-startup-20261122.log`](backend/audio_files/benchmarks/rtx4050/api-auto-startup-20261122.log). `/health` returned 200 and `/api/tts/status` reported `model_loaded=true`. |
| Real GPU memory | MEASURED | `nvidia-smi`: 6,141 MiB total; at idle after warm load, 5,209 MiB used / 712 MiB free. During the identical long-fixture GPU runs, peak was 5,395 MiB used (746 MiB below the reported total); a separate E2E sample peaked at 5,453 MiB. PyTorch allocation and global NVIDIA readings differed, so the peak column below uses `nvidia-smi`. |
| VRAM guard | MEASURED / CONFIGURED | `MIN_FREE_VRAM_GIB` now defaults to 4.5 GiB in code and `.env.example`. The startup log measured 4.96 GiB free before loading, about 0.46 GiB over the threshold. This pre-load threshold was tested with `DEVICE=auto`; it is not an OOM guarantee. |
| Model configuration | MEASURED | Model loaded once and kept warm, BF16 on CUDA, `load_denoiser=False`, AudioVAE on CPU. The server runs generation under `torch.inference_mode()` and empties CUDA cache between chunks. |
| GPU-vs-CPU long fixture | MEASURED | Same 1,097-character request (1,093 non-newline code points plus four paragraph separators), seed `20261119`, Vanna reference, 14 CPS target, chunking, pronunciation dictionary, and steps 4/6/10. The next table contains generation time, first-audio latency, RTF, estimated speech CPS, pauses, pace retries, and VRAM. |
| Process RAM | PARTIAL | CPU process peak working set was 15.132 GiB; host free RAM reached 3.663 GiB at 10 steps. GPU process peak RAM was not captured. |
| Browser E2E | PASS, exit code 0 | `test_signup_character_counter_generation_and_audio_playback`: 1 passed in 111.694 s against the CUDA API; signup, settings, generation stream, and audio playback were exercised. This did not force a real OOM. |
| Backend device/OOM tests | PASS, exit code 0 | `python -m unittest tests.test_tts_engine_device -v` — 4 passed after the VRAM default change, including the mocked OOM retry on CPU. |
| Frontend lint/build | PASS, exit code 0 | `npm run lint` and `npm run build` previously passed for the unchanged frontend. |
| Real OOM fallback | NOT CONFIRMED | GPU pressure was raised until `nvidia-smi` showed 41 MiB free, but synthesis still completed on CUDA. The mocked fallback test passed; actual OOM-to-CPU behavior and its live UI message were not observed. |
| Speed experiments | NOT CONFIRMED | SDPA/Flash, `torch.compile`, smaller initial chunk, AudioVAE-on-GPU, and retry-limit changes were not A/B-tested. No speedup is claimed. |
| Short 35-character case | NOT CONFIRMED | The earlier WAV named `gpu-auto-35char-steps4.wav` contains 33 characters, not 35; do not treat it as a valid 35-character benchmark. |

### Identical long-fixture server benchmark

`Total / RTF` uses the API's measured request generation duration divided by the final WAV duration. First-audio latency is the first completed stream chunk. The speech-only CPS values are estimated from RMS activity and are not forced-aligned; per-third estimates distribute the 1,093 non-newline code points evenly across the audio thirds. The API's overall CPS metric counts all 1,097 request characters, including the four paragraph separators.

| Engine | Steps | Audio duration | Total / RTF | First audio | Overall speech CPS | First / middle / last third CPS (estimated) | Pauses >=250 ms | Pace retries | Peak VRAM (`nvidia-smi`) | Peak host process RAM |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| RTX 4050 | 4 | 72.15 s | 190.64 s / 2.64 | 7.78 s | 16.32 | 16.08 / 16.39 / 15.77 | 7 | 2 | 5,395 MiB | NOT RECORDED |
| RTX 4050 | 6 | 75.46 s | 251.92 s / 3.34 | 9.34 s | 15.37 | 15.50 / 15.38 / 15.01 | 5 | 3 | 5,395 MiB | NOT RECORDED |
| RTX 4050 | 10 | 74.96 s | 264.71 s / 3.53 | 16.83 s | 15.37 | 15.08 / 15.11 / 15.45 | 6 | 3 | 5,395 MiB | NOT RECORDED |
| CPU | 4 | 72.64 s | 1,106.03 s / 15.22 | 41.74 s | 16.06 | 15.82 / 16.50 / 15.55 | 7 | 3 | 0 MiB | 15.132 GiB |
| CPU | 6 | 73.55 s | 1,164.85 s / 15.84 | 45.44 s | 15.70 | 15.53 / 15.63 / 15.59 | 6 | 2 | 0 MiB | 15.132 GiB |
| CPU | 10 | 73.90 s | 1,970.38 s / 26.66 | 113.25 s | 15.66 | 15.47 / 15.70 / 15.53 | 6 | 4 | 0 MiB | 15.132 GiB |

Activity-derived pauses per first/middle/last third were GPU 4 steps `2/3/1`, GPU 6 `2/2/1`, GPU 10 `2/1/3`; CPU 4 `2/4/1`, CPU 6 `2/1/2`, CPU 10 `2/2/1`. Corresponding estimated >4 kHz energy ratios by third were GPU 4 `5.66%/5.48%/2.55%`, GPU 6 `4.97%/3.67%/2.07%`, GPU 10 `4.80%/4.33%/2.21%`; CPU 4 `5.07%/5.71%/2.60%`, CPU 6 `4.40%/4.37%/2.03%`, CPU 10 `3.94%/5.60%/1.98%`. These spectral ratios are descriptive features, not clarity scores.

GPU generation was approximately 5.80x / 4.62x / 7.44x faster than CPU at 4/6/10 steps. The measured WAV pairs are [`gpu-natural14-seed20261119-steps4.wav`](backend/audio_files/benchmarks/rtx4050/gpu-natural14-seed20261119-steps4.wav), [`gpu-natural14-seed20261119-steps6.wav`](backend/audio_files/benchmarks/rtx4050/gpu-natural14-seed20261119-steps6.wav), [`gpu-natural14-seed20261119-steps10.wav`](backend/audio_files/benchmarks/rtx4050/gpu-natural14-seed20261119-steps10.wav), and the corresponding [`CPU 4-step WAV`](backend/audio_files/benchmarks/rtx4050/cpu-natural14-seed20261119-steps4.wav), [`CPU 6-step WAV`](backend/audio_files/benchmarks/rtx4050/cpu-natural14-seed20261119-steps6.wav), and [`CPU 10-step WAV`](backend/audio_files/benchmarks/rtx4050/cpu-natural14-seed20261119-steps10.wav). Compare Khmer clarity, English words, Vanna identity, chunk joins, and pace by ear; the agent cannot hear audio.

Historical unchunked/unpaced output durations were 48.32/36.48/40.16 seconds at 4/6/10 steps because autoregressive generation produced 302/228/251 patches and different stop positions. Step count is not a duration control. The integrated path uses chunking and pace normalization, resulting in more consistent ~72–75 second files. The raw-model pace estimates were 22.80/30.11/27.82 CPS before site pace processing.

### Historical raw-generation context (not the integrated benchmark)

The earlier RTX 4050 BF16 unchunked/unpaced fixture runs generated 302, 228, and 251 autoregressive patches at 4, 6, and 10 DiT steps respectively. Their measured audio durations were 48.32, 36.48, and 40.16 seconds. These differences track the model's variable autoregressive patch/stop-token output, not just DiT step count. The corresponding historical speech-only pace estimates were 22.80, 30.11, and 27.82 characters/second; they are RMS-activity estimates from raw audio, before the current website pace and global-speed controls. These values do not predict the new chunked path. The historical peak allocated VRAM was 5.657 GiB with AudioVAE on CPU; see the detailed table below.

## Prior ONNX/reference-engine diagnostics (captured before server integration)

**Updated**: 2026-11-22  
**Overall**: MEASURED PyTorch generation and ONNX VAE-decoder parity succeeded with the AudioVAE offloaded to CPU. The full autoregressive ONNX/INT4 generator is still NOT IMPLEMENTED/NOT CONFIRMED, so these WAVs do not establish end-to-end ONNX quality. At the time of those tests, the API/server was not started or changed; later server validation is summarized above.

### Public sampler and corrected runner

The sampler was checked against the pinned [OpenBMB `unified_cfm.py`](https://github.com/OpenBMB/VoxCPM/blob/5e293dee2f4745d93ff6459886121467b41aaf1b/src/voxcpm/modules/locdit/unified_cfm.py#L51-L136) and [VoxCPM2 `_inference`](https://github.com/OpenBMB/VoxCPM/blob/5e293dee2f4745d93ff6459886121467b41aaf1b/src/voxcpm/model/voxcpm2.py#L998-L1133). `sway_sampling_coef=1`, no separate t-shift, one CFG-zero* zero-init step for each tested schedule, noise shape `[1,64,4]` generated after upstream `apply_generation_seed` on the model device, `cfg_value=2`, and `mean_mode=false` (the DiT's `dt` input is zeroed while the outer Euler update uses the schedule `dt`). Each generated patch conditions the next patch.

The runner now computes BF16 `t/dt` in CUDA and passes those exact values as FP32 to ONNX. Values below are measured CUDA BF16 schedules; ONNX receives the same values. Detailed JSON: [`stage-e-sampler-schedule.json`](backend/model_cache/onnx-parity-artifacts/stage-e-sampler-schedule.json).

| Steps | `t` by step | Euler `dt` by step |
|---|---|---|
| 4 | 1, .8828, .7070, .4258 | .1172, .1758, .2813, .4258 |
| 6 | 1, .9297, .8281, .7070, .5352, .2988 | .0703, .1016, .1211, .1719, .2363, .2988 |
| 10 | 1, .9531, .9102, .8516, .7891, .7070, .6094, .4922, .3496, .1885 | .0469, .0430, .0586, .0625, .0820, .0977, .1172, .1426, .1611, .1885 |

The earlier seed-123 6-step failure at step 5 (cosine .9434) did **not** reproduce after correcting the staged runner to use upstream device-local seeding and patch-to-patch conditioning; it is not now considered a confirmed INT4 failure for that seed. The exact contribution of those two corrections was not isolated. The original Euler schedule expression was verified unchanged against upstream.

Teacher-forced ONNX `dit_step` comparisons use each PyTorch `x_t` and condition. All were finite, and every tested 4/6/10-step comparison exceeded cosine .99:

| Steps | Teacher-forced cosine mean | Minimum |
|---|---:|---:|
| 4 | .999428 | .997390 |
| 6 | .999786 | .998992 |
| 10 | .999928 | .999780 |

Free-running comparison covers 2 patches per test (the same fixed `dit_hidden` is used for both, with generated patch conditioning), 5 seeds, and the final latent after patch 2. This is a DiT-only trajectory diagnostic, **not** a full autoregressive LM/stop-token run.

| Seed | 4 steps | 6 steps | 10 steps |
|---:|---:|---:|---:|
| 123 | .997124 | .998322 | .996876 |
| 1234 | .990851 | .962902 | .936281 |
| 2025 | .894993 | .914041 | .978860 |
| 20261121 | .976065 | .843584 | .981930 |
| 20261122 | .891137 | .909847 | .994136 |
| Mean / minimum | .950034 / .891137 | .925739 / .843584 | .977617 / .936281 |

**Interpretation:** teacher-forced individual steps are close, but free-running error varies substantially by seed and step count; several final-latent cosines are below .98. This remains a quantization/trajectory risk, not proof of final audio failure. See [`stage-e-onnx-results.json`](backend/model_cache/onnx-parity-artifacts/stage-e-onnx-results.json) and [`stage-e-reference.npz`](backend/model_cache/onnx-parity-artifacts/stage-e-reference.npz).

### Stage (f): RTX 4050 PyTorch generation and VAE-only ONNX comparison

**MEASURED** on the RTX 4050 Laptop GPU, BF16, fixed seed 123, Vanna reference WAV, `load_denoiser=False`, using upstream streaming generation. The full PyTorch VoxCPM2 model loaded in 24.601 s. To fit, AudioVAE was explicitly offloaded to CPU; CUDA allocation after that offload was 4.594 GiB (0.26 GiB reported free), and peak allocated GPU memory during long-text generation was 5.657 GiB. First audio latency and total generation time below include the measured run; `RTF = generation seconds / audio seconds`. The per-third character rates are **estimates** based on evenly assigning the text across thirds, not forced alignment. These raw generations do not apply the website pace/global-speed post-processing.

| Text | Steps | First audio | Total / RTF | Audio duration | Patches / stop position | Estimated speech-only CPS by third | Pauses >=250 ms by third | >4 kHz energy ratio by third |
|---|---:|---:|---:|---:|---:|---|---|---|
| 35 chars | 4 | 7.12 s | 12.51 s / 4.34 | 2.88 s | 18 / 18 | 12.15 / 12.15 / 12.28 | 0 / 0 / 0 | 39.45% / 9.21% / 9.23% |
| 35 chars | 6 | 3.53 s | 8.68 s / 3.19 | 2.72 s | 17 / 17 | 12.87 / 12.87 / 14.06 | 0 / 0 / 0 | 18.42% / 8.05% / 2.61% |
| 35 chars | 10 | 3.83 s | 9.58 s / 3.74 | 2.56 s | 16 / 16 | 13.67 / 13.67 / 13.67 | 0 / 0 / 0 | 15.26% / 7.45% / 3.44% |
| 1,093-char fixture | 4 | 20.32 s | 100.39 s / 2.08 | 48.32 s | 302 / 302 | 22.80 / 22.62 / 22.77 | 0 / 0 / 0 | 7.65% / 8.19% / 4.22% |
| 1,093-char fixture | 6 | 20.28 s | 88.92 s / 2.44 | 36.48 s | 228 / 228 | 30.11 / 29.96 / 30.16 | 0 / 0 / 0 | 9.40% / 9.82% / 4.30% |
| 1,093-char fixture | 10 | 20.38 s | 114.01 s / 2.84 | 40.16 s | 251 / 251 | 27.82 / 27.59 / 27.32 | 0 / 0 / 0 | 6.99% / 6.80% / 3.56% |

RMS for the six PyTorch outputs was respectively `-27.07`, `-22.12`, `-21.35`, `-22.83`, `-20.97`, and `-21.14 dBFS`; overall >4 kHz ratios were `20.01%`, `10.72%`, `10.30%`, `6.50%`, `7.56%`, and `5.61%`. Activity/pause detection is an RMS heuristic. These measurements do not certify intelligibility or quality; the long fixture rates are faster than the user pace target and should be compared by ear.

The paired ONNX WAVs were produced by feeding the **PyTorch-generated latents** to the public Uddom `vae_decoder` graph, not by running the ONNX autoregressive generator. All six ONNX decoder outputs matched the PyTorch streamed WAV dimensions and duration; relative waveform L2 was `1.08e-6`–`2.08e-6`, cosine `>0.99999999999`. This confirms VAE decoder parity only. ONNX full-generation duration, patch/stop position, quality, and same-seed audio are **NOT CONFIRMED**.

Comparison files are under [`stage-f-audio/`](backend/model_cache/onnx-parity-artifacts/stage-f-audio/); repeatable harness: [`benchmark_voxcpm_full_pipeline.py`](backend/scripts/benchmark_voxcpm_full_pipeline.py). Listen for Khmer clarity, English words, voice identity, joins, speed, and artifacts. The agent cannot hear the WAVs.

### WebGPU status at time of those diagnostics

| Item | Status |
|---|---|
| Server/API | NOT STARTED DURING THESE ONNX TESTS; the integrated PyTorch API was tested later and is summarized above. |
| Stage 1 sampler and DiT diagnostics | MEASURED; all teacher-forced steps >=.99, but free-running final latents vary by seed and often fall below .98. |
| Stage (f) | MEASURED PyTorch outputs and VAE-only ONNX decoder comparison; not full ONNX generation. |
| Stages 2–4 (Web Worker/WebGPU, resume/offline, browser GPU benchmarks) | NOT RUN. Full ONNX free-running output has not been validated; no browser output or WebGPU memory/speed/Khmer quality claim is made. |
| 3 GiB VRAM target / RTX 4050 peak | PyTorch full model needed VAE CPU offload and peaked at 5.657 GiB allocated; this does not establish that the Uddom browser model fits 3 GiB. |
| Vanna support in Uddom browser engine | NOT CONFIRMED; the PyTorch reference voice was used only for the measured baseline. |
| Khmer listening/ASR quality | NOT CONFIRMED. |
| PWA offline generation, download resume, local history/WAVs | NOT VERIFIED / NOT COMPLETE. |

**Public source and license notes:** Uddom `ad57fe543ef28fab07741c8603035e13ee68f770` is Apache-2.0; its [LICENSE](https://huggingface.co/Uddom/voxcpm2-khmer-webgpu/blob/ad57fe543ef28fab07741c8603035e13ee68f770/LICENSE) and [NOTICE](https://huggingface.co/Uddom/voxcpm2-khmer-webgpu/blob/ad57fe543ef28fab07741c8603035e13ee68f770/NOTICE) are retained. The model card reports 1,937,476,083 bytes, but does not confirm this RTX 4050, 3 GiB memory, Khmer ASR/intelligibility, or Vanna compatibility. Upstream OpenBMB source commit `5e293dee2f4745d93ff6459886121467b41aaf1b` is Apache-2.0; generation references are the pinned [`_inference` and patch-conditioning loop](https://github.com/OpenBMB/VoxCPM/blob/5e293dee2f4745d93ff6459886121467b41aaf1b/src/voxcpm/model/voxcpm2.py#L998-L1133), [seed helper](https://github.com/OpenBMB/VoxCPM/blob/5e293dee2f4745d93ff6459886121467b41aaf1b/src/voxcpm/model/utils.py#L27-L37), [CFM schedule/CFG-zero*](https://github.com/OpenBMB/VoxCPM/blob/5e293dee2f4745d93ff6459886121467b41aaf1b/src/voxcpm/modules/locdit/unified_cfm.py#L51-L136), and [AudioVAE decoder](https://github.com/OpenBMB/VoxCPM/blob/5e293dee2f4745d93ff6459886121467b41aaf1b/src/voxcpm/modules/audiovae/audio_vae_v2.py#L452-L474). No KamVox code or reverse engineering was used. The Hugging Face community request was not posted because creating one requires authentication.

## Historical Stage-1 report (superseded by current diagnostics)

## Historical Browser inference implementation attempt (superseded by current diagnostics)

**Updated**: 2026-11-22  
**Overall**: STAGE 1 BLOCKED AT LATENT TRAJECTORY (e). Stages (b)–(d) pass their teacher-forced checks. For seed 123, the 4-step trajectory passes, but the 6-step trajectory falls below the predeclared cosine threshold at step 5. Per the fail-fast gate, the remaining seeds, 10-step trajectory, full waveform stage (f), and Stages 2–4 were not run. The server/API was not restarted or touched.

| Stage/check | Result | Evidence |
|---|---|---|
| Numerical acceptance criteria | SET BEFORE RECHECK | Each tensor passes at cosine >=0.98 and finite values; cosine <0.99 is flagged WATCH. Relative L2, max/mean absolute error, and cosine are reported. This accommodates the known INT4-versus-BF16 precision difference without treating high error as exact parity; it does not waive non-finite or low-cosine outputs. |
| Approved public-source basis | READY TO DERIVE | The user approved using the pinned [OpenBMB VoxCPM2 source](https://github.com/OpenBMB/VoxCPM/blob/5e293dee2f4745d93ff6459886121467b41aaf1b/src/voxcpm/model/voxcpm2.py), the [Uddom README](https://huggingface.co/Uddom/voxcpm2-khmer-webgpu/blob/ad57fe543ef28fab07741c8603035e13ee68f770/README.md) and [manifest](https://huggingface.co/Uddom/voxcpm2-khmer-webgpu/blob/ad57fe543ef28fab07741c8603035e13ee68f770/manifest.json), plus public implementation references. The Uddom README has no complete generation section; its specification-doc paths remain absent. |
| ONNX Runtime | MEASURED | `onnxruntime 1.32.0` is installed in `backend\.venv`; available providers are `AzureExecutionProvider` and `CPUExecutionProvider` (no Python CUDA provider). |
| Pinned graph download and integrity | PASS | Downloaded the Uddom revision `ad57fe543ef28fab07741c8603035e13ee68f770` (manifest total 1,937,476,083 bytes). All 13 manifest-declared graph, external-data, and embedding files match their expected byte counts and SHA-256 hashes. |
| Graph load and manifest signatures | PASS | All six ONNX graphs load with `CPUExecutionProvider`. Actual input/output names, element types, ranks, and fixed dimensions match the pinned manifest for all six graphs, including the 28-layer base-LM and 8-layer residual-LM KV-cache interfaces. |
| Reference runner and step (a) | PASS | [The fail-fast runner](backend/scripts/validate_onnx_reference.py) used the first 35 Unicode codepoints of `backend/tests/fixtures/khmer_long_sample.txt` (95 tokens including `<|audio_start|>`). Against the pinned PyTorch embedding tensor, max absolute difference was `2.98e-8`, mean absolute difference `2.72e-12`, relative RMSE `5.84e-9`, cosine `0.99999994`; the ONNX FP16 table exactly equals the PyTorch BF16 rows after FP16 conversion. A separate lookup over the full 1,093-codepoint fixture (2,293 tokens) also exactly matched after FP16 conversion: max absolute difference `2.98e-8`, mean `2.15e-12`, cosine `0.99999803`. Source: [upstream text embedding in `_inference`](https://github.com/OpenBMB/VoxCPM/blob/5e293dee2f4745d93ff6459886121467b41aaf1b/src/voxcpm/model/voxcpm2.py#L1054-L1063) and [Uddom embedding manifest](https://huggingface.co/Uddom/voxcpm2-khmer-webgpu/blob/ad57fe543ef28fab07741c8603035e13ee68f770/manifest.json#L1734-L1740). |
| Reference runner and step (b) | PASS-WITH-QUANTIZATION-NOISE | Current memory-isolated rerun used the 35-codepoint text (95 tokens) and BF16 PyTorch on CUDA. Hidden: finite, cosine `0.99408329`, relative L2 `0.10890029`. All 56 K/V tensors finite; minimum cosine `0.99200827` (`present.27.value`), maximum relative L2 `0.12618779`. No tensor was in WATCH. An initial run with the new loader failed because meta initialization left non-persistent rotary buffers uninitialized; the buffers were re-created using the pinned upstream MiniCPM rotary implementation, and this corrected rerun matches the earlier base-LM parity result. The PyTorch reference inference itself took `1.196 s` (model load excluded); the ONNX CPU session-load-plus-prefill phase took `2.185 s` (not separately timed). Source: [upstream `_inference`](https://github.com/OpenBMB/VoxCPM/blob/5e293dee2f4745d93ff6459886121467b41aaf1b/src/voxcpm/model/voxcpm2.py#L1054-L1082), [MiniCPM forward/cache and RoPE](https://github.com/OpenBMB/VoxCPM/blob/5e293dee2f4745d93ff6459886121467b41aaf1b/src/voxcpm/modules/minicpm4/model.py#L355-L444), and [Uddom base-LM manifest](https://huggingface.co/Uddom/voxcpm2-khmer-webgpu/blob/ad57fe543ef28fab07741c8603035e13ee68f770/manifest.json#L77-L90). |
| Stage (b) memory | MEASURED | CUDA PyTorch reference: available RAM before load `9.05 GiB`, minimum during load `6.72 GiB`, after load `8.80 GiB`; peak process working set `3.60 GiB`. Peak CUDA allocated/reserved `3.06/3.24 GiB`; free CUDA after load `1.72 GiB`. ONNX CPU phase: available RAM `9.37 GiB` before, minimum `8.59 GiB`, peak process working set `1.38 GiB`. These are this run's measurements, not general minimum requirements. |
| Stage (c), residual-LM teacher forcing | PASS-WITH-QUANTIZATION-NOISE | The PyTorch inputs (base hidden, zero audio features/mask, seed 123 setup) were fed through the ONNX residual graph. `dit_hidden` cosine `0.99966574`, relative L2 `0.02635178`; stop-logit cosine `0.99999660`, relative L2 `0.00281189`; all 16 K/V outputs finite and >=0.98, no WATCH tensors. CUDA PyTorch residual reference: peak allocated/reserved `0.76/0.80 GiB`, minimum available RAM `8.84 GiB`; ONNX CPU process peak working set `0.85 GiB`. |
| Stage (d), one teacher-forced DiT step | PASS-WITH-QUANTIZATION-NOISE | Same PyTorch inputs, seed `123`: finite; cosine `0.99995685`, relative L2 `0.00934051`, max absolute error `0.03232479`. CUDA PyTorch peak allocated/reserved `0.48/0.50 GiB`, minimum available RAM `15.86 GiB`; ONNX CPU process peak working set `0.68 GiB`. |
| Stage (e), teacher-forced latent trajectory | FAIL — STOPPED AT FIRST MISMATCH | Euler time/delta now advances per step as in the pinned public sampler. Seed `123`, 4 steps: every step passed; final cosine `0.99421668`, relative L2 `0.10741054` (steps 3–4 are WATCH, still >=0.98). Seed `123`, 6 steps: steps 1–4 passed; step 5 cosine `0.94339651`, relative L2 `0.33649117`, so it failed the 0.98 threshold and the run stopped. No NaN/Inf was detected before failure. The other four seeds and 10-step runs were not run because the fail-fast gate stopped at this first mismatch. This is measured error accumulation in the quantized DiT trajectory, not final-waveform quality evidence. |
| Public fallback candidate | ASSESSED FROM MODEL CARD; NOT EXECUTED | The pinned [`soniqo/VoxCPM2-ONNX` revision `78f54fc97ac9401802e7f79027e81b33f711a7b0`](https://huggingface.co/soniqo/VoxCPM2-ONNX/tree/78f54fc97ac9401802e7f79027e81b33f711a7b0) declares Apache-2.0, voice cloning/reference-audio support, and INT8 graphs. Its README lists approximately `2.6 GB` text-prefill + `3.1 GB` token-step + `183 MB` audio encoder + `175 MB` decoder (about `6.1 GB` total). The token-step file alone exceeds the requested 3 GB VRAM budget by file size; actual VRAM use is NOT CONFIRMED. The model card reports prefill-hidden cosine `0.991–0.995` vs its FP32 export and identical ASR round-trip transcripts, but those are upstream-reported results, not measurements here. Khmer is not among the individually named languages; its multilingual tag does not establish Khmer quality. This candidate uses a different two-graph prefill/token-step split, with residual/CFM work combined into token-step, so the current Uddom stage (c)/(d)/(e) harness cannot be reused as-is. It was not downloaded or run: adapting a runner and collecting reference outputs is required, and current available RAM was `6.86 GiB` with no ONNX Runtime CUDA provider. |
| Uddom community request | NOT POSTED | The public discussion list was empty. “New discussion” redirects to Hugging Face login, so no issue/discussion was opened and no URL exists yet. |
| Stage (f), waveform / free-running test | NOT RUN — BLOCKED BY (e) | No PyTorch-versus-ONNX full-generation comparison was attempted after the latent trajectory failed. Duration, patch count, stop-token position, RMS, >4 kHz ratio, speech-only chars/s, and WAV files are NOT CONFIRMED. |
| Stages 2–4 | NOT RUN — BLOCKED BY (e) | No onnxruntime-web worker, browser graph-loading check, browser output comparison, interrupted-download resume test, offline-generation test, or browser/RTX 4050 memory and speed benchmark was run. |

**Source/licensing notes:** Uddom declares Apache-2.0 and its pinned [`LICENSE`](https://huggingface.co/Uddom/voxcpm2-khmer-webgpu/blob/ad57fe543ef28fab07741c8603035e13ee68f770/LICENSE) and [`NOTICE`](https://huggingface.co/Uddom/voxcpm2-khmer-webgpu/blob/ad57fe543ef28fab07741c8603035e13ee68f770/NOTICE) remain in the downloaded snapshot; the project's existing copies under `frontend/public/vendor/` are retained. The pinned OpenBMB source is [Apache-2.0](https://github.com/OpenBMB/VoxCPM/blob/5e293dee2f4745d93ff6459886121467b41aaf1b/LICENSE). The generation procedure is being derived only from the pinned public [`_generate`/`_inference` and prompt layout](https://github.com/OpenBMB/VoxCPM/blob/5e293dee2f4745d93ff6459886121467b41aaf1b/src/voxcpm/model/voxcpm2.py#L468-L684) and [`_inference` loop](https://github.com/OpenBMB/VoxCPM/blob/5e293dee2f4745d93ff6459886121467b41aaf1b/src/voxcpm/model/voxcpm2.py#L998-L1133), [seed handling](https://github.com/OpenBMB/VoxCPM/blob/5e293dee2f4745d93ff6459886121467b41aaf1b/src/voxcpm/model/utils.py#L27-L37), [Euler/CFG sampler](https://github.com/OpenBMB/VoxCPM/blob/5e293dee2f4745d93ff6459886121467b41aaf1b/src/voxcpm/modules/locdit/unified_cfm.py#L51-L131), [MiniCPM base LM and cache](https://github.com/OpenBMB/VoxCPM/blob/5e293dee2f4745d93ff6459886121467b41aaf1b/src/voxcpm/modules/minicpm4/model.py#L355-L444), [AudioVAE](https://github.com/OpenBMB/VoxCPM/blob/5e293dee2f4745d93ff6459886121467b41aaf1b/src/voxcpm/modules/audiovae/audio_vae_v2.py#L366-L500), and [scalar quantization](https://github.com/OpenBMB/VoxCPM/blob/5e293dee2f4745d93ff6459886121467b41aaf1b/src/voxcpm/modules/layers/scalar_quantization_layer.py#L5-L22). The Uddom README reports teacher-forced patch cosine similarity of at least 0.974, but that is not a published acceptance tolerance for base-LM hidden states or KV tensors. The Hugging Face discussion was not posted because creating one requires authentication.

## Historical Browser WebGPU engine assessment (status superseded above)

**Updated**: 2026-11-22
**Overall at the time**: NOT CLEARED FOR IMPLEMENTATION — this assessment predates the corrected seeded, patch-conditioned DiT diagnostics and the PyTorch/ONNX-VAE audio comparisons documented above. Full ONNX autoregressive generation and browser inference remain NOT CONFIRMED.

| Check | Result | Evidence |
|---|---|---|
| Public 4-bit candidate | MEASURED FROM PUBLIC MODEL CARD/MANIFEST | `Uddom/voxcpm2-khmer-webgpu` reports 1.937 GB, ONNX graphs for WebGPU, Apache-2.0, voice-cloning tags, and GPTQ 4-bit transformer weights. The public manifest lists a 1,937,476,083-byte total and SHA-256 hashes. |
| License | MEASURED | Apache-2.0 `LICENSE` and `NOTICE` are present in the public repository/model and copied to `frontend/public/vendor/`. |
| Khmer quality | NOT CONFIRMED | The model card states Khmer intelligibility was not formally evaluated with Khmer ASR and asks for listening verification. |
| Voice cloning | PARTIAL / NOT RUN | The model card is tagged for voice cloning, but the public reference-audio input layout is not available; exact Vanna compatibility is not confirmed. |
| RTX 4050 6 GB | NOT CONFIRMED | The model card reports real-time speed on one unspecified NVIDIA GPU. It does not confirm this RTX 4050 Laptop GPU, dedicated-memory usage, or OOM behavior. |
| Browser inference | NOT IMPLEMENTED | The app detects WebGPU and has model-download UI, but generation remains intentionally blocked because the teacher-forced 6-step latent trajectory fell below the acceptance threshold at step 5. No claim is made that browser inference works. |
| Offline PWA | IMPLEMENTED PARTIALLY | App shell and model attribution files are cached by a service worker. Cached ONNX model files are stored in Cache API after successful SHA-256 verification. Full offline generation is not verified. |
| Download resume | NOT IMPLEMENTED | Complete files remain cached and can be reused. Interrupted downloads do not yet resume from byte offsets. |
| Local history/WAV storage | NOT IMPLEMENTED | Existing server history and login behavior remain unchanged. |

## User pace calibration

**Updated**: 2026-11-22  
**Overall**: PARTIAL — the user's feedback brackets the preferred pace between 14 and 23 CPS. An 18 CPS target is set as the midpoint; the speech is not yet confirmed by listening.

| Check | Result | Evidence |
|---|---|---|
| Latest attached output | MEASURED | The new 676-character WAV matches generation `a6f1339a-f523-4828-9539-dbf6bec22a2f`: High/10, speed 1.0x, target 14 CPS. It is 49.117 s total / 47.75 s detected speech, about 14.16 speech-only CPS and 13.76 CPS including duration; one pause >=250 ms totals 0.44 s. This attachment therefore did not test the current 18 CPS target. |
| User-confirmed output | MEASURED + USER CONFIRMED | The latest 676-character WAV matches generation `50f632b0-9b35-48a2-8d1f-2cfe260c75c9`: Vanna, High/10, speed 1.0x, target 18 CPS. It is 38.202 s total / 37.20 s detected speech, about 18.17 speech-only CPS and 17.70 CPS including duration; one pause >=250 ms totals 0.30 s. The user reports the speaking speed is now correct. |
| Noise-reduction preview | CREATED, NOT LISTENING-VERIFIED | ZipEnhancer processed a copy of the confirmed WAV; the original is unchanged. Preview is 16 kHz / 38.202 s, RMS -32.69 dBFS, high-frequency energy ratio 1.05% (original 2.88%). RMS activity detection counts 33.64 s speech and 7 pauses >=250 ms totaling 2.90 s (original 37.20 s and one 0.30 s pause). This may indicate quiet speech was suppressed or the detector threshold was crossed; it does not prove intelligibility. The user must compare by ear; denoising is not enabled by default. |
| Latest denoised output | MEASURED; USER REPORTS HARD TO LISTEN TO | The 676-character WAV matches generation `50eafce8-3322-488c-b94d-1b3704c9d708` with `noise_reduction=true`, target18, High/10, and 1.0x speed. It is 38.202 s; activity detection finds 33.63 s speech and 7 pauses >=250 ms totaling 2.90 s. High-frequency energy fell from 2.88% to 1.00%, and spectral centroid from 2,844.6 Hz to 1,408.7 Hz, consistent with a muffled result. The user reports this output is difficult to listen to. Existing saved noise-reduction choices now reset to off once on page refresh; the option remains available manually. |
| WAV (6), denoiser off | MEASURED; USER REPORTS STILL MUFFLED | SHA-256 matched generation `c8343a45-fe70-4091-b433-65dee5e00813`: Vanna, CFG 2, 17 steps, custom pace target18, and `noise_reduction=false`. Duration is 37.663 s / 36.97 s detected speech, with no pauses >=250 ms. Its matching raw cache is 28.00 s / 27.90 s speech, so pace normalization stretched it at rate 0.7434 (about 1.345x duration). The final's high-frequency ratio/centroid are 3.63% / 2,660 Hz versus 3.63% / 2,639 Hz raw; these metrics show no broad high-frequency attenuation and cannot determine perceived clarity. |
| Same-pace WSOLA comparison | USER CONFIRMED CLEARER; IMPLEMENTED | Created a separate FFmpeg `atempo` preview from WAV (6)'s exact raw-cache audio, not from a new model run. Duration is 37.619 s, RMS is matched to WAV (6) at -22.62 dBFS, peak is -3.22 dBFS, and high-frequency ratio is 3.13%. The user confirmed this sounds clearer. Pace correction now uses FFmpeg `atempo`, RMS-matches with a 0.98 peak cap, and increments the processed-audio cache version to prevent stale output reuse. |
| WAV (9), residual muffling/noise | MEASURED; USER REPORTS SLIGHT RESIDUAL | SHA-256 matched generation `145bac09-f540-4fc2-8751-295a8167ad97`: Vanna, CFG 2, 10 steps, target18, `noise_reduction=false`. Duration is 38.165 s / 37.40 s detected speech, with no pauses >=250 ms. Its raw cache is 31.36 s / 30.80 s speech, so pace correction is about 0.822x. High-frequency ratio/centroid are 2.52% / 2,754 Hz final versus 2.89% / 2,832 Hz raw; this is a modest change, not an objective clarity score. |
| Gentle cleanup preview for WAV (9) | USER CONFIRMED BETTER; IMPLEMENTED AS OPTIONAL | FFmpeg `afftdn` at 3 dB noise reduction was applied as a separate copy and RMS-matched to the original. Duration and RMS remain 38.165 s and -20.72 dBFS; peak is -0.18 dBFS. The high-frequency ratio changes only from 2.516% to 2.501%, and centroid from 2,754 Hz to 2,696 Hz. The user confirmed the preview is better; the app now uses the same gentle filter behind its off-by-default option. |
| Noise-reduction app option | IMPLEMENTED | The saved, off-by-default checkbox now runs FFmpeg `afftdn` at 3 dB noise reduction after pace/global-speed processing, RMS-matches with a 0.98 peak cap, and stores the choice in generation history. This replaces the stronger ZipEnhancer experiment that made a prior sample muffled. The processed-audio cache version was incremented to 6. |
| Fast output | MEASURED | Prior High/10 WAV for the same text has 25.71 s of speech, about 23.03 CPS; user reports it is too fast. |
| Pace candidate | MEASURED SIMULATION | Applying pitch-preserving stretch to the fast WAV: target16 → 36.76 s / 16.10 CPS; target18 → 32.82 s / 18.04 CPS; target20 → 29.61 s / 19.99 CPS. Natural now defaults to 18 CPS, with Relaxed16 and Brisk20. |
| Preference migration | IMPLEMENTED, NOT E2E VERIFIED | Legacy saved 14 CPS values migrate once to Natural/18 even if the old profile field says Custom. Other newer custom targets are preserved. The browser test covers this migration but was not run because it proceeds to costly model generation. |
| Runtime | PASS | Frontend lint/build and browser-test syntax checks passed. The API was stopped during the latest check, so it was restarted; `/health`, OpenAPI, and TTS status returned HTTP 200, with VoxCPM2 loaded on CPU. User/admin frontends returned HTTP 200. |
| Listening confirmation | USER CONFIRMED | The user confirms that the latest 18 CPS sample has the right speed. This confirms pace for this sample, not every text or voice. |
| ZipEnhancer experiment | HISTORICAL; REMOVED FROM APP | The initial preview used the direct ModelScope pipeline because the upstream wrapper's optional Torchaudio loudness normalization could not load TorchCodec DLLs on this Windows environment. The later app ZipEnhancer path was exercised, but the user found its output difficult to listen to; the app now uses the user-approved gentle FFmpeg filter instead. |
| Regression checks | PASS WITH LIMITATIONS | 26 focused backend tests passed; frontend lint/build and Python syntax checks passed. OpenAPI exposes `noise_reduction` with default false, the PostgreSQL column migration is applied, and API health/status returned HTTP 200. Browser E2E was not rerun; its test now covers the unchecked default and preference toggle. |

## Earlier speech pace correction at 14 CPS (superseded)

**Updated**: 2026-11-22  
**Overall**: PARTIAL — the newest High/10 WAV was measured and the pace correction was tested directly on it. The corrected file has not yet been generated in the app or checked by listening.

| Check | Result | Evidence |
|---|---|---|
| Newest output | MEASURED | The attached WAV matches its saved generation: 592 characters, High/10 steps, Vanna, 1.0x speed, Custom pace target 14 CPS. It is 25.76 s with 25.71 s detected speech and no pauses >=250 ms, about 23.0 speech-only CPS. |
| Cause | MEASURED | The previous lower stretch bound skipped the required 0.608x pace correction, leaving the full-speed High/10 audio unchanged. |
| Corrected pace simulation | MEASURED | Applying pitch-preserving stretch at 0.608x to the supplied WAV (no model call) produced 42.37 s total, 42.02 s detected speech, and 14.09 speech-only CPS. |
| Changes | PASS | Pace normalization now supports 0.50x–1.15x; the full processed-audio cache version was bumped so stale fast output is not reused. Generation remains single-pass. The matching raw audio is cached, so repeating the same request can reuse model output and only redo pace processing. |
| Validation | PASS WITH LIMITATIONS | Focused backend suite: 23 passed; Python syntax checks passed. API `/health` returned 200, OpenAPI reports a 10-step default, and the model is loaded on CPU. Both frontends returned 200. No new model inference or human listening comparison was run. |

## Earlier user audio clarity follow-up (superseded)

**Updated**: 2026-11-22  
**Overall**: PARTIAL — the supplied WAV was matched byte-for-byte to its saved generation and its settings and processing were traced. The preset default and excessive pace-correction behavior were adjusted; no new High-quality WAV has been generated, and perceptual improvement is not confirmed.

| Check | Result | Evidence |
|---|---|---|
| Supplied audio identity | MEASURED | SHA-256 matched the saved generation WAV. Its database record has 592 text characters, Vanna voice, CFG 2.0, Fast/4 steps, 1.0x speed, Custom pace at 14 CPS, and the pronunciation dictionary enabled. |
| Output signal | MEASURED | Mono 48 kHz, 33.80 s; peak 0.609 (no clipping), RMS -26.00 dBFS, 33.65 s detected speech, and no pauses >=250 ms. These measurements do not prove or disprove perceived noise/clarity. |
| Pace processing | MEASURED | The matching raw-cache key was verified from the generation settings. Raw speech was 28.60 s (about 20.7 CPS); the output was lengthened to 33.80 s by the 0.85x minimum stretch, still about 17.6 CPS. Reaching 14 CPS would require a stretch outside the supported range. |
| Spectral comparison | MEASURED, NOT A CLARITY SCORE | Raw vs final >4 kHz energy ratio was 3.607% vs 3.602%; centroid 3,469.8 vs 3,455.7 Hz; flatness 0.00386 vs 0.00214. These features cannot establish whether a person would hear hiss, distortion, or reduced clarity. |
| Changes | HISTORICAL | High quality/10 became the default. At that stage, pace stretching was skipped below 0.85x; the speech pace follow-up above now allows 0.50x–1.15x so the new sample can reach its target. |
| Validation | HISTORICAL PASS | Focused backend suite: 22 passed. Frontend lint/build and Python syntax checks passed. API `/health` and both frontend pages returned 200; model loaded on CPU in 13.27 s. No real-model regeneration or listening comparison was run. |

## Single-pass TTS switch

**Updated**: 2026-11-22  
**Overall**: PASS WITH LIMITATIONS — production synthesis now sends the complete input through one VoxCPM2 inference call. Local services, focused regressions, frontend lint, and production build passed. No new audio was generated, so current single-pass speed and audio quality are not measured.

| Check | Result | Evidence |
|---|---|---|
| Synthesis path | PASS | `_synthesize_request` now synthesizes full text once, performs pace processing on that result, and emits one complete WAV event. It does not retry inference or join chunks. |
| Regression tests | PASS | Focused backend suite: 20 passed, 0 failed. Includes long-text one-call behavior, one streamed output, voice/seed/settings preservation, and no retry inference. Legacy chunk-splitter helper tests remain but are not used by production synthesis. |
| Frontend | PASS | `npm run lint` and `npm run build` both passed after the single-pass UI update. |
| Diagnostics utility | PASS | Python syntax check and `diagnose_tts_chunks.py --help` passed; the utility now accepts full-text settings and saves raw/processed/final WAVs rather than per-chunk output. |
| Local services | PASS | API `/health` returned HTTP 200; user and admin frontends returned HTTP 200. API status reported the model loaded on CPU in 13.69 s, with the full 4,957,031,714-byte cache and 0 VRAM. |
| Time-to-first-audio, total time, RTF, quality | NOT MEASURED | No new real-model synthesis or browser E2E was run. Since streaming now emits audio after full-text inference, audio does not begin early. Historical chunked-mode timings below are not a single-pass baseline. |

## Earlier local CPU generation speed adjustment (superseded)

**Updated**: 2026-11-22  
**Overall**: HISTORICAL — Fast/4 was selected as the default at this stage to reduce CPU inference time. The current default is High quality/10; see the user audio clarity follow-up above.

| Check | Result | Evidence |
|---|---|---|
| Before-change latency | MEASURED | During the reported 592-character user request, chunk 1 (76 characters) took 85.09 s of model inference; first audio arrived after 91.17 s. The model was running on CPU. The request's selected inference-step count was not logged, so this is not a controlled 10-vs-4 comparison. |
| Default preset at that time | HISTORICAL | Frontend and API defaults were Fast/4. The later clarity follow-up supersedes this setting with High quality/10. |
| Backend regression tests | PASS | `tests.test_generation_preferences` and `tests.test_tts_chunking`: 18 passed, 0 failed. |
| Frontend | PASS | `npm run lint` and `npm run build` completed successfully. |
| Speed improvement amount | NOT MEASURED | No same-text CPU comparison at 10 and 4 steps was run; reduced steps are expected to trade some quality for shorter inference but exact latency improvement is unconfirmed. |
| Queue diagnosis and restart | PASS | The browser showed `Queued · position 1 · 4 chunks`; the in-process queue allows one active generation. An earlier CPU generation was still occupying the worker. Per user confirmation, the API was restarted, which cancelled that active request and the queued request; neither produced a new history record. The API/model then restarted and passed health checks. |

## Local-only migration and runtime verification

**Updated**: 2026-11-20T19:40:03+07:00  
**Overall**: PASS WITH LIMITATIONS — Docker data was merged into the existing host database, project-specific Docker resources were removed, and the API plus both frontends run locally. No new TTS generation was run during this transition.

| Check | Result | Evidence |
|---|---|---|
| Database merge | PASS | Local database now has 17 users, 6 voices, and 18 generations. Eight Docker users were added; one same-email account was mapped to its existing local user, preserving the local account credentials. All five Docker voices matched existing local voices by name. |
| Audio migration | PASS | 120 Docker audio-volume files (183,063,816 bytes) were copied under `backend/audio_files/docker-imported`. All 14 imported generation paths and all 4 existing local generation paths resolve to files. The Vanna reference WAV and `VoiceClone/Vanna.m4a` remain present. |
| Database backup | PASS | Custom-format backups of both databases were saved outside the project before any merge writes. |
| Local PostgreSQL schema | PASS | Startup added the missing speed/style/pace columns to the older host schema. The startup migration now adds columns and adjusts `target_cps` nullability in separate statements; API startup against the migrated database succeeded. |
| API and model | PASS | `http://localhost:8000/health` returned 200. `/api/tts/status` reported the model loaded, 100% cached (4,957,031,714 bytes), on CPU in 17.05 s. Only 4.96 GiB of the 6 GiB GPU was free, below the app's 8 GiB CUDA threshold. |
| Frontends | PASS | User app `http://localhost:3000` and admin app `http://localhost:3001` each returned HTTP 200. This was an HTTP smoke check, not a browser E2E. |
| Backend regression tests | PASS | `test_generation_preferences.py` and `test_tts_chunking.py`: 18 passed, 0 failed. `app/main.py` also passed `py_compile`. |
| Docker cleanup | PASS | KalapakTTS containers, Compose network, named volumes, and locally built images are absent. Other Docker projects and shared upstream images were left untouched; host PostgreSQL and PC-installed pgAdmin4 were not removed. |
| New TTS generation | NOT RUN | CPU generation can take many minutes. Existing history files were verified, but a fresh synthesis was not part of this database/runtime transition. |

## Latest absolute-pace and natural-flow follow-up

**Updated**: 2026-11-20T18:53:51+07:00  
**Overall**: PASS WITH LIMITATIONS — the full 1,093-code-point fixture generated a valid WAV with stable estimated speech pace close to the 14 CPS target and a visible per-generation pace metric. RMS VAD found 8 pauses >=250 ms in this output; the user's separate audio is unavailable for a direct comparison. CPU inference remains very slow, and the estimated pace is not word-aligned or perceptually verified.

### Changes

- Natural is now the default absolute target (14 CPS); Relaxed/Natural/Brisk are 12/14/16 CPS. Global speed remains separate and is applied after chunk joining.
- Synchronous, job, and streaming routes now share 250–450-character chunking with a first-chunk cap of 90 characters. Sentence/paragraph boundaries can create smaller chunks.
- Chunks use 40 ms edge padding, 200 ms sentence pauses, 400 ms paragraph pauses, 40 ms equal-power continuation crossfades, and bounded pitch-preserving correction. A chunk outside the ±12% target tolerance is corrected only within 0.85x–1.15x; otherwise up to two alternate deterministic seeds are tried.
- Retry candidates are now named in SSE progress so long CPU inference does not appear frozen.
- The displayed speech CPS comes from RMS-based speech-activity detection. Per-third CPS is estimated by distributing chunk characters across each chunk's active duration; neither method aligns words or syllables. Vanna has no verified transcript, so reference CPS and syllables/s are unconfirmed.

### Full-fixture pace comparison

The saved fixture contains 1,093 code points including newlines (1,088 chunk text characters). Before values below are from the prior 5-step/eight-chunk run in this report; after values are from a 10-step/six-chunk run at 1.0x. Both third-by-third series are chunk-timeline estimates, not forced alignment. Because steps and chunk boundaries changed, this is indicative rather than a controlled one-variable comparison.

| Audio third | Before CPS (ESTIMATED) | After CPS (ESTIMATED) |
|---|---:|---:|
| First | 17.50 | 14.17 |
| Middle | 16.72 | 14.66 |
| Last | 16.15 | 14.54 |

| Metric | Before | After |
|---|---:|---:|
| Total duration | 64.685 s | 79.652 s |
| Total input code points / duration | 16.89/s | 13.72/s |
| Speech-only CPS | NOT CONFIRMED from the retained before waveform | 14.51/s (VAD estimate; 1,088 spoken code points / 75.00 active seconds) |
| Pauses >=250 ms | NOT CONFIRMED | 8 pauses, 3.38 s total (VAD estimate) |
| Third-to-third spread | 1.35 CPS | 0.50 CPS |

After-generation third estimates are within 4.8% of the 14 CPS target, and their range is about 3.4% of their mean. The final mono 48 kHz PCM_16 WAV is 79.652 s, 7,646,674 bytes, and 3,823,315 frames. Its six chunk rates were 15.15, 14.05, 14.10, 15.01, 14.83, and 14.22 CPS; no alternate-seed retry was needed for this 10-step run. RMS VAD detected 75.00 s of speech and 8 pauses of at least 250 ms totaling 3.38 s. The audio is available at [final.wav](./backend/audio_files/diagnostics-natural-pacing/final.wav); its numeric summary is [summary.json](./backend/audio_files/diagnostics-natural-pacing/summary.json).

Summed model inference for the six chunks was 772.37 s (12 min 52 s) on CPU. The model was loaded once and kept warm; after the final API rebuild, startup load was 14.41 s, device was CPU (bfloat16), peak process RAM was 11.96 GB (11.14 GiB), and VRAM use was 0. The complete model cache remained at 4,957,031,714 / 4,957,031,714 bytes. The local RTX 4050 was not used.

### Vanna reference and user-provided comparison

The reference WAV (`backend/voice_references/vanna.wav`) is 19.968 s at 24 kHz. RMS VAD found 18.928 s speech and one pause >=250 ms lasting 0.33 s. There is no verified transcript, so its speech-only CPS and syllables/s cannot be calculated. The separate 83-second user WAV is not present in the workspace; its reported 11 pauses and 13–14 speech-only CPS remain user-reported, not independently remeasured.

### Browser, test, and deployment evidence

| Check | Result | Evidence |
|---|---|---|
| Focused backend tests | PASS, exit 0 | `python -m unittest tests.test_audio_diagnostics tests.test_tts_chunking tests.test_tts_cache tests.test_tts_jobs -v`: 19 passed, 0 failed. Covers default 14 CPS, absolute normalization, retry cap/seeds/progress, chunk boundaries, joins, queue, and cache. |
| Frontend lint/build | PASS, exit 0 | `npm run lint` and `npm run build`; TypeScript and production build completed. |
| Compose config and rebuild | PASS, exit 0 | `docker compose --env-file .env.docker config -q`; API and user frontend rebuilt. All five Compose services report healthy after the final API restart. |
| Startup/model | PASS | API `/api/tts/status`: model loaded, CPU, 14.41 s load, model download 100%. |
| Browser E2E | PASS, exit 0 | `test_tts_browser.py`: 1 passed in 256.560 s. Signup/login, Khmer character counter, enabled Generate, saved preferences, Vanna selection, streamed WAV, browser playback/history, and estimated pace metric. |
| Browser first-audio latency | MEASURED | Fast/4, News style, Vanna, 0.8x, first 35-character chunk: 142.18 s. Three seed candidates took 44.73, 46.71, and 47.22 s; the selected chunk still needed a clamped 1.15x stretch and produced 6.90 CPS after the requested 0.8x global speed. The two retries did not reach the 14 CPS target. |
| Browser request total / RTF | MEASURED | The two-chunk, 241-character browser sample completed in 249.10 s and returned 23.14 s of audio (RTF 10.77). This demonstrates audio playback, but not a fast start or real-time CPU synthesis. |
| Perceptual joins/style | NOT CONFIRMED | No human listening comparison was performed. Spectral/audio metrics do not prove clarity, naturalness, or absence of clicks. |
| Reference CPS and syllables/s | NOT CONFIRMED | No verified Vanna transcript or Khmer syllabifier is available. |

The first browser attempt stopped at a stale test assertion that expected a null target after the UI began defaulting to 14 CPS. The assertion was updated, and the rerun passed. The full-fixture API generation itself returned a valid WAV and wrote its summary, but the one-off host harness exited 1 when it printed the Khmer voice name through Windows CP1252 (`UnicodeEncodeError: 'charmap' codec can't encode characters in position 676-680`). The saved WAV and JSON were then read successfully with UTF-8 output; the temporary account cleanup check returned zero remaining `pace-verify` accounts.

```text
environment:
  docker: AVAILABLE — Docker Engine 29.4.3; daemon responsive
  node: AVAILABLE — v20.19.5
  playwright: AVAILABLE — Python Playwright with headless Chromium; browser E2E passed
  infra-tier: PRIMARY(Docker PostgreSQL) — Compose database and services healthy
  browser-tier: PRIMARY(Playwright) — streamed-generation browser E2E passed
startup: PASS — API healthy; warmed VoxCPM2 loaded on CPU in 14.41 s
integration: PASS — exit_code: 0, passed: 19, failed: 0, skipped: 0; focused backend unit/cache/job suite
e2e: PASS — exit_code: 0, passed: 1, failed: 0; signup through streamed WAV playback/history and pace metric
overall: PASS WITH LIMITATIONS — target pace is close on the full fixture, but CPU TTFA/RTF are poor; thirds are estimates, and perceptual quality/reference rate are unconfirmed
```

The browser E2E above ran before the final retry-progress message was added; the final API image was rebuilt and its retry callback is covered by the focused unit test, but the browser suite was not repeated for that copy-only update.

## Latest streaming and performance follow-up

**Updated**: 2026-11-19T19:19:37+07:00  
**Latest overall**: NEEDS SIGN-OFF — the CPU streaming path and final WAV passed a browser E2E on a 511-character first-paragraph fixture sample, but first audio still took 52.49 seconds. The target of a few seconds is not met. A full-fixture streaming E2E, exact RTF, GPU inference, and perceptual quality comparisons remain unconfirmed.

| Check | Result | Evidence |
|---|---|---|
| Docker services and model warm-up | PASS | Compose services were healthy. API status reported VoxCPM2 loaded once on CPU; model cache 4,957,031,714 / 4,957,031,714 bytes. Current API load after restart was 22.75 s; peak RSS metric 11,989,811,200 bytes; VRAM 0. |
| Backend TTS suite | PASS, exit 0 | `python -m unittest discover -s tests -p 'test_tts_*.py' -v`: 17 passed, 0 failed, 0 skipped in 189.187 s. Includes real-model speed/style, Vanna, WAV/history, cache, queue, and chunking flows. |
| Frontend | PASS, exit 0 | `npm run lint` and `npm run build`. |
| Browser E2E | PASS, exit 0 | `python ..\frontend\tests\test_tts_browser.py`: 1 passed in 223.907 s. The input was the first 511-character paragraph from the saved fixture, Vanna, News, Fast/4 steps, 0.8x. It verified signup/login, character counter and enabled Generate, restored speed/style, four streamed chunks, saved WAV/history, WAV parsing, and playback. |
| Time to first audio | MEASURED | 52.49 s for the first 35-character chunk. Per-chunk inference for the four-chunk E2E was 52.43, 53.22, 47.13, and 64.26 s (217.04 s summed inference time). The full browser test time includes signup and page setup, so it is not reported as request generation time. |
| Model/resource profile | MEASURED | CPU only; 22.75 s startup load; peak RSS metric about 12.0 GB; Docker memory sample about 7.06 GiB / 15.43 GiB while generating; VRAM 0. |
| GPU Compose profile | PARTIAL | `docker compose --profile gpu config -q` passed. The CUDA image was not built or run. The RTX 4050 has 6 GiB and the app requires at least 8 GiB free; no GPU speed, precision, or offload claim is made. |
| Full-sample timing / RTF | NOT CONFIRMED | No complete, clean final browser run covered all 1,093 fixture characters at the final settings, and the final run did not retain an exact output-duration measurement. Do not compare this TTFA against the uninstrumented pre-streaming baseline or claim an RTF. |
| Quality presets | PARTIAL | Fast/4 was used in the passing browser test. The 4/6/10 same-sentence timing and audible-quality comparison was not completed. |
| Style and joins | PARTIAL | Bilingual prompts and style metadata were covered by the backend integration test. Durations do not establish audible style differences; no listening/joins review was performed. |

During an earlier two-paragraph browser run, the app generated and loaded audio, but the test exited nonzero at its metadata assertion: the quality preset had reset to High/10 after reload while the test expected Fast/4. The test now explicitly reselects Fast after reload and the first-paragraph browser E2E passes. This run took 611.155 s and is not counted as a passing test.

Streaming makes a chunk available before the final WAV is ready, but 52.49 s to first audio is still too slow for the requested few-second start. Four-chunk inference took 217.04 s in total, while exact output duration was not captured; therefore RTF and gapless playback remain unconfirmed, and playback buffering gaps are possible. The first chunk was 35 characters because the splitter kept a sentence boundary; the 60–90 target is approximate, not a strict minimum. No CPU thread-count, `torch.compile`, quantization, ONNX, or OpenVINO benchmark was run. Existing Split/Pause controls remain; per-paragraph voice selection and parallel inference batching were not implemented. Cache behavior and queue-position behavior have unit coverage, but cache-hit latency and multi-user throughput were not benchmarked.

```text
environment:
  docker: AVAILABLE — Docker Engine 29.4.3; daemon responsive
  node: AVAILABLE — v20.19.5
  playwright: AVAILABLE — Python Playwright 1.61.0 and headless Chromium
  infra-tier: PRIMARY(Docker PostgreSQL) — Compose database and services healthy
  browser-tier: PRIMARY(Playwright) — headless browser E2E passed
startup: PASS — Compose API healthy; warmed model loaded on CPU in 22.75 s
integration: PASS — exit_code: 0, passed: 17, failed: 0, skipped: 0; backend TTS suite
e2e: PASS — exit_code: 0, passed: 1, failed: 0; signup through streamed WAV playback/history, first fixture paragraph
overall: NEEDS_SIGNOFF — CPU TTFA was 52.49 s; full-fixture RTF, GPU, quality comparison, and gapless playback are unconfirmed
```

## Latest long-text chunking and pacing follow-up

**Updated**: 2026-11-19T18:02:59+07:00  
**Latest overall**: PASS WITH LIMITATIONS — chunked background synthesis, consistent speed scaling, progress/history, valid WAV output, and the prior browser flow passed. The exact five-paragraph sample is preserved at `backend/tests/fixtures/khmer_long_sample.txt` (1,093 Unicode code points and 1,003 non-whitespace characters). Post-fix 0.8x/1.0x/1.2x durations scale inversely. The pre-change one-pass total is recorded below, but its waveform was not retained, so comparable before-change thirds cannot be reconstructed.

| Check | Result | Evidence |
|---|---|---|
| Docker services | PASS | API, user app, admin app, PostgreSQL, and pgAdmin report healthy. The model remains loaded on CPU; its 4,957,031,714-byte cache is 100% complete. |
| Backend suite | PASS, exit 0 | `python -m unittest discover -s tests -p 'test_*.py' -v`: 19 passed, 0 failed, 0 skipped in 231.737 s against isolated Compose PostgreSQL on port 5433. |
| Post-fix chunking/speed unit tests | PASS, exit 0 | `tests.test_tts_chunking`: 10 passed, including the saved-sample boundaries and the regression that applies speed after joining. |
| Exact-sample API generation | PASS, exit 0 | Authenticated signup/login, three 8-chunk jobs at 0.8x/1.0x/1.2x, exact history-text checks, valid WAV parsing, and test-account/audio cleanup. |
| Frontend | PASS, exit 0 | `npm run lint` and `npm run build` both passed after the final UI changes. |
| Browser E2E | PASS, exit 0 | `test_tts_browser.py`: 1 passed in 145.602 s; signup/login, preferences after reload, Vanna selection, job progress, WAV validation/playback, and history metadata. |
| Seeded speed comparison | PASS | Same short Khmer input, same seed/voice/settings: 0.8x = 3.00 s, 1.0x = 2.40 s, 1.2x = 2.00 s. Duration scaled inversely as expected. |
| Style prompts | PARTIAL | News anchor, Vlogger, and Storyteller prompts were accepted and stored. Their output durations were 6.56 s, 6.88 s, and 6.88 s respectively; durations alone do not prove an audible style change, and no perceptual review was performed. |
| Multi-chunk WAV | PASS, representative input only | A repeated mixed Khmer/English 1,188-code-point input completed as 6 chunks; progress reached 6/6. The Vanna WAV was valid, 48 kHz, 1,696,800 frames, 3,393,644 bytes, 35.35 s. This is a structural test, not the exact user text or a natural-pace assessment. |

### Exact five-paragraph speed run before the follow-up fix

The preserved fixture is 1,093 Unicode code points (1,003 non-whitespace), five paragraphs, split into eight chunks. Using Vanna, seed `20261119`, five inference steps, and the same Custom request settings, all three background jobs completed 8/8, persisted the exact submitted text, and returned valid mono 48 kHz WAVs.

| Requested speed | Duration | WAV bytes | Result |
|---:|---:|---:|---|
| 0.8x | 77.642 s | 7,453,704 | Valid WAV; 8/8 chunks |
| 1.0x | 64.685 s | 6,209,792 | Valid WAV; 8/8 chunks |
| 1.2x | 62.373 s | 5,987,820 | Valid WAV; 8/8 chunks |

These did not scale inversely as requested. API logs show the cause: per-chunk time-stretching reached the 1.25x clamp on four chunks at 1.0x and seven chunks at 1.2x. The speed target was therefore unreachable for much of the sample, and the fixed pauses were not scaled. The implementation now normalizes chunks at their baseline pace, joins them, then applies the clamped speed once to the complete waveform.

### Exact five-paragraph run after the global speed fix

The same Vanna voice, text, fixed seed (`20261119`), five inference steps, and 8-chunk job flow were used. All jobs completed at 8/8, stored the exact submitted text in history, and returned valid mono 48 kHz WAVs. The model was loaded on CPU from the complete cache.

| Speed | Measured duration | WAV bytes | Result |
|---:|---:|---:|---|
| 0.8x | 80.856 s | 7,762,228 | Valid WAV; 8/8 chunks |
| 1.0x | 64.685 s | 6,209,792 | Valid WAV; 8/8 chunks |
| 1.2x | 53.904 s | 5,174,834 | Valid WAV; 8/8 chunks |

All runs used the same text, Vanna preset, seed, and inference settings, with pitch-preserving speed applied to the joined waveform. Durations scale inversely as expected (0.8x is 1.25× the 1.0x duration; 1.2x is 1/1.2×).

For the 1.0x post-fix audio, equal-duration thirds were estimated from logged chunk character counts/CPS and the configured sentence/paragraph pauses:

| Audio third | Estimated characters/s |
|---|---:|
| First | 17.50 |
| Middle | 16.72 |
| Last | 16.15 |

The first-to-last difference is about 7.7%, within the requested 15% band. This is a chunk-timeline estimate (piecewise-constant pace within each chunk), not forced alignment. The earlier one-pass WAV was not retained, so equivalent before-change thirds cannot be measured. Its recorded overall average was 1,093 / 49.12 = 22.25 code points/s. The separately generated baseline paragraphs (16.37, 15.08, and 14.10 code points/s) are not thirds of that waveform and are not presented as such.

The user's request to shorten the verification arrived after the full 1.2x job had already been launched; that run was allowed to finish. No optional Latin-text comparison or additional full-sample runs were started.

The final backend suite emitted a non-fatal Numba `RuntimeWarning: invalid value encountered in cast` during the synthetic pitch-preservation test. The test completed successfully and its finite-output, duration, and pitch assertions passed.

Custom mode defaults to relative pace: without an explicit target CPS, the first chunk's measured pace is the baseline and following chunks are normalized to it at 1.0x. Auto mode uses 16 CPS by default. The requested speed is applied once to the joined waveform, avoiding inconsistent per-chunk speed clamps; effective global stretching is clamped to 0.75x–1.25x, so out-of-range API speeds are approximate. The API also supports an optional fixed seed; one seed is reused across every chunk in a request.

Crossfades, silence insertion, and pitch-preserving scaling were verified with synthetic audio tests. Audio joins were not listened to or perceptually evaluated, so clicks/volume continuity are not claimed as confirmed. Style changes remain prompt guidance, not guaranteed voice transformations.

An initial full-suite attempt failed two persistence assertions because the test process queried the host PostgreSQL on port 5432 while the API used the isolated Compose database on port 5433. Setting the test process `DATABASE_URL` to that same Compose database resolved the mismatch; the final full suite passed. Eight orphaned test-only accounts from the earlier attempt were removed, and the cleanup check found zero test-only accounts remaining.

## Long-text baseline diagnosis (before changes)

The user-supplied five-paragraph sample was captured before changing generation code. It contained 1,093 Python Unicode code points (1,003 non-whitespace characters); the user's separate 1,084-character count may use a different counting convention or omit a few characters.

| Baseline run | Text | Voice | Steps | Speed | Duration | Code points/sec |
|---|---:|---|---:|---:|---:|---:|
| Full one-pass | 1,093 chars | វណ្ណា (reference) | 5 | 1.0x | 49.12 s | 22.25 |
| Full one-pass | 1,093 chars | មាលី (no reference) | 5 | 1.0x | 47.36 s | 23.08 |
| First paragraph, separate request | 241 chars | វណ្ណា | 5 | 1.0x | 14.72 s | 16.37 |
| Middle paragraph, separate request | 251 chars | វណ្ណា | 5 | 1.0x | 16.64 s | 15.08 |
| Last paragraph, separate request | 176 chars | វណ្ណា | 5 | 1.0x | 12.48 s | 14.10 |

These are independent model requests, not transcript-aligned thirds of the same full-length audio. They cannot establish whether the audio itself accelerates toward the end. They do show that the full Vanna pass averages more code points/second than the separately generated paragraph samples. The Vanna and Mali full-pass durations differ by about 3.7%, so the reference voice is not supported as the main cause by this test, although inference is stochastic and no fixed seed was available through the API at baseline.

Source inspection found that the application did not pass VoxCPM2's supported `seed` argument, so each call chose a random seed. The VoxCPM2 wrapper defaults `normalize=False`; the app does not enable its text normalizer, so that normalizer is not the cause. The generation path applies a maximum audio-token length of at most 4,096; no evidence showed that this cap was reached in the 49-second sample, so a cap-induced compression is not confirmed. The strongest available evidence points to long single-pass generation/context as a likely contributor, but does not prove a within-audio acceleration pattern. Mixed-script tokenizer pronunciation remains uncertain and motivates the requested dictionary.

## Previous speed/style validation summary (historical)

| Step | Status | Exit Code | Details |
|---|---|---:|---|
| Docker startup | PASS | 0 | Compose started PostgreSQL, API, user app, and admin app; all four services report healthy. Model-download job exited successfully. |
| pgAdmin dashboard | PASS | 0 | pgAdmin 4.9.11 is healthy at `http://localhost:5050`; browser login succeeded, the KalapakTTS PostgreSQL server is registered, and the pgAdmin container can reach PostgreSQL. |
| Model download and integrity | PASS | 0 | Docker downloaded the 4.62 GiB VoxCPM2 snapshot; all 9 files passed the downloader's size and SHA-256 checks. |
| Frontend lint/build | PASS | 0 | `npm run lint` and `npm run build` both passed after the speed/style UI changes. |
| Backend integration | PASS | 0 | `python -m unittest discover -s tests -p 'test_*.py' -v`: 10 passed, 0 failed, 0 skipped against the isolated Docker PostgreSQL on port 5433. |
| Final API regression | PASS | 0 | After rebuilding the final backend, `tests.test_tts_controls_e2e`: 1 passed; six additional speed/style generations returned valid WAVs and persisted the requested metadata. |
| Browser E2E | PASS | 0 | `python -m unittest discover -s tests -p 'test_tts_browser.py' -v`: 1 passed; sign-in, speed/style persistence, style preview, Khmer generation, WAV validation, playback, and history. |
| Admin authentication | PASS | 0 | Docker-seeded administrator login succeeded and returned the admin role. |

**Overall**: PASS WITH LIMITATIONS — the full Docker stack is healthy; speed processing, prompt/style plumbing, persisted history, and browser playback work. Separate model generations produced variable base durations, so their duration measurements were not monotonic with speed. Distinct acting-style effects were not objectively confirmed.

## Model and hardware evidence

- Downloaded VoxCPM2 revision `7936080b41163874f38610f46f438e2c0757f9fd`.
- `model.safetensors`: 4,580,080,592 bytes; SHA-256 `f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d`.
- `audiovae.pth`: 376,951,122 bytes; SHA-256 `94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1`.
- The cache was redirected to E:. Free space was 261.7 GiB at download start and 244.8 GiB after validation; the downloader requires 10 GiB plus the model files.
- The engine loaded successfully on **CPU**. The RTX 4050 has 6,141 MiB VRAM, below the backend's 8 GiB GPU threshold. During loaded-model API/browser tests, combined API process working set peaked at 5.23 GiB, system free RAM bottomed at 6.44 GiB, and GPU memory use remained 73 MiB (5,848 MiB free).
- CPU inference completed successfully. CPU performance can be slower than GPU inference.
- In Docker, `/api/tts/status` reported `model_loaded: true`, `model_device: cpu`, and 100% downloaded. The post-generation API container sample was 7.37 GiB RAM out of the Docker VM's 15.43 GiB; this CPU-only image did not use VRAM. After setup and tests, C: had 26.15 GiB free and E: had 224.85 GiB free.
- After this rebuild and test run, the API still reported `model_loaded: true`, `model_device: cpu`, and 4,957,031,714 / 4,957,031,714 bytes downloaded. A post-browser `docker stats` sample was 7.18 GiB / 15.43 GiB. Host free space was 8.17 GiB on C: and 244.03 GiB on E:; the model cache is already complete and remains in its persistent volume.
- Docker uses a separate PostgreSQL database on `localhost:5433` and persistent named volumes for PostgreSQL, model cache, and audio. The existing host database on port 5432 was not changed.
- `VoiceClone\Vanna.m4a` is 138.39 seconds of AAC stereo. A 19.97-second excerpt was converted locally to mono 24 kHz PCM at `backend\voice_references\vanna.wav`; VoxCPM2 reference-only cloning uses this clip without a transcript.
- **វណ្ណា** is seeded as a built-in public voice preset so it appears in the voice selector for app users. The API does not expose its filesystem path.

## Test evidence

- Backend suite against Docker PostgreSQL covered signup/login and persisted identity, authenticated voice creation/read-back, unauthenticated speech-generation rejection, Khmer generation for `សួស្តី នេះជាការសាកល្បង`, WAV retrieval and parsing, persisted history read-back, and cleanup.
- The new speed/style integration flow generated the same Khmer sentence at 0.8x, 1.0x, and 1.2x, then at 1.0x with News, Vlogger, and Storyteller prompts. All six responses returned valid WAVs, and speed/style metadata was verified in history.
- Measured durations from the full suite were: speed — 0.8x: 2.20 s, 1.0x: 2.40 s, 1.2x: 2.27 s; style — News: 5.92 s, Vlogger: 6.24 s, Storyteller: 2.24 s. After rebuilding the final backend, the control test measured speed — 0.8x: 3.20 s, 1.0x: 2.56 s, 1.2x: 1.20 s; style — News: 7.04 s, Vlogger: 2.24 s, Storyteller: 4.48 s. The inconsistent durations show that independent VoxCPM2 inferences have variable base output lengths, so these runs do not establish repeatable inverse scaling. A separate pitch-preservation unit test stretched the same one-second 440 Hz signal to the expected durations (1.25 s, 1.00 s, 0.83 s) while retaining pitch; the time-stretch operation itself scales duration as expected.
- Style IDs and bilingual prompts were sent, accepted, and persisted. Audio samples were not perceptually rated, and the variable model outputs do not establish that the model reliably performed distinct acting styles; treat these presets as prompt guidance, not guaranteed voice transformations.
- The public voice-list endpoint returned all five public presets without authentication, including វណ្ណា with `has_reference_audio: true`; authenticated users still receive their own private voices.
- Backend clone test found វណ្ណា in the authenticated voice list, verified `has_reference_audio`, generated Khmer speech with that preset, and parsed the returned WAV.
- Browser test verified the live character counter and enabled Generate button; invalid-login feedback; signup/sign-in; preference restoration after reload; selection of វណ្ណា and News style; speed slider at 0.80x; style-preview API generation; first-run loading/progress feedback; HTTP 201 generation; nonempty `audio/wav` with valid frames; browser audio duration and playback progress; and visible history with speed/style metadata.
- Docker PostgreSQL passed the persistence flows. The API health/status endpoints and both frontend home pages returned HTTP 200; Compose health checks passed for all four long-running services.

## Environment and verdict

```text
environment:
  docker: AVAILABLE — Docker Engine 29.4.3; Compose 5.1.3
  node: AVAILABLE — v20.19.5
  playwright: AVAILABLE — Python Playwright 1.61.0; headless Chromium browser E2E passed
  infra-tier: PRIMARY(Docker PostgreSQL) — isolated Compose database on localhost:5433
  browser-tier: PRIMARY(Playwright)
startup: PASS — docker compose up -d; PostgreSQL, API, user app, and admin app healthy; all public endpoints returned HTTP 200
integration: PASS — exit_code: 0, passed: 10, failed: 0, skipped: 0; post-fix chunk/pacing tests, including joined-audio speed application. Earlier 19-test backend suite also passed against Compose PostgreSQL.
e2e: PASS — exit_code: 0; prior Playwright browser flow passed (1 test); latest authenticated exact-sample API run passed for 3/3 speeds with progress 8/8, WAV parsing, and history verification.
overall: NEEDS_SIGNOFF — long-text speed scaling now matches the requested ratios and post-fix thirds remain within 15%; pre-change thirds cannot be compared because the baseline WAV was not retained, and distinct style performance / join artifacts remain perceptually unconfirmed.
```

## Issues and fixes

| # | Severity | Description | Resolution |
|---|---|---|---|
| 1 | Environment | The previous C: cache contained a confirmed zero-byte VAE blob, and C: lacked enough space for the model. | Removed only that zero-byte cache entry and downloaded to E:. Both model files now pass size and hash verification. |
| 2 | Test-only | The first browser run ended with `TimeoutError: Locator.wait_for: Timeout 30000ms exceeded` while looking for the history item. Generation, WAV checks, and playback had already passed; the test had not opened the History tab. | Updated the test to select History before asserting the item; the rerun passed. |
| 3 | Command wrapper | The initial detached PowerShell download invocation reported exit code 1 with a `NativeCommandError` while the Hugging Face progress display wrote to stderr; the downloader log recorded snapshot and integrity success. | Re-ran the downloader against the completed cache; all checks passed with exit code 0. No remaining download error. |
| 4 | Docker build | VoxCPM's `setuptools_scm` could not infer a version because the source-only Docker build context has no `.git` metadata. | Added a package-specific fallback version for the local install in `backend/Dockerfile`; all Docker images then built successfully. |
| 5 | Test environment | Leftover local FastAPI and Next.js processes shadowed the Docker host ports, so initial persistence assertions targeted the wrong database. | Stopped only the confirmed local KalapakTTS processes on ports 3000, 3001, and 8000; verified host signup persisted in Docker PostgreSQL and reran all tests successfully. |
| 6 | Test dependency | The backend test virtual environment lacked Python Playwright even though Chromium and Playwright were available globally. | Installed the matching Playwright 1.61.0 package into the local test virtual environment; browser E2E passed. |
| 7 | Voice visibility | The anonymous generator showed only its static fallback because listing public voices required authentication. | Made authentication optional for the voice-list endpoint and load public voices on the signed-out page; all five presets now appear before sign-in, while private voices remain user-scoped. |
| 8 | Browser test setup | The first updated browser run failed at the final Generate click because a page reload cleared the editor text. The style preview and preference restore had already succeeded. | Re-entered the test sentence after reload and reran the browser test successfully. |
| 9 | Test database isolation | The backend test environment's default URL pointed to localhost:5432, not the Compose test database. | Set `DATABASE_URL` from Compose configuration to localhost:5433 for the test process and cleanup subprocesses; the local PostgreSQL database was not touched. |
| 10 | Non-fatal test warning | The first full backend run emitted one Numba `RuntimeWarning: invalid value encountered in cast` during the time-stretch unit test while all 10 tests passed. | The focused speed-processing rerun passed with an added finite-sample assertion and no warning; no non-finite output was observed. |
| 11 | Functional regression | The exact long sample did not scale inversely before the final speed fix because multiple chunks hit the per-chunk 1.25x limit. | Normalize chunk pace at 1.0x, then apply the clamped speed once to the joined WAV; the exact-sample duration ratios now match. |
| 12 | Test harness | An initial runner assertion counted raw CRLF line endings differently from the normalized code-point count; a later runner could not match the Khmer voice name after PowerShell piped source encoding. Both errors occurred before synthesis. | Transport the original fixture bytes and select the public reference voice by its `has_reference_audio` metadata; the full post-fix API run then passed. |

## Remaining gaps

- No sustained-load or concurrent-generation performance test was run.
- Fixed-seed exact-sample speed runs after the final patch produced inverse duration ratios. Without a fixed seed, stochastic model output can vary between requests.
- Style quality remains approximate and unconfirmed by perceptual evaluation.
