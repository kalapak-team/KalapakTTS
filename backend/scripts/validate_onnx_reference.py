from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from safetensors import safe_open
from transformers import LlamaTokenizerFast


ROOT = Path(__file__).resolve().parents[2]
MODEL_CACHE = ROOT / "backend" / "model_cache"
VOXCPM_SRC = ROOT / "VoxCPM" / "src"
UPSTREAM_SNAPSHOT = (
    MODEL_CACHE
    / "hub"
    / "models--openbmb--VoxCPM2"
    / "snapshots"
    / "7936080b41163874f38610f46f438e2c0757f9fd"
)
ONNX_SNAPSHOT = MODEL_CACHE / "onnx-uddom-voxcpm2-ad57fe5"
FIXTURE = ROOT / "backend" / "tests" / "fixtures" / "khmer_long_sample.txt"
MIN_COSINE = 0.98
WATCH_COSINE = 0.99


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | bool | None]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} != {candidate.shape}")
    finite = bool(np.isfinite(reference).all() and np.isfinite(candidate).all())
    if not finite:
        return {
            "finite": False,
            "max_abs": None,
            "mean_abs": None,
            "relative_l2": None,
            "cosine": None,
        }
    difference = candidate - reference
    reference_norm = np.linalg.norm(reference.ravel())
    candidate_norm = np.linalg.norm(candidate.ravel())
    cosine = (
        float(np.dot(reference.ravel(), candidate.ravel()) / (reference_norm * candidate_norm))
        if reference_norm and candidate_norm
        else 1.0
    )
    return {
        "finite": True,
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference.ravel()) / max(float(reference_norm), 1e-12)),
        "cosine": cosine,
    }


def is_accepted(result: dict[str, float | bool | None]) -> bool:
    cosine = result["cosine"]
    return bool(result["finite"] and cosine is not None and cosine >= MIN_COSINE)


class AudioVaeMetadata:
    chunk_size = 4
    decode_chunk_size = 4
    sample_rate = 16_000
    out_sample_rate = 16_000


def load_reference_model(
    checkpoint_path: Path,
    config_path: Path,
    tokenizer: LlamaTokenizerFast,
    max_length: int,
) -> torch.nn.Module:
    sys.path.insert(0, str(VOXCPM_SRC))
    from voxcpm.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["max_length"] = max_length
    config["device"] = "cpu"
    config["dtype"] = "bfloat16"
    model_config = VoxCPMConfig.model_validate(config)

    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = VoxCPM2Model(
            model_config,
            tokenizer=tokenizer,
            audio_vae=AudioVaeMetadata(),
            device="cpu",
        )
    finally:
        torch.set_default_dtype(original_dtype)
    model.eval()
    target_state = model.state_dict()

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as source:
        source_keys = set(source.keys())
        missing = sorted(set(target_state) - source_keys)
        if missing:
            raise RuntimeError(f"PyTorch checkpoint is missing model keys: {missing[:10]}")
        with torch.no_grad():
            for name, target in target_state.items():
                tensor = source.get_tensor(name)
                if tensor.shape != target.shape:
                    raise RuntimeError(
                        f"Checkpoint shape mismatch for {name}: {tuple(tensor.shape)} != {tuple(target.shape)}"
                    )
                target.copy_(tensor)
    return model


def emit_stage(step: str, tensor_metrics: list[tuple[str, dict[str, float | bool | None]]], **details: object) -> bool:
    accepted = all(is_accepted(result) for _, result in tensor_metrics)
    watch_tensors = [
        name
        for name, result in tensor_metrics
        if result["cosine"] is not None and result["cosine"] < WATCH_COSINE
    ]
    print(
        json.dumps(
            {
                "step": step,
                "criteria": {
                    "minimum_cosine": MIN_COSINE,
                    "watch_below": WATCH_COSINE,
                    "all_values_finite": True,
                },
                "tensor_metrics": [{"name": name, **result} for name, result in tensor_metrics],
                "watch_tensors": watch_tensors,
                "status": "PASS-WITH-QUANTIZATION-NOISE" if accepted else "FAIL",
                **details,
            },
            ensure_ascii=False,
        )
    )
    return accepted


def cache_feed(feed: dict[str, np.ndarray], layer_count: int) -> None:
    for layer in range(layer_count):
        for kind in ("key", "value"):
            feed[f"past_key_values.{layer}.{kind}"] = np.zeros((1, 2, 0, 128), dtype=np.float16)


def guided_dit_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    dit_hidden: torch.Tensor,
    cond: torch.Tensor,
    t: torch.Tensor,
    dt: torch.Tensor,
    cfg_value: float,
) -> torch.Tensor:
    batch = x.size(0)
    x_in = torch.cat((x, x), dim=0)
    mu_in = torch.cat((dit_hidden, torch.zeros_like(dit_hidden)), dim=0)
    t_in = t.expand(2 * batch)
    dt_in = torch.zeros_like(t_in)
    cond_in = torch.cat((cond, cond), dim=0)
    velocity = model.feat_decoder.estimator(x_in, mu_in, t_in, cond_in, dt_in)
    positive, negative = velocity.split(batch, dim=0)
    positive_flat = positive.reshape(batch, -1)
    negative_flat = negative.reshape(batch, -1)
    scale = model.feat_decoder.optimized_scale(positive_flat, negative_flat)
    scale = scale.reshape(batch, *([1] * (positive.ndim - 1)))
    guided = negative * scale + cfg_value * (positive - negative * scale)
    return x - dt * guided


def run(args: argparse.Namespace) -> int:
    fixture_text = args.fixture.read_text(encoding="utf-8").strip()
    text = fixture_text[: args.text_chars]

    sys.path.insert(0, str(VOXCPM_SRC))
    from voxcpm.model.utils import mask_multichar_chinese_tokens

    tokenizer = LlamaTokenizerFast.from_pretrained(str(args.upstream_snapshot))
    token_ids = mask_multichar_chinese_tokens(tokenizer)(text) + [101]
    if not token_ids:
        raise ValueError("The fixture produced no token IDs.")

    manifest = json.loads((args.onnx_snapshot / "manifest.json").read_text(encoding="utf-8"))
    embedding_config = manifest["config"]["embedding"]
    embedding = np.memmap(
        args.onnx_snapshot / embedding_config["file"],
        mode="r",
        dtype="<f2",
        shape=tuple(embedding_config["shape"]),
    )
    candidate_embeddings = np.array(embedding[token_ids], dtype=np.float32, copy=True)

    checkpoint_path = args.upstream_snapshot / "model.safetensors"
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as source:
        original_embeddings = source.get_tensor("base_lm.embed_tokens.weight")[token_ids].float().numpy()
    embedding_metrics = metrics(original_embeddings, candidate_embeddings)
    fp16_roundtrip = np.asarray(original_embeddings, dtype=np.float16).astype(np.float32)
    embedding_pass = bool(np.isfinite(candidate_embeddings).all() and np.array_equal(candidate_embeddings, fp16_roundtrip))
    print(
        json.dumps(
            {
                "step": "a_text_embedding",
                "text_codepoints": len(text),
                "token_count_including_audio_start": len(token_ids),
                "source_embedding_dtype": "bfloat16",
                "onnx_embedding_dtype": embedding_config["dtype"],
                "metrics_vs_pytorch": embedding_metrics,
                "exact_after_fp16_conversion": bool(embedding_pass),
            },
            ensure_ascii=False,
        )
    )
    if not embedding_pass:
        print("STOP: text embedding lookup differs from the PyTorch embedding after FP16 conversion.")
        return 2

    max_length = max(len(token_ids) + 1, 128)
    start = time.perf_counter()
    model = load_reference_model(
        checkpoint_path,
        args.upstream_snapshot / "config.json",
        tokenizer,
        max_length,
    )
    pytorch_load_seconds = time.perf_counter() - start

    ids_tensor = torch.tensor([token_ids], dtype=torch.long)
    with torch.inference_mode():
        scale = model.config.lm_config.scale_emb if model.config.lm_config.use_mup else 1.0
        text_embeds = model.base_lm.embed_tokens(ids_tensor) * scale
        pytorch_hidden, pytorch_cache = model.base_lm(inputs_embeds=text_embeds, is_causal=True)

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = args.threads
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    graph_path = args.onnx_snapshot / "q4f16" / "base_lm.onnx"
    start = time.perf_counter()
    session = ort.InferenceSession(
        str(graph_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    session_load_seconds = time.perf_counter() - start

    q = len(token_ids)
    feed: dict[str, np.ndarray] = {
        "text_embeds": np.asarray(candidate_embeddings[None, ...], dtype=np.float16),
        "feat_embeds": np.zeros((1, q, 2048), dtype=np.float16),
        "audio_mask": np.zeros((1, q, 1), dtype=np.float16),
    }
    cache_feed(feed, model.config.lm_config.num_hidden_layers)

    start = time.perf_counter()
    onnx_outputs = session.run(None, feed)
    onnx_run_seconds = time.perf_counter() - start
    output_map = {item.name: value for item, value in zip(session.get_outputs(), onnx_outputs)}

    compared = {
        "hidden": metrics(pytorch_hidden.float().numpy(), output_map["hidden"]),
    }
    cache_metrics: list[tuple[str, dict[str, float | bool | None]]] = []
    for layer, (key_cache, value_cache) in enumerate(pytorch_cache):
        for kind, expected in (("key", key_cache), ("value", value_cache)):
            name = f"present.{layer}.{kind}"
            cache_metrics.append((name, metrics(expected.float().numpy(), output_map[name])))
    all_tensor_metrics = [("hidden", compared["hidden"]), *cache_metrics]
    watch_tensors = [
        name
        for name, result in all_tensor_metrics
        if result["cosine"] is not None and result["cosine"] < WATCH_COSINE
    ]
    cache_summary = {
        "tensor_count": len(cache_metrics),
        "max_abs_max": max(item["max_abs"] for _, item in cache_metrics),
        "mean_abs_mean": float(np.mean([item["mean_abs"] for _, item in cache_metrics])),
        "relative_l2_max": max(item["relative_l2"] for _, item in cache_metrics),
        "cosine_min": min(item["cosine"] for _, item in cache_metrics),
    }
    accepted = all(is_accepted(result) for _, result in all_tensor_metrics)
    print(
        json.dumps(
            {
                "step": "b_base_lm_prefill_and_kv_cache",
                "q": q,
                "pytorch_load_seconds": pytorch_load_seconds,
                "onnx_session_load_seconds": session_load_seconds,
                "onnx_run_seconds": onnx_run_seconds,
                "criteria": {
                    "minimum_cosine": MIN_COSINE,
                    "watch_below": WATCH_COSINE,
                    "all_values_finite": True,
                },
                "hidden_metrics": compared["hidden"],
                "kv_cache_metrics": cache_summary,
                "kv_tensor_metrics": [{"name": name, **result} for name, result in cache_metrics],
                "watch_tensors": watch_tensors,
                "status": "PASS-WITH-QUANTIZATION-NOISE" if accepted else "FAIL",
            },
            ensure_ascii=False,
        )
    )
    if not accepted:
        print(f"STOP: one or more base-LM/KV tensors are non-finite or have cosine below {MIN_COSINE}.")
        return 2

    del session, onnx_outputs, output_map, pytorch_cache
    gc.collect()

    feat_mask = torch.zeros((1, len(token_ids)), dtype=torch.bfloat16)
    text_mask = torch.ones_like(feat_mask)
    feat_embeds = torch.zeros_like(pytorch_hidden)
    with torch.inference_mode():
        base_hidden = (
            model.fsq_layer(pytorch_hidden) * feat_mask.unsqueeze(-1)
            + pytorch_hidden * text_mask.unsqueeze(-1)
        )
        residual_inputs = model.fusion_concat_proj(
            torch.cat((base_hidden, feat_mask.unsqueeze(-1) * feat_embeds), dim=-1)
        )
        residual_hidden_sequence, residual_cache = model.residual_lm(
            inputs_embeds=residual_inputs,
            is_causal=True,
        )
        residual_hidden = residual_hidden_sequence[:, -1, :]
        lm_hidden = base_hidden[:, -1, :]
        pytorch_dit_hidden = torch.cat(
            (model.lm_to_dit_proj(lm_hidden), model.res_to_dit_proj(residual_hidden)),
            dim=-1,
        )
        pytorch_stop_logits = model.stop_head(model.stop_actn(model.stop_proj(lm_hidden)))

    residual_feed: dict[str, np.ndarray] = {
        "base_hidden": base_hidden.float().numpy().astype(np.float16),
        "feat_embeds": feat_embeds.float().numpy().astype(np.float16),
        "audio_mask": feat_mask.unsqueeze(-1).float().numpy().astype(np.float16),
    }
    cache_feed(residual_feed, model.config.residual_lm_num_layers)
    residual_graph = args.onnx_snapshot / "q4f16" / "residual_lm_step.onnx"
    residual_session = ort.InferenceSession(
        str(residual_graph),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    residual_values = residual_session.run(None, residual_feed)
    residual_output_map = {
        item.name: value for item, value in zip(residual_session.get_outputs(), residual_values)
    }
    residual_metrics = [
        ("dit_hidden", metrics(pytorch_dit_hidden.float().numpy(), residual_output_map["dit_hidden"])),
        ("stop_logits", metrics(pytorch_stop_logits.float().numpy(), residual_output_map["stop_logits"])),
    ]
    for layer, (key_cache, value_cache) in enumerate(residual_cache):
        for kind, expected in (("key", key_cache), ("value", value_cache)):
            name = f"present.{layer}.{kind}"
            residual_metrics.append((name, metrics(expected.float().numpy(), residual_output_map[name])))
    if not emit_stage(
        "c_residual_lm_teacher_forced",
        residual_metrics,
        q=len(token_ids),
        seed=args.seeds[0],
    ):
        print(f"STOP: residual-LM output is non-finite or has cosine below {MIN_COSINE}.")
        return 2

    del residual_session, residual_values, residual_output_map, residual_cache
    gc.collect()

    seed = args.seeds[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.inference_mode():
        x = torch.randn((1, 64, 4), generator=generator, dtype=torch.bfloat16)
        cond = torch.zeros((1, 64, 4), dtype=torch.bfloat16)
        t_span = torch.linspace(1, 0, 5, dtype=torch.bfloat16)
        t_span = t_span + torch.cos(torch.pi / 2 * t_span) - 1 + t_span
        t = t_span[0]
        dt = t_span[0] - t_span[1]
        expected_x_next = guided_dit_step(
            model,
            x,
            pytorch_dit_hidden,
            cond,
            t,
            dt,
            args.cfg_value,
        )

    locdit_graph = args.onnx_snapshot / "q4f16" / "locdit_step.onnx"
    locdit_session = ort.InferenceSession(
        str(locdit_graph),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    def run_locdit(
        input_x: torch.Tensor | np.ndarray,
        input_t: torch.Tensor,
        input_dt: torch.Tensor,
    ) -> np.ndarray:
        x_numpy = (
            input_x.detach().float().numpy()
            if isinstance(input_x, torch.Tensor)
            else np.asarray(input_x, dtype=np.float32)
        )
        return locdit_session.run(
            ["x_next"],
            {
                "x": x_numpy.transpose(1, 2).astype(np.float32),
                "dit_hidden": pytorch_dit_hidden.float().numpy().astype(np.float16),
                "cond": cond.transpose(1, 2).float().numpy().astype(np.float32),
                "t": np.asarray(input_t.float().item(), dtype=np.float32),
                "dt": np.asarray(input_dt.float().item(), dtype=np.float32),
                "cfg": np.asarray(args.cfg_value, dtype=np.float32),
            },
        )[0]

    onnx_x_next = run_locdit(x, t, dt)
    dit_metrics = metrics(expected_x_next.transpose(1, 2).float().numpy(), onnx_x_next)
    if not emit_stage(
        "d_locdit_active_step_teacher_forced",
        [("x_next", dit_metrics)],
        seed=seed,
        cfg_value=args.cfg_value,
        t=float(t),
        dt=float(dt),
        x_shape=list(onnx_x_next.shape),
    ):
        print(f"STOP: DiT output is non-finite or has cosine below {MIN_COSINE}.")
        return 2

    latent_results: list[dict[str, object]] = []
    for seed in args.seeds:
        for steps in (4, 6, 10):
            e_generator = torch.Generator(device="cpu").manual_seed(seed)
            with torch.inference_mode():
                latent = torch.randn((1, 64, 4), generator=e_generator, dtype=torch.bfloat16)
                span = torch.linspace(1, 0, steps + 1, dtype=torch.bfloat16)
                span = span + torch.cos(torch.pi / 2 * span) - 1 + span
                zero_init_steps = max(1, int(len(span) * 0.04))
                onnx_latent = latent.transpose(1, 2).float().numpy().astype(np.float32)
                step_results: list[dict[str, object]] = []
                for step_index in range(1, len(span)):
                    step_t = span[0]
                    step_dt = span[0] - span[1]
                    if step_index <= zero_init_steps:
                        expected_latent = latent
                    else:
                        expected_latent = guided_dit_step(
                            model,
                            latent,
                            pytorch_dit_hidden,
                            cond,
                            step_t,
                            step_dt,
                            args.cfg_value,
                        )
                    latent = expected_latent
                    if step_index > zero_init_steps:
                        onnx_latent = run_locdit(
                            onnx_latent.transpose(0, 2, 1),
                            step_t,
                            step_dt,
                        )
                    stage_metrics = metrics(latent.transpose(1, 2).float().numpy(), onnx_latent)
                    step_results.append({"step": step_index, **stage_metrics})
                    if not is_accepted(stage_metrics):
                        break
            stage_pass = all(is_accepted(item) for item in step_results)
            latent_results.append(
                {
                    "steps": steps,
                    "seed": seed,
                    "zero_init_steps": zero_init_steps,
                    "per_step": step_results,
                    "status": "PASS-WITH-QUANTIZATION-NOISE" if stage_pass else "FAIL",
                }
            )
            print(json.dumps({"step": "e_latent_sampler", **latent_results[-1]}, ensure_ascii=False))
            if not stage_pass:
                print(
                    f"STOP: latent trajectory at {steps} steps, seed {seed}, "
                    f"fell below cosine {MIN_COSINE} or was non-finite."
                )
                return 2

    del locdit_session
    return 0
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-fast PyTorch/ONNX reference parity check.")
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--upstream-snapshot", type=Path, default=UPSTREAM_SNAPSHOT)
    parser.add_argument("--onnx-snapshot", type=Path, default=ONNX_SNAPSHOT)
    parser.add_argument("--text-chars", type=int, default=35)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 1234, 2025, 20261121, 20261122])
    parser.add_argument("--cfg-value", type=float, default=2.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
