from __future__ import annotations

import argparse
import ctypes
import gc
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from safetensors import safe_open
from transformers import LlamaTokenizerFast

from validate_onnx_reference import (
    FIXTURE,
    MIN_COSINE,
    ONNX_SNAPSHOT,
    UPSTREAM_SNAPSHOT,
    VOXCPM_SRC,
    WATCH_COSINE,
    is_accepted,
    metrics,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "backend" / "model_cache" / "onnx-parity-artifacts"
CHECKPOINT = UPSTREAM_SNAPSHOT / "model.safetensors"
CONFIG = UPSTREAM_SNAPSHOT / "config.json"
MEMORY_FLOOR = 2 * 1024**3


@contextmanager
def default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def available_ram_bytes() -> int:
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed; cannot enforce the RAM safety floor.")
    return int(status.ullAvailPhys)


def check_ram(label: str) -> float:
    available_gib = available_ram_bytes() / 1024**3
    print(json.dumps({"event": "ram_check", "label": label, "available_gib": round(available_gib, 2)}))
    if available_gib * 1024**3 < MEMORY_FLOOR:
        raise MemoryError(f"Only {available_gib:.2f} GiB RAM available; refusing {label} below 2 GiB.")
    return available_gib


class RamMonitor:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._low = threading.Event()
        self.start_bytes = available_ram_bytes()
        self.minimum_bytes = self.start_bytes
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(0.25):
            current = available_ram_bytes()
            self.minimum_bytes = min(self.minimum_bytes, current)
            if current < MEMORY_FLOOR:
                self._low.set()

    def __enter__(self) -> "RamMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()

    def verify(self, label: str) -> None:
        self.minimum_bytes = min(self.minimum_bytes, available_ram_bytes())
        if self._low.is_set() or self.minimum_bytes < MEMORY_FLOOR:
            raise MemoryError(
                f"Available RAM fell below 2 GiB while {label}; minimum was "
                f"{self.minimum_bytes / 1024**3:.2f} GiB."
            )

    @property
    def minimum_gib(self) -> float:
        return self.minimum_bytes / 1024**3


def peak_process_working_set_bytes() -> int:
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    process = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(
        process,
        ctypes.byref(counters),
        counters.cb,
    ):
        raise OSError("GetProcessMemoryInfo failed; cannot report peak process RAM.")
    return int(counters.PeakWorkingSetSize)


def text_context(args: argparse.Namespace) -> tuple[str, list[int], LlamaTokenizerFast]:
    import sys

    sys.path.insert(0, str(VOXCPM_SRC))
    from voxcpm.model.utils import mask_multichar_chinese_tokens

    text = args.fixture.read_text(encoding="utf-8").strip()[: args.text_chars]
    tokenizer = LlamaTokenizerFast.from_pretrained(str(args.upstream_snapshot))
    token_ids = mask_multichar_chinese_tokens(tokenizer)(text) + [101]
    if not token_ids:
        raise ValueError("The fixture produced no token IDs.")
    return text, token_ids, tokenizer


def load_weights(
    module: torch.nn.Module,
    source_prefixes: dict[str, str],
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    target_state = module.state_dict()
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as source:
        source_keys = set(source.keys())
        missing = []
        with torch.no_grad():
            for name, target in target_state.items():
                source_name = name
                for local_prefix, source_prefix in source_prefixes.items():
                    if name.startswith(local_prefix):
                        source_name = source_prefix + name[len(local_prefix) :]
                        break
                if source_name not in source_keys:
                    missing.append(source_name)
                    continue
                tensor = source.get_tensor(source_name)
                if tensor.shape != target.shape:
                    raise RuntimeError(
                        f"Checkpoint shape mismatch for {source_name}: "
                        f"{tuple(tensor.shape)} != {tuple(target.shape)}"
                    )
                target.copy_(tensor.to(device=device, dtype=target.dtype))
                del tensor
        if missing:
            raise RuntimeError(f"PyTorch checkpoint is missing weights: {missing[:10]}")


def initialize_nonpersistent_rope_buffers(module: torch.nn.Module) -> None:
    for child in module.modules():
        rope = getattr(child, "rope_emb", None)
        if rope is None:
            continue
        device = rope.inv_freq.device
        dim = rope.dim
        inv_freq = 1.0 / (
            rope.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        with torch.no_grad():
            rope.inv_freq.copy_(inv_freq)
        rope._set_cos_sin_cache(
            seq_len=rope.max_position_embeddings,
            device=device,
            dtype=torch.float32,
        )


def select_device(args: argparse.Namespace, parameter_bytes: int, stage: str) -> str:
    if args.device == "cpu" or not torch.cuda.is_available():
        return "cpu"
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    required_bytes = parameter_bytes + 1024**3
    print(
        json.dumps(
            {
                "event": "gpu_preflight",
                "stage": stage,
                "free_gib": round(free_bytes / 1024**3, 2),
                "total_gib": round(total_bytes / 1024**3, 2),
                "estimated_weights_gib": round(parameter_bytes / 1024**3, 2),
                "required_with_headroom_gib": round(required_bytes / 1024**3, 2),
            }
        )
    )
    return "cuda" if free_bytes >= required_bytes else "cpu"


def module_parameter_bytes(module: torch.nn.Module, dtype: torch.dtype = torch.bfloat16) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    return sum(parameter.numel() for parameter in module.parameters()) * element_size


def check_cpu_materialization_budget(available_gib: float, parameter_bytes: int, stage: str) -> None:
    projected_gib = available_gib - parameter_bytes / 1024**3 - 0.5
    print(
        json.dumps(
            {
                "event": "cpu_memory_preflight",
                "stage": stage,
                "available_gib": round(available_gib, 2),
                "weights_gib": round(parameter_bytes / 1024**3, 2),
                "projected_after_load_gib": round(projected_gib, 2),
            }
        )
    )
    if projected_gib < 2:
        raise MemoryError(
            f"Estimated {stage} load would leave {projected_gib:.2f} GiB RAM; refusing below 2 GiB."
        )


def memory_report(
    stage: str,
    device: str,
    monitor: RamMonitor,
    ram_before_gib: float,
) -> dict[str, object]:
    ram_after_gib = available_ram_bytes() / 1024**3
    result: dict[str, object] = {
        "device": device,
        "available_ram_at_monitor_start_gib": round(monitor.start_bytes / 1024**3, 2),
        "available_ram_before_load_gib": round(ram_before_gib, 2),
        "minimum_available_ram_during_load_gib": round(monitor.minimum_gib, 2),
        "available_ram_after_load_gib": round(ram_after_gib, 2),
        "peak_process_working_set_gib": round(peak_process_working_set_bytes() / 1024**3, 2),
    }
    if device == "cuda":
        result["peak_cuda_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        result["peak_cuda_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        result["free_cuda_after_load_gib"] = round(torch.cuda.mem_get_info()[0] / 1024**3, 2)
    print(json.dumps({"event": "model_memory", "stage": stage, **result}))
    return result


def load_base_lm(args: argparse.Namespace, token_ids: list[int]):
    import sys

    sys.path.insert(0, str(VOXCPM_SRC))
    from voxcpm.modules.minicpm4 import MiniCPM4Config, MiniCPMModel

    ram_before = check_ram("PyTorch base-LM load")
    config = json.loads((args.upstream_snapshot / "config.json").read_text(encoding="utf-8"))
    lm_config = MiniCPM4Config.model_validate(config["lm_config"])
    max_length = max(len(token_ids) + 1, 128)
    original_dtype = torch.get_default_dtype()
    monitor = RamMonitor()
    model = None
    device_name = "cpu"
    try:
        # VoxCPM2 is a custom nn.Module, so meta initialization provides low_cpu_mem_usage semantics.
        with torch.device("meta"), default_dtype(torch.bfloat16):
            meta_model = MiniCPMModel(lm_config)
        estimated_bytes = module_parameter_bytes(meta_model)
        device_name = select_device(args, estimated_bytes, "base_lm")
        for attempt_device in (device_name, "cpu") if device_name == "cuda" else ("cpu",):
            if attempt_device == "cpu":
                ram_before = check_ram("PyTorch base-LM CPU materialization")
                check_cpu_materialization_budget(ram_before, estimated_bytes, "base_lm")
            monitor = RamMonitor()
            try:
                if attempt_device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                with monitor:
                    model = meta_model.to_empty(device=attempt_device)
                    initialize_nonpersistent_rope_buffers(model)
                    model.setup_cache(1, max_length, attempt_device, torch.bfloat16)
                    model.eval()
                    load_weights(
                        model,
                        {"": "base_lm."},
                        args.upstream_snapshot / "model.safetensors",
                        torch.device(attempt_device),
                    )
                    monitor.verify(f"PyTorch base-LM load on {attempt_device}")
                device_name = attempt_device
                break
            except torch.cuda.OutOfMemoryError:
                if attempt_device != "cuda":
                    raise
                del model
                model = None
                gc.collect()
                torch.cuda.empty_cache()
                with torch.device("meta"), default_dtype(torch.bfloat16):
                    meta_model = MiniCPMModel(lm_config)
                device_name = "cpu"
        if model is None:
            raise RuntimeError("Base-LM materialization did not produce a model.")
    finally:
        torch.set_default_dtype(original_dtype)
    report = memory_report("base_lm", device_name, monitor, ram_before)
    return model, report, config


def load_residual_modules(args: argparse.Namespace):
    import sys

    sys.path.insert(0, str(VOXCPM_SRC))
    from voxcpm.model.voxcpm2 import VoxCPMConfig
    from voxcpm.modules.minicpm4 import MiniCPMModel

    ram_before = check_ram("PyTorch residual-LM load")
    config = VoxCPMConfig.model_validate(json.loads((args.upstream_snapshot / "config.json").read_text("utf-8")))
    residual_config = config.lm_config.model_copy(deep=True)
    residual_config.num_hidden_layers = config.residual_lm_num_layers
    residual_config.vocab_size = 0
    residual_config.no_rope = config.residual_lm_no_rope

    class ResidualReference(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.residual_lm = MiniCPMModel(residual_config)
            self.fusion_concat_proj = torch.nn.Linear(config.lm_config.hidden_size * 2, config.lm_config.hidden_size)
            self.lm_to_dit_proj = torch.nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)
            self.res_to_dit_proj = torch.nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)
            self.stop_proj = torch.nn.Linear(config.lm_config.hidden_size, config.lm_config.hidden_size)
            self.stop_actn = torch.nn.SiLU()
            self.stop_head = torch.nn.Linear(config.lm_config.hidden_size, 2, bias=False)

    original_dtype = torch.get_default_dtype()
    monitor = RamMonitor()
    model = None
    device_name = "cpu"
    try:
        with torch.device("meta"), default_dtype(torch.bfloat16):
            meta_model = ResidualReference()
        estimated_bytes = module_parameter_bytes(meta_model)
        device_name = select_device(args, estimated_bytes, "residual_lm")
        for attempt_device in (device_name, "cpu") if device_name == "cuda" else ("cpu",):
            if attempt_device == "cpu":
                ram_before = check_ram("PyTorch residual-LM CPU materialization")
                check_cpu_materialization_budget(ram_before, estimated_bytes, "residual_lm")
            monitor = RamMonitor()
            try:
                if attempt_device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                with monitor:
                    model = meta_model.to_empty(device=attempt_device)
                    initialize_nonpersistent_rope_buffers(model)
                    model.residual_lm.setup_cache(
                        1,
                        max(128, args.text_chars * 4),
                        attempt_device,
                        torch.bfloat16,
                    )
                    model.eval()
                    load_weights(
                        model,
                        {"residual_lm.": "residual_lm."},
                        args.upstream_snapshot / "model.safetensors",
                        torch.device(attempt_device),
                    )
                    monitor.verify(f"PyTorch residual-LM load on {attempt_device}")
                device_name = attempt_device
                break
            except torch.cuda.OutOfMemoryError:
                if attempt_device != "cuda":
                    raise
                del model
                model = None
                gc.collect()
                torch.cuda.empty_cache()
                ram_before = check_ram("PyTorch residual-LM CPU fallback")
                with torch.device("meta"), default_dtype(torch.bfloat16):
                    meta_model = ResidualReference()
                device_name = "cpu"
        if model is None:
            raise RuntimeError("Residual-LM materialization did not produce a model.")
    finally:
        torch.set_default_dtype(original_dtype)
    report = memory_report("residual_lm", device_name, monitor, ram_before)
    return model, config, report


def load_dit_estimator(args: argparse.Namespace):
    import sys

    sys.path.insert(0, str(VOXCPM_SRC))
    from voxcpm.model.voxcpm2 import VoxCPMConfig
    from voxcpm.modules.locdit import VoxCPMLocDiTV2

    ram_before = check_ram("PyTorch DiT load")
    config = VoxCPMConfig.model_validate(json.loads((args.upstream_snapshot / "config.json").read_text("utf-8")))
    dit_config = config.lm_config.model_copy(deep=True)
    dit_config.hidden_size = config.dit_config.hidden_dim
    dit_config.intermediate_size = config.dit_config.ffn_dim
    dit_config.num_attention_heads = config.dit_config.num_heads
    dit_config.num_hidden_layers = config.dit_config.num_layers
    dit_config.kv_channels = config.dit_config.kv_channels
    dit_config.vocab_size = 0

    original_dtype = torch.get_default_dtype()
    monitor = RamMonitor()
    estimator = None
    device_name = "cpu"
    try:
        with torch.device("meta"), default_dtype(torch.bfloat16):
            meta_estimator = VoxCPMLocDiTV2(dit_config, in_channels=config.feat_dim)
        estimated_bytes = module_parameter_bytes(meta_estimator)
        device_name = select_device(args, estimated_bytes, "dit")
        for attempt_device in (device_name, "cpu") if device_name == "cuda" else ("cpu",):
            if attempt_device == "cpu":
                ram_before = check_ram("PyTorch DiT CPU materialization")
                check_cpu_materialization_budget(ram_before, estimated_bytes, "dit")
            monitor = RamMonitor()
            try:
                if attempt_device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                with monitor:
                    estimator = meta_estimator.to_empty(device=attempt_device).eval()
                    initialize_nonpersistent_rope_buffers(estimator)
                    load_weights(
                        estimator,
                        {"": "feat_decoder.estimator."},
                        args.upstream_snapshot / "model.safetensors",
                        torch.device(attempt_device),
                    )
                    monitor.verify(f"PyTorch DiT load on {attempt_device}")
                device_name = attempt_device
                break
            except torch.cuda.OutOfMemoryError:
                if attempt_device != "cuda":
                    raise
                del estimator
                estimator = None
                gc.collect()
                torch.cuda.empty_cache()
                ram_before = check_ram("PyTorch DiT CPU fallback")
                with torch.device("meta"), default_dtype(torch.bfloat16):
                    meta_estimator = VoxCPMLocDiTV2(dit_config, in_channels=config.feat_dim)
                device_name = "cpu"
        if estimator is None:
            raise RuntimeError("DiT materialization did not produce an estimator.")
    finally:
        torch.set_default_dtype(original_dtype)
    report = memory_report("dit", device_name, monitor, ram_before)
    return estimator, config, report


def pytorch_dit_step(
    estimator: torch.nn.Module,
    x: torch.Tensor,
    dit_hidden: torch.Tensor,
    cond: torch.Tensor,
    t: torch.Tensor,
    dt: torch.Tensor,
    cfg_value: float,
    mean_mode: bool,
) -> torch.Tensor:
    batch = x.size(0)
    x_in = torch.cat((x, x), dim=0)
    mu_in = torch.cat((dit_hidden, torch.zeros_like(dit_hidden)), dim=0)
    t_in = t.expand(2 * batch)
    dt_in = (
        dt.expand(2 * batch)
        if mean_mode
        else torch.zeros(2 * batch, dtype=x.dtype, device=x.device)
    )
    cond_in = torch.cat((cond, cond), dim=0)
    velocity = estimator(x_in, mu_in, t_in, cond_in, dt_in)
    positive, negative = velocity.split(batch, dim=0)
    positive_flat = positive.reshape(batch, -1)
    negative_flat = negative.reshape(batch, -1)
    scale = (positive_flat * negative_flat).sum(dim=1, keepdim=True)
    scale = scale / ((negative_flat**2).sum(dim=1, keepdim=True) + 1e-8)
    scale = scale.reshape(batch, *([1] * (positive.ndim - 1)))
    guided = negative * scale + cfg_value * (positive - negative * scale)
    return x - dt * guided


def cfm_schedule(
    steps: int,
    sway_sampling_coef: float,
    device: torch.device | str = "cpu",
) -> tuple[list[dict[str, float | int | bool]], int]:
    span = torch.linspace(1, 0, steps + 1, dtype=torch.bfloat16, device=device)
    span = span + sway_sampling_coef * (torch.cos(torch.pi / 2 * span) - 1 + span)
    zero_init_steps = max(1, int(len(span) * 0.04))
    t = span[0]
    dt = span[0] - span[1]
    schedule: list[dict[str, float | int | bool]] = []
    for step_index in range(1, steps + 1):
        torch_t = float(t.float().item())
        torch_dt = float(dt.float().item())
        onnx_t = float(np.float32(torch_t))
        onnx_dt = float(np.float32(torch_dt))
        schedule.append(
            {
                "step": step_index,
                "pytorch_t": torch_t,
                "onnx_t": onnx_t,
                "pytorch_dt": torch_dt,
                "onnx_dt": onnx_dt,
                "zero_init": step_index <= zero_init_steps,
                "schedule_inputs_match": onnx_t == torch_t and onnx_dt == torch_dt,
            }
        )
        t = t - dt
        if step_index < steps:
            dt = t - span[step_index + 1]
    return schedule, zero_init_steps


def stage_e_schedule(args: argparse.Namespace) -> int:
    schedules = {}
    schedule_device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    for steps in (4, 6, 10):
        schedule, zero_init_steps = cfm_schedule(steps, args.sway_sampling_coef, schedule_device)
        schedules[str(steps)] = {"zero_init_steps": zero_init_steps, "steps": schedule}
    result = {
                "stage": "e_sampler_contract",
                "result": "MEASURED",
                "source": (
                    "VoxCPM/src/voxcpm/modules/locdit/unified_cfm.py:51-76,84-136; "
                    "VoxCPM/src/voxcpm/model/voxcpm2.py:1087-1127"
                ),
                "sway_sampling_coef": args.sway_sampling_coef,
                "schedule_device": schedule_device,
                "t_shift": "none",
                "noise": {
                    "shape": [1, 64, 4],
                    "dtype": "bfloat16",
                    "temperature": 1.0,
                    "seed": "apply_generation_seed(seed), then torch.randn on model device",
                },
                "cfg": {
                    "cfg_value": args.cfg_value,
                    "cfg_zero_star": True,
                    "mean_mode": False,
                    "dit_dt_input": "zeroed when mean_mode is false; Euler dt still updates x",
                },
                "patch_conditioning": (
                    "initial cond is the final prompt feature; zero for zero-shot; "
                    "each generated patch becomes the next patch's cond"
                ),
                "schedules": schedules,
            }
    save_json(args.artifacts / "stage-e-sampler-schedule.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stage_b_torch(args: argparse.Namespace) -> int:
    text, token_ids, _ = text_context(args)
    model, load_memory, _ = load_base_lm(args, token_ids)
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
    started = time.perf_counter()
    execution_monitor = RamMonitor()
    with execution_monitor:
        with torch.inference_mode():
            ids = torch.tensor([token_ids], dtype=torch.long, device=next(model.parameters()).device)
            scale = model.config.scale_emb if model.config.use_mup else 1.0
            text_embeds = model.embed_tokens(ids) * scale
            hidden, cache = model(inputs_embeds=text_embeds, is_causal=True)
        execution_monitor.verify("base-LM reference inference")
    if hidden.is_cuda:
        torch.cuda.synchronize()
    artifact = args.artifacts / "stage-b-base-hidden.npz"
    arrays: dict[str, np.ndarray] = {
        "base_hidden": hidden.float().cpu().numpy(),
        "text_embeds": text_embeds.float().cpu().numpy(),
        "text": np.asarray(text),
        "token_ids": np.asarray(token_ids, dtype=np.int64),
    }
    for layer, (key, value) in enumerate(cache):
        arrays[f"present.{layer}.key"] = key.float().cpu().numpy()
        arrays[f"present.{layer}.value"] = value.float().cpu().numpy()
    save_npz(artifact, **arrays)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "stage": "b_reference_input",
                "result": "PYTORCH_BASE_HIDDEN_SAVED",
                "artifact": str(artifact),
                "text_codepoints": len(text),
                "token_count": len(token_ids),
                "elapsed_seconds": round(elapsed, 3),
                "memory": {
                    **load_memory,
                    "available_ram_after_inference_gib": round(available_ram_bytes() / 1024**3, 2),
                    "minimum_available_ram_during_inference_gib": round(execution_monitor.minimum_gib, 2),
                    "peak_process_working_set_gib": round(peak_process_working_set_bytes() / 1024**3, 2),
                    "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                    if hidden.is_cuda
                    else None,
                    "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 2)
                    if hidden.is_cuda
                    else None,
                },
            },
            ensure_ascii=False,
        )
    )
    device = next(model.parameters()).device
    del model, ids, text_embeds, hidden, cache, arrays
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 0


def stage_b_onnx(args: argparse.Namespace) -> int:
    artifact = args.artifacts / "stage-b-base-hidden.npz"
    if not artifact.is_file():
        raise FileNotFoundError(f"Run --phase b-torch first; missing {artifact}")
    ram_before = check_ram("base-LM ONNX session load")
    started = time.perf_counter()
    monitor = RamMonitor()
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = args.threads
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    with np.load(artifact, allow_pickle=False) as reference:
        q = int(reference["token_ids"].size)
        feed = {
            "text_embeds": reference["text_embeds"].astype(np.float16),
            "feat_embeds": np.zeros((1, q, 2048), dtype=np.float16),
            "audio_mask": np.zeros((1, q, 1), dtype=np.float16),
        }
        for layer in range(28):
            for kind in ("key", "value"):
                feed[f"past_key_values.{layer}.{kind}"] = np.zeros((1, 2, 0, 128), dtype=np.float16)
        with monitor:
            session = ort.InferenceSession(
                str(args.onnx_snapshot / "q4f16" / "base_lm.onnx"),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            monitor.verify("base-LM ONNX session load")
            outputs = session.run(None, feed)
        output_map = {item.name: value for item, value in zip(session.get_outputs(), outputs)}
        tensor_metrics = [("hidden", metrics(reference["base_hidden"], output_map["hidden"]))]
        for layer in range(28):
            for kind in ("key", "value"):
                name = f"present.{layer}.{kind}"
                tensor_metrics.append(
                    (name, metrics(reference[name], output_map[name]))
                )
    del session, outputs, output_map, feed
    gc.collect()
    elapsed = time.perf_counter() - started
    accepted = all(is_accepted(result) for _, result in tensor_metrics)
    print(
        json.dumps(
            {
                "stage": "b_base_lm_prefill_and_kv_cache",
                "result": "PASS-WITH-QUANTIZATION-NOISE" if accepted else "FAIL",
                "criteria": {"minimum_cosine": MIN_COSINE, "watch_below": WATCH_COSINE},
                "tensor_metrics": [{"name": name, **result} for name, result in tensor_metrics],
                "watch_tensors": [
                    name
                    for name, result in tensor_metrics
                    if result["cosine"] is not None and result["cosine"] < WATCH_COSINE
                ],
                "elapsed_seconds": round(elapsed, 3),
                "memory": {
                    "available_ram_before_load_gib": round(ram_before, 2),
                    "minimum_available_ram_gib": round(monitor.minimum_gib, 2),
                    "available_ram_after_load_gib": round(available_ram_bytes() / 1024**3, 2),
                    "peak_process_working_set_gib": round(peak_process_working_set_bytes() / 1024**3, 2),
                    "provider": "CPUExecutionProvider",
                },
            },
            ensure_ascii=False,
        )
    )
    return 0 if accepted else 2


def stage_c_torch(args: argparse.Namespace) -> int:
    text, token_ids, tokenizer = text_context(args)
    base_artifact = args.artifacts / "stage-b-base-hidden.npz"
    if not base_artifact.is_file():
        raise FileNotFoundError(f"Run --phase b-torch first; missing {base_artifact}")
    model, config, memory = load_residual_modules(args)
    with np.load(base_artifact, allow_pickle=False) as base_reference:
        base_hidden = torch.from_numpy(base_reference["base_hidden"]).to(
            device=next(model.parameters()).device,
            dtype=torch.bfloat16,
        )
        if not np.array_equal(base_reference["token_ids"], np.asarray(token_ids, dtype=np.int64)):
            raise ValueError("The saved base hidden tensors were generated from different token IDs.")
    if base_hidden.shape[1] != len(token_ids):
        raise ValueError(
            f"Base hidden sequence length {base_hidden.shape[1]} does not match tokenizer length {len(token_ids)}."
        )
    execution_monitor = RamMonitor()
    with execution_monitor:
        with torch.inference_mode():
            feat_mask = torch.zeros((1, len(token_ids)), dtype=torch.bfloat16, device=base_hidden.device)
            feat_embeds = torch.zeros_like(base_hidden)
            residual_inputs = model.fusion_concat_proj(torch.cat((base_hidden, feat_embeds), dim=-1))
            residual_sequence, residual_cache = model.residual_lm(
                inputs_embeds=residual_inputs,
                is_causal=True,
            )
            lm_hidden = base_hidden[:, -1, :]
            residual_hidden = residual_sequence[:, -1, :]
            dit_hidden = torch.cat(
                (model.lm_to_dit_proj(lm_hidden), model.res_to_dit_proj(residual_hidden)),
                dim=-1,
            )
            stop_logits = model.stop_head(model.stop_actn(model.stop_proj(lm_hidden)))
        execution_monitor.verify("residual-LM reference inference")
    arrays: dict[str, np.ndarray] = {
        "base_hidden": base_hidden.float().cpu().numpy().astype(np.float16),
        "feat_embeds": feat_embeds.float().cpu().numpy().astype(np.float16),
        "audio_mask": feat_mask.unsqueeze(-1).float().cpu().numpy().astype(np.float16),
        "dit_hidden": dit_hidden.float().cpu().numpy(),
        "stop_logits": stop_logits.float().cpu().numpy(),
        "text": np.asarray(text),
        "token_count": np.asarray(len(token_ids)),
        "seed": np.asarray(args.seeds[0], dtype=np.int64),
    }
    for layer, (key, value) in enumerate(residual_cache):
        arrays[f"present.{layer}.key"] = key.float().cpu().numpy()
        arrays[f"present.{layer}.value"] = value.float().cpu().numpy()
    artifact = args.artifacts / "stage-c-reference.npz"
    save_npz(artifact, **arrays)
    print(
        json.dumps(
            {
                "stage": "c",
                "result": "PYTORCH_REFERENCE_SAVED",
                "artifact": str(artifact),
                "text_codepoints": len(text),
                "token_count": len(token_ids),
                "seed": args.seeds[0],
                "memory": {
                    **memory,
                    "available_ram_after_inference_gib": round(available_ram_bytes() / 1024**3, 2),
                    "minimum_available_ram_during_inference_gib": round(execution_monitor.minimum_gib, 2),
                    "peak_process_working_set_gib": round(peak_process_working_set_bytes() / 1024**3, 2),
                    "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                    if next(model.parameters()).is_cuda
                    else None,
                    "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 2)
                    if next(model.parameters()).is_cuda
                    else None,
                },
            },
            ensure_ascii=False,
        )
    )
    device = next(model.parameters()).device
    del model, base_hidden, feat_mask, feat_embeds, residual_inputs, residual_sequence
    del residual_cache, lm_hidden, residual_hidden, dit_hidden, stop_logits, arrays
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 0


def stage_c_onnx(args: argparse.Namespace) -> int:
    artifact = args.artifacts / "stage-c-reference.npz"
    if not artifact.is_file():
        raise FileNotFoundError(f"Run --phase c-torch first; missing {artifact}")
    check_ram("residual-LM ONNX session load")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = args.threads
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    monitor = RamMonitor()
    with np.load(artifact, allow_pickle=False) as reference:
        feed = {
            "base_hidden": reference["base_hidden"],
            "feat_embeds": reference["feat_embeds"],
            "audio_mask": reference["audio_mask"],
        }
        for layer in range(8):
            for kind in ("key", "value"):
                feed[f"past_key_values.{layer}.{kind}"] = np.zeros((1, 2, 0, 128), dtype=np.float16)
        with monitor:
            session = ort.InferenceSession(
                str(args.onnx_snapshot / "q4f16" / "residual_lm_step.onnx"),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            monitor.verify("residual-LM ONNX session load")
            outputs = session.run(None, feed)
        output_map = {item.name: value for item, value in zip(session.get_outputs(), outputs)}
        tensor_metrics = [
            ("dit_hidden", metrics(reference["dit_hidden"], output_map["dit_hidden"])),
            ("stop_logits", metrics(reference["stop_logits"], output_map["stop_logits"])),
        ]
        for layer in range(8):
            for kind in ("key", "value"):
                name = f"present.{layer}.{kind}"
                tensor_metrics.append((name, metrics(reference[name], output_map[name])))
    del session, outputs, output_map
    import gc

    gc.collect()
    accepted = all(is_accepted(result) for _, result in tensor_metrics)
    print(
        json.dumps(
            {
                "stage": "c",
                "result": "PASS-WITH-QUANTIZATION-NOISE" if accepted else "FAIL",
                "criteria": {"minimum_cosine": MIN_COSINE, "watch_below": WATCH_COSINE},
                "tensor_metrics": [{"name": name, **result} for name, result in tensor_metrics],
                "watch_tensors": [
                    name
                    for name, result in tensor_metrics
                    if result["cosine"] is not None and result["cosine"] < WATCH_COSINE
                ],
                "minimum_available_gib": round(monitor.minimum_gib, 2),
                "peak_process_working_set_gib": round(peak_process_working_set_bytes() / 1024**3, 2),
            },
            ensure_ascii=False,
        )
    )
    return 0 if accepted else 2


def stage_d_torch(args: argparse.Namespace) -> int:
    dit_artifact = args.artifacts / "stage-c-reference.npz"
    if not dit_artifact.is_file():
        raise FileNotFoundError(f"Run --phase c-torch first; missing {dit_artifact}")
    estimator, config, memory = load_dit_estimator(args)
    with np.load(dit_artifact, allow_pickle=False) as c_reference:
        dit_hidden = torch.from_numpy(c_reference["dit_hidden"]).to(
            device=next(estimator.parameters()).device,
            dtype=torch.bfloat16,
        )
    execution_monitor = RamMonitor()
    with execution_monitor:
        with torch.inference_mode():
            device = dit_hidden.device
            cond = torch.zeros((1, 64, 4), dtype=torch.bfloat16, device=device)
            seed = args.seeds[0]
            generator = torch.Generator(device="cpu").manual_seed(seed)
            x = torch.randn((1, 64, 4), generator=generator, dtype=torch.bfloat16).to(device)
            span = torch.linspace(1, 0, 5, dtype=torch.bfloat16)
            span = span + torch.cos(torch.pi / 2 * span) - 1 + span
            t = span[0].to(device)
            dt = (span[0] - span[1]).to(device)
            x_next = pytorch_dit_step(
                estimator,
                x,
                dit_hidden,
                cond,
                t,
                dt,
                args.cfg_value,
                config.dit_config.dit_mean_mode,
            )
        execution_monitor.verify("DiT reference inference")
    artifact = args.artifacts / "stage-d-reference.npz"
    save_npz(
        artifact,
        x=x.float().cpu().numpy(),
        dit_hidden=dit_hidden.float().cpu().numpy(),
        cond=cond.float().cpu().numpy(),
        t=np.asarray(t.float().cpu().item(), dtype=np.float32),
        dt=np.asarray(dt.float().cpu().item(), dtype=np.float32),
        cfg=np.asarray(args.cfg_value, dtype=np.float32),
        x_next=x_next.transpose(1, 2).float().cpu().numpy(),
    )
    print(
        json.dumps(
            {
                "stage": "d",
                "result": "PYTORCH_REFERENCE_SAVED",
                "artifact": str(artifact),
                "seed": seed,
                "memory": {
                    **memory,
                    "available_ram_after_inference_gib": round(available_ram_bytes() / 1024**3, 2),
                    "minimum_available_ram_during_inference_gib": round(execution_monitor.minimum_gib, 2),
                    "peak_process_working_set_gib": round(peak_process_working_set_bytes() / 1024**3, 2),
                    "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                    if x.is_cuda
                    else None,
                    "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 2)
                    if x.is_cuda
                    else None,
                },
            }
        )
    )
    device = next(estimator.parameters()).device
    del estimator, dit_hidden, cond, x, span, t, dt, x_next
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 0


def stage_d_onnx(args: argparse.Namespace) -> int:
    artifact = args.artifacts / "stage-d-reference.npz"
    if not artifact.is_file():
        raise FileNotFoundError(f"Run --phase d-torch first; missing {artifact}")
    check_ram("DiT ONNX session load")
    with np.load(artifact, allow_pickle=False) as reference:
        monitor = RamMonitor()
        with monitor:
            session = ort.InferenceSession(
                str(args.onnx_snapshot / "q4f16" / "locdit_step.onnx"),
                providers=["CPUExecutionProvider"],
            )
            monitor.verify("DiT ONNX session load")
            output = session.run(
                ["x_next"],
                {
                    "x": reference["x"].transpose(0, 2, 1).astype(np.float32),
                    "dit_hidden": reference["dit_hidden"].astype(np.float16),
                    "cond": reference["cond"].transpose(0, 2, 1).astype(np.float32),
                    "t": reference["t"],
                    "dt": reference["dt"],
                    "cfg": reference["cfg"],
                },
            )[0]
        result = metrics(reference["x_next"], output)
    del session
    accepted = is_accepted(result)
    print(
        json.dumps(
            {
                "stage": "d",
                "result": "PASS-WITH-QUANTIZATION-NOISE" if accepted else "FAIL",
                "seed": args.seeds[0],
                "metrics": result,
                "watch": bool(result["cosine"] is not None and result["cosine"] < WATCH_COSINE),
                "minimum_available_gib": round(monitor.minimum_gib, 2),
                "peak_process_working_set_gib": round(peak_process_working_set_bytes() / 1024**3, 2),
            }
        )
    )
    return 0 if accepted else 2


def stage_e_torch(args: argparse.Namespace) -> int:
    dit_artifact = args.artifacts / "stage-c-reference.npz"
    if not dit_artifact.is_file():
        raise FileNotFoundError(f"Run --phase c-torch first; missing {dit_artifact}")
    import sys

    sys.path.insert(0, str(VOXCPM_SRC))
    from voxcpm.model.utils import apply_generation_seed

    estimator, config, memory = load_dit_estimator(args)
    with np.load(dit_artifact, allow_pickle=False) as c_reference:
        dit_hidden = torch.from_numpy(c_reference["dit_hidden"]).to(
            device=next(estimator.parameters()).device,
            dtype=torch.bfloat16,
        )
    execution_monitor = RamMonitor()
    with execution_monitor:
        with torch.inference_mode():
            device = dit_hidden.device
            arrays: dict[str, np.ndarray] = {
                "dit_hidden": dit_hidden.float().cpu().numpy(),
                "cfg": np.asarray(args.cfg_value, dtype=np.float32),
            }
            for seed in args.seeds:
                for steps in (4, 6, 10):
                    apply_generation_seed(seed)
                    schedule, zero_init_steps = cfm_schedule(steps, args.sway_sampling_coef, device)
                    arrays[f"zero_init.{seed}.{steps}"] = np.asarray(zero_init_steps, dtype=np.int32)
                    cond = torch.zeros((1, 64, 4), dtype=torch.bfloat16, device=device)
                    for patch_index in range(args.patch_count):
                        latent = torch.randn((1, 64, 4), dtype=torch.bfloat16, device=device)
                        arrays[f"initial.{seed}.{steps}.{patch_index}"] = (
                            latent.transpose(1, 2).float().cpu().numpy()
                        )
                        for item in schedule:
                            step_index = int(item["step"])
                            x_t = latent
                            raw_x_next = pytorch_dit_step(
                                estimator,
                                x_t,
                                dit_hidden,
                                cond,
                                torch.tensor(item["pytorch_t"], dtype=torch.bfloat16, device=device),
                                torch.tensor(item["pytorch_dt"], dtype=torch.bfloat16, device=device),
                                args.cfg_value,
                                config.dit_config.dit_mean_mode,
                            )
                            arrays[f"x_t.{seed}.{steps}.{patch_index}.{step_index}"] = (
                                x_t.transpose(1, 2).float().cpu().numpy()
                            )
                            arrays[f"cond.{seed}.{steps}.{patch_index}.{step_index}"] = (
                                cond.transpose(1, 2).float().cpu().numpy()
                            )
                            arrays[f"raw_x_next.{seed}.{steps}.{patch_index}.{step_index}"] = (
                                raw_x_next.transpose(1, 2).float().cpu().numpy()
                            )
                            if bool(item["zero_init"]):
                                latent = x_t
                            else:
                                latent = raw_x_next
                            arrays[
                                f"trajectory_x_next.{seed}.{steps}.{patch_index}.{step_index}"
                            ] = latent.transpose(1, 2).float().cpu().numpy()
                            arrays[f"t.{seed}.{steps}.{step_index}"] = np.asarray(
                                item["onnx_t"], dtype=np.float32
                            )
                            arrays[f"dt.{seed}.{steps}.{step_index}"] = np.asarray(
                                item["onnx_dt"], dtype=np.float32
                            )
                        arrays[f"final.{seed}.{steps}.{patch_index}"] = (
                            latent.transpose(1, 2).float().cpu().numpy()
                        )
                        cond = latent
        execution_monitor.verify("DiT latent reference inference")
    artifact = args.artifacts / "stage-e-reference.npz"
    save_npz(artifact, **arrays)
    result = {
                "stage": "e",
                "result": "PYTORCH_REFERENCE_SAVED",
                "artifact": str(artifact),
                "seeds": args.seeds,
                "step_counts": [4, 6, 10],
                "patch_count": args.patch_count,
                "seed_method": "upstream apply_generation_seed + device-local torch.randn",
                "sway_sampling_coef": args.sway_sampling_coef,
                "schedule_device": str(device),
                "memory": {
                    **memory,
                    "available_ram_after_inference_gib": round(available_ram_bytes() / 1024**3, 2),
                    "minimum_available_ram_during_inference_gib": round(execution_monitor.minimum_gib, 2),
                    "peak_process_working_set_gib": round(peak_process_working_set_bytes() / 1024**3, 2),
                    "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                    if dit_hidden.is_cuda
                    else None,
                    "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 2)
                    if dit_hidden.is_cuda
                    else None,
                },
            }
    save_json(args.artifacts / "stage-e-torch-results.json", result)
    print(json.dumps(result))
    device = next(estimator.parameters()).device
    del estimator, dit_hidden, cond, latent, arrays
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 0


def run_onnx_dit_step(
    session: ort.InferenceSession,
    x: np.ndarray,
    dit_hidden: np.ndarray,
    cond: np.ndarray,
    t: np.ndarray,
    dt: np.ndarray,
    cfg: np.ndarray,
) -> np.ndarray:
    return session.run(
        ["x_next"],
        {
            "x": x.astype(np.float32),
            "dit_hidden": dit_hidden.astype(np.float16),
            "cond": cond.astype(np.float32),
            "t": t,
            "dt": dt,
            "cfg": cfg,
        },
    )[0]


def stage_e_onnx(args: argparse.Namespace) -> int:
    artifact = args.artifacts / "stage-e-reference.npz"
    if not artifact.is_file():
        raise FileNotFoundError(f"Run --phase e-torch first; missing {artifact}")
    check_ram("DiT ONNX session load")
    monitor = RamMonitor()
    with monitor:
        session = ort.InferenceSession(
            str(args.onnx_snapshot / "q4f16" / "locdit_step.onnx"),
            providers=["CPUExecutionProvider"],
        )
        monitor.verify("DiT ONNX session load")
        with np.load(artifact, allow_pickle=False) as reference:
            results: list[dict[str, object]] = []
            teacher_forced_by_steps: dict[int, list[float]] = {4: [], 6: [], 10: []}
            free_running_by_steps: dict[int, list[float]] = {4: [], 6: [], 10: []}
            for seed in args.seeds:
                for steps in (4, 6, 10):
                    zero_init_steps = int(reference[f"zero_init.{seed}.{steps}"])
                    per_patch: list[dict[str, object]] = []
                    free_cond = np.zeros((1, 4, 64), dtype=np.float32)
                    free_latent = None
                    for patch_index in range(args.patch_count):
                        free_latent = reference[
                            f"initial.{seed}.{steps}.{patch_index}"
                        ].astype(np.float32)
                        teacher_steps: list[dict[str, object]] = []
                        free_steps: list[dict[str, object]] = []
                        for step_index in range(1, steps + 1):
                            x_t = reference[
                                f"x_t.{seed}.{steps}.{patch_index}.{step_index}"
                            ].astype(np.float32)
                            teacher_cond = reference[
                                f"cond.{seed}.{steps}.{patch_index}.{step_index}"
                            ].astype(np.float32)
                            raw_x_next = reference[
                                f"raw_x_next.{seed}.{steps}.{patch_index}.{step_index}"
                            ]
                            onnx_teacher_next = run_onnx_dit_step(
                                session,
                                x_t,
                                reference["dit_hidden"],
                                teacher_cond,
                                reference[f"t.{seed}.{steps}.{step_index}"],
                                reference[f"dt.{seed}.{steps}.{step_index}"],
                                reference["cfg"],
                            )
                            teacher_result = metrics(raw_x_next, onnx_teacher_next)
                            teacher_steps.append({"step": step_index, **teacher_result})
                            if teacher_result["cosine"] is not None:
                                teacher_forced_by_steps[steps].append(float(teacher_result["cosine"]))

                            if step_index > zero_init_steps:
                                free_latent = run_onnx_dit_step(
                                    session,
                                    free_latent,
                                    reference["dit_hidden"],
                                    free_cond,
                                    reference[f"t.{seed}.{steps}.{step_index}"],
                                    reference[f"dt.{seed}.{steps}.{step_index}"],
                                    reference["cfg"],
                                )
                            free_result = metrics(
                                reference[
                                    f"trajectory_x_next.{seed}.{steps}.{patch_index}.{step_index}"
                                ],
                                free_latent,
                            )
                            free_steps.append({"step": step_index, **free_result})
                        final_result = metrics(
                            reference[f"final.{seed}.{steps}.{patch_index}"],
                            free_latent,
                        )
                        per_patch.append(
                            {
                                "patch": patch_index,
                                "teacher_forced_steps": teacher_steps,
                                "free_running_steps": free_steps,
                                "free_running_final": final_result,
                            }
                        )
                        free_cond = free_latent
                    if free_latent is not None:
                        final_result = metrics(
                            reference[f"final.{seed}.{steps}.{args.patch_count - 1}"],
                            free_latent,
                        )
                        if final_result["cosine"] is not None:
                            free_running_by_steps[steps].append(float(final_result["cosine"]))
                    results.append(
                        {
                            "seed": seed,
                            "steps": steps,
                            "zero_init_steps": zero_init_steps,
                            "patches": per_patch,
                            "final_latent": final_result,
                        }
                    )
    del session
    step_summary = {
        str(steps): {
            "teacher_forced_single_step_cosine_mean": (
                float(np.mean(teacher_forced_by_steps[steps]))
                if teacher_forced_by_steps[steps]
                else None
            ),
            "teacher_forced_single_step_cosine_min": (
                float(np.min(teacher_forced_by_steps[steps]))
                if teacher_forced_by_steps[steps]
                else None
            ),
            "all_teacher_forced_steps_ge_0_99": bool(
                teacher_forced_by_steps[steps]
                and min(teacher_forced_by_steps[steps]) >= 0.99
            ),
            "free_running_final_cosine_mean": (
                float(np.mean(free_running_by_steps[steps]))
                if free_running_by_steps[steps]
                else None
            ),
            "free_running_final_cosine_min": (
                float(np.min(free_running_by_steps[steps]))
                if free_running_by_steps[steps]
                else None
            ),
        }
        for steps in (4, 6, 10)
    }
    result = {
                "stage": "e",
                "result": "DIAGNOSTIC_COMPLETE",
                "criteria": {
                    "teacher_forced_single_step_cosine": ">=0.99 to attribute drift to accumulation",
                    "free_running_cosine": "reported diagnostically; not a standalone pass/fail gate",
                },
                "patch_count": args.patch_count,
                "results": results,
                "summary_by_step_count": step_summary,
                "minimum_available_gib": round(monitor.minimum_gib, 2),
                "peak_process_working_set_gib": round(peak_process_working_set_bytes() / 1024**3, 2),
            }
    save_json(args.artifacts / "stage-e-onnx-results.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


PHASES = {
    "b-torch": stage_b_torch,
    "b-onnx": stage_b_onnx,
    "c-torch": stage_c_torch,
    "c-onnx": stage_c_onnx,
    "d-torch": stage_d_torch,
    "d-onnx": stage_d_onnx,
    "e-schedule": stage_e_schedule,
    "e-torch": stage_e_torch,
    "e-onnx": stage_e_onnx,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Memory-light, process-isolated ONNX parity stages.")
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--upstream-snapshot", type=Path, default=UPSTREAM_SNAPSHOT)
    parser.add_argument("--onnx-snapshot", type=Path, default=ONNX_SNAPSHOT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--text-chars", type=int, default=35)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 1234, 2025, 20261121, 20261122])
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=1.0)
    parser.add_argument("--patch-count", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    try:
        return PHASES[args.phase](args)
    except MemoryError as error:
        print(json.dumps({"result": "ABORTED_LOW_RAM", "error": str(error)}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
