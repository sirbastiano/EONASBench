"""
Verification utility for the reduced HPC macro-profile.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from bench.profile import count_params
from models.build import build_model


def maybe_get_gpu_name() -> str | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def make_output_dir(root: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(root) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def estimate_attention_bytes(height: int, width: int, heads: int, bytes_per_element: int) -> int:
    tokens = height * width
    return heads * tokens * tokens * bytes_per_element


def estimate_worst_case_vram_bytes(dtype: torch.dtype) -> dict:
    bytes_per_element = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    stage1_attn = estimate_attention_bytes(32, 32, 8, bytes_per_element)
    estimate = {
        "dtype_bytes": bytes_per_element,
        "stage1_attention_bytes": stage1_attn,
        "stage1_attention_mb": stage1_attn / (1024 ** 2),
        "saved_attention_activations_mb": [120, 180],
        "backbone_activations_mb": [80, 120],
        "upernet_activations_mb": [80, 120],
        "params_grads_optimizer_mb": [50, 150],
        "runtime_margin_mb": [250, 350],
        "estimated_peak_vram_mb": [600, 900],
        "hard_limit_vram_mb": 1024,
    }
    return estimate


def training_step(model: torch.nn.Module, device: torch.device, img_size: tuple[int, int], dtype: torch.dtype, num_classes: int) -> dict:
    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    x = torch.randn(1, 3, img_size[0], img_size[1], device=device)
    cls_target = torch.randint(0, num_classes, (1,), device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and dtype == torch.float16))
    autocast_dtype = dtype if use_amp else None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    try:
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
            output = model(x)
            seg = output["seg"]
            cls = output["cls"]
            seg_target = torch.randint(0, num_classes, (1, seg.shape[2], seg.shape[3]), device=device)
            seg_loss = torch.nn.functional.cross_entropy(seg, seg_target)
            cls_loss = torch.nn.functional.cross_entropy(cls, cls_target)
            loss = seg_loss + cls_loss
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_time = time.perf_counter() - start

        allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        return {
            "loss": float(loss.detach().cpu()),
            "step_time_sec": step_time,
            "max_memory_allocated_bytes": int(allocated),
            "max_memory_reserved_bytes": int(reserved),
            "seg_shape": list(seg.shape),
            "cls_shape": list(cls.shape),
            "oom": False,
        }
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            allocated = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
            torch.cuda.empty_cache()
        else:
            allocated = 0
            reserved = 0
        return {
            "error": str(exc),
            "step_time_sec": None,
            "max_memory_allocated_bytes": int(allocated),
            "max_memory_reserved_bytes": int(reserved),
            "oom": True,
        }


def parse_dtype(value: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[value.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {value}") from exc


def collect_environment(config_path: str, dtype: torch.dtype) -> dict:
    return {
        "timestamp": datetime.now().isoformat(),
        "config_path": config_path,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu_name": maybe_get_gpu_name(),
        "dtype": str(dtype),
        "git_commit": get_git_commit(),
    }


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid]


def maybe_count_flops(model: torch.nn.Module, img_size: tuple[int, int]) -> str | None:
    try:
        from ptflops import get_model_complexity_info

        macs, _ = get_model_complexity_info(
            model,
            (3, img_size[0], img_size[1]),
            as_strings=True,
            print_per_layer_stat=False,
        )
        return macs
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify reduced HPC macro-profile memory and throughput.")
    parser.add_argument("--config", default="configs/search_macro_reduced.yaml")
    parser.add_argument("--img", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-root", default="outputs/baseline")
    parser.add_argument("--vram-limit-mb", type=float, default=1024.0)
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)
    output_dir = make_output_dir(args.output_root)
    environment = collect_environment(args.config, dtype)
    analytic_estimate = estimate_worst_case_vram_bytes(dtype)
    results = {
        "environment": environment,
        "analytic_estimate": analytic_estimate,
        "runs": [],
    }
    config_path = Path(args.config)
    if config_path.exists():
        (output_dir / config_path.name).write_text(config_path.read_text())

    if not torch.cuda.is_available():
        results["error"] = "CUDA is not available; measured VRAM verification requires a GPU."
        (output_dir / "verification.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
        return 1

    device = torch.device("cuda")
    for run_idx in range(args.runs):
        model = build_model(args.config).to(device)
        num_classes = model.num_classes
        run_result = training_step(
            model=model,
            device=device,
            img_size=(args.img[0], args.img[1]),
            dtype=dtype,
            num_classes=num_classes,
        )
        run_result["run"] = run_idx + 1
        run_result["oom"] = run_result["oom"] or (
            run_result["max_memory_allocated_bytes"] > args.vram_limit_mb * 1024 * 1024
        )
        results["runs"].append(run_result)
        del model

    summary_model = build_model(args.config).to(device)
    allocated_values = [run["max_memory_allocated_bytes"] / (1024 ** 2) for run in results["runs"]]
    reserved_values = [run["max_memory_reserved_bytes"] / (1024 ** 2) for run in results["runs"]]
    step_times = [run["step_time_sec"] for run in results["runs"] if run["step_time_sec"] is not None]
    params = count_params(summary_model)

    results["summary"] = {
        "params": params,
        "flops": maybe_count_flops(summary_model, (args.img[0], args.img[1])),
        "median_step_time_sec": median(step_times) if step_times else None,
        "median_max_memory_allocated_mb": median(allocated_values),
        "median_max_memory_reserved_mb": median(reserved_values),
        "passed_vram_gate": all(v <= args.vram_limit_mb for v in allocated_values)
        and not any(run["oom"] for run in results["runs"]),
    }

    (output_dir / "verification.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0 if results["summary"]["passed_vram_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
