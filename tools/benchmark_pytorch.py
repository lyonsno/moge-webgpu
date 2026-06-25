#!/usr/bin/env python3
"""
PyTorch MPS/CPU benchmark for MoGe-2 inference.

Measures: cold load, warm init, first/warm inference latency, memory.
Uses the same test fixture as the WebGPU benchmark.

Usage:
    python tools/benchmark_pytorch.py [--runs 10] [--device mps] [--json]
"""

import argparse
import json
import time
import sys
import os

import torch
import numpy as np
from PIL import Image


def get_model_and_input(device_str, fixture_path):
    """Load MoGe-2 model and prepare input from the test fixture."""
    # Import moge
    moge_dir = os.path.expanduser("~/dev/moge-standalone")
    sys.path.insert(0, moge_dir)

    from moge.model.v2 import MoGeModel

    t_load_start = time.perf_counter()
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
    t_load_model = time.perf_counter() - t_load_start

    device = torch.device(device_str)
    t_to_device = time.perf_counter()
    model = model.to(device).eval()
    t_to_device = time.perf_counter() - t_to_device

    # Load test image
    img = Image.open(fixture_path).convert("RGB")
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    return model, img_tensor, t_load_model, t_to_device


def benchmark_inference(model, img_tensor, device_str, num_runs=10):
    """Run inference and collect timing."""
    device = torch.device(device_str)
    timings = []

    for i in range(num_runs + 1):  # +1 for first/warmup run
        if device_str == "mps":
            torch.mps.synchronize()

        t0 = time.perf_counter()

        with torch.no_grad():
            output = model.infer(img_tensor)

        if device_str == "mps":
            torch.mps.synchronize()
        elif device_str == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

        label = "first" if i == 0 else f"warm {i}"
        print(f"  {label}: {elapsed:.3f}s", file=sys.stderr)

    return timings[0], timings[1:]  # first, warm_runs


def stats(arr):
    s = sorted(arr)
    return {
        "min": s[0],
        "max": s[-1],
        "median": s[len(s) // 2],
        "mean": sum(s) / len(s),
        "samples": s,
    }


def get_memory_info(device_str):
    """Get memory usage info."""
    info = {}
    if device_str == "mps":
        try:
            info["mps_allocated_mb"] = torch.mps.current_allocated_memory() / 1024 / 1024
            info["mps_driver_allocated_mb"] = torch.mps.driver_allocated_memory() / 1024 / 1024
        except Exception:
            pass
    elif device_str == "cuda":
        info["cuda_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        info["cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024

    import resource
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    info["rss_mb"] = rusage.ru_maxrss / 1024 / 1024  # macOS reports in bytes

    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--image", default=None, help="Path to input image (default: test fixture)")
    args = parser.parse_args()

    fixture = args.image or os.path.join(
        os.path.dirname(__file__), "..", "public", "test_fixtures", "input.png"
    )
    fixture = os.path.abspath(fixture)

    if not args.json:
        print(f"PyTorch MoGe-2 benchmark — device={args.device}, runs={args.runs}", file=sys.stderr)
        print(f"Image: {fixture}", file=sys.stderr)

    # Load
    print("Loading model...", file=sys.stderr)
    model, img_tensor, t_load_model, t_to_device = get_model_and_input(args.device, fixture)

    if not args.json:
        print(f"Model load: {t_load_model:.3f}s", file=sys.stderr)
        print(f"To device ({args.device}): {t_to_device:.3f}s", file=sys.stderr)

    # Benchmark
    print(f"Running {args.runs + 1} inferences...", file=sys.stderr)
    first_time, warm_times = benchmark_inference(model, img_tensor, args.device, args.runs)

    # Memory
    mem = get_memory_info(args.device)

    results = {
        "runtime": f"PyTorch {args.device}",
        "precision": "fp32",
        "model": "moge-2-vitl-normal",
        "device": args.device,
        "torchVersion": torch.__version__,
        "runs": args.runs,
        "modelLoadMs": t_load_model * 1000,
        "toDeviceMs": t_to_device * 1000,
        "firstInferenceMs": first_time * 1000,
        "warmInferenceMs": [t * 1000 for t in warm_times],
        "warmStats": {k: v * 1000 if isinstance(v, float) else [x * 1000 for x in v] if isinstance(v, list) else v for k, v in stats(warm_times).items()},
        "memoryMB": mem,
        "imageSize": f"{img_tensor.shape[3]}x{img_tensor.shape[2]}",
    }

    if not args.json:
        print(f"\n--- Results ---", file=sys.stderr)
        print(f"Model load:       {t_load_model:.3f}s", file=sys.stderr)
        print(f"To device:        {t_to_device:.3f}s", file=sys.stderr)
        print(f"First inference:  {first_time:.3f}s", file=sys.stderr)
        ws = stats(warm_times)
        print(f"Warm inference:   median={ws['median']:.3f}s, min={ws['min']:.3f}s, max={ws['max']:.3f}s", file=sys.stderr)
        for k, v in mem.items():
            print(f"Memory ({k}): {v:.1f} MB", file=sys.stderr)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
