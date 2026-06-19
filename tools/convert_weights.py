#!/usr/bin/env python3
"""
Convert MoGe-2 PyTorch checkpoint to flat binary format for WebGPU.

Usage:
    python tools/convert_weights.py [--model Ruicheng/moge-2-vitl] [--output weights.bin] [--dtype fp32]

Output format:
    Header:
        4 bytes: magic "MOGE"
        4 bytes: version (1)
        4 bytes: num_tensors
        4 bytes: header_size (total bytes before weight data starts)

    Tensor table (repeated num_tensors times):
        64 bytes: name (null-padded ASCII)
        4 bytes: dtype (0=fp32, 1=fp16)
        4 bytes: ndim
        4 * ndim bytes: shape (padded to 16 bytes max = 4 dims)
        4 bytes: offset into weight data section
        4 bytes: size in bytes

    Weight data:
        Packed contiguous tensors in row-major order
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import torch
import numpy as np


MAGIC = b"MOGE"
VERSION = 1
MAX_NAME_LEN = 64
MAX_DIMS = 4


def load_checkpoint(model_name_or_path: str) -> dict:
    """Load MoGe-2 checkpoint from HuggingFace or local path."""
    path = Path(model_name_or_path)
    if path.exists():
        checkpoint_path = path
    else:
        from huggingface_hub import hf_hub_download
        checkpoint_path = hf_hub_download(
            repo_id=model_name_or_path,
            repo_type="model",
            filename="model.pt",
        )
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)


def convert(checkpoint: dict, output_path: str, dtype: str = "fp32"):
    """Convert checkpoint to flat binary."""
    model_config = checkpoint["model_config"]
    state_dict = checkpoint["model"]

    # Write model config as JSON sidecar
    config_path = Path(output_path).with_suffix(".json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2, default=str)
    print(f"Config written to {config_path}")

    # Prepare tensors
    tensor_entries = []
    weight_data = bytearray()

    dtype_code = 0 if dtype == "fp32" else 1
    np_dtype = np.float32 if dtype == "fp32" else np.float16

    for name, tensor in sorted(state_dict.items()):
        arr = tensor.detach().float().numpy().astype(np_dtype)
        data = arr.tobytes()

        shape = list(arr.shape)
        if len(shape) > MAX_DIMS:
            # Flatten extra dims
            shape = [int(np.prod(shape[:-3]))] + list(shape[-3:])
            arr = arr.reshape(shape)
            data = arr.tobytes()

        offset = len(weight_data)
        size = len(data)
        weight_data.extend(data)

        # Pad to 16-byte alignment
        pad = (16 - (len(weight_data) % 16)) % 16
        weight_data.extend(b"\x00" * pad)

        tensor_entries.append({
            "name": name,
            "dtype": dtype_code,
            "shape": shape,
            "offset": offset,
            "size": size,
        })

    # Build header
    num_tensors = len(tensor_entries)

    # Tensor table entry size: 64 (name) + 4 (dtype) + 4 (ndim) + 16 (shape) + 4 (offset) + 4 (size) = 96
    ENTRY_SIZE = 96
    header_size = 16 + num_tensors * ENTRY_SIZE  # 16 = magic + version + count + header_size

    # Adjust offsets to be relative to start of file
    for entry in tensor_entries:
        entry["offset"] += header_size

    # Write binary
    with open(output_path, "wb") as f:
        # Header
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", num_tensors))
        f.write(struct.pack("<I", header_size))

        # Tensor table
        for entry in tensor_entries:
            # Name (64 bytes, null-padded)
            name_bytes = entry["name"].encode("ascii")[:MAX_NAME_LEN]
            f.write(name_bytes.ljust(MAX_NAME_LEN, b"\x00"))

            # dtype
            f.write(struct.pack("<I", entry["dtype"]))

            # ndim
            ndim = len(entry["shape"])
            f.write(struct.pack("<I", ndim))

            # shape (padded to 4 dims)
            shape_padded = entry["shape"] + [0] * (MAX_DIMS - ndim)
            for s in shape_padded:
                f.write(struct.pack("<I", s))

            # offset and size
            f.write(struct.pack("<I", entry["offset"]))
            f.write(struct.pack("<I", entry["size"]))

        # Weight data
        f.write(weight_data)

    total_mb = len(weight_data) / (1024 * 1024)
    print(f"Weights written to {output_path}")
    print(f"  Tensors: {num_tensors}")
    print(f"  Size: {total_mb:.1f} MB ({dtype})")
    print(f"  Header: {header_size} bytes")

    # Print tensor summary
    print(f"\nTensor summary:")
    for entry in tensor_entries:
        shape_str = "x".join(str(s) for s in entry["shape"])
        print(f"  {entry['name']:60s} {shape_str:>20s}  {entry['size'] / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Convert MoGe-2 weights for WebGPU")
    parser.add_argument("--model", default="Ruicheng/moge-2-vitl",
                        help="HuggingFace model name or local checkpoint path")
    parser.add_argument("--output", default="weights.bin",
                        help="Output binary file path")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"],
                        help="Weight data type")
    parser.add_argument("--list-only", action="store_true",
                        help="Only list tensor names and shapes, don't write")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.model}")
    checkpoint = load_checkpoint(args.model)

    if args.list_only:
        state_dict = checkpoint["model"]
        print(f"\nModel config:")
        print(json.dumps(checkpoint["model_config"], indent=2, default=str))
        print(f"\nState dict ({len(state_dict)} tensors):")
        total_params = 0
        for name, tensor in sorted(state_dict.items()):
            shape = list(tensor.shape)
            params = tensor.numel()
            total_params += params
            shape_str = "x".join(str(s) for s in shape)
            print(f"  {name:60s} {shape_str:>20s}  {params:>10d}")
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Total size (fp32): {total_params * 4 / 1024 / 1024:.1f} MB")
        print(f"Total size (fp16): {total_params * 2 / 1024 / 1024:.1f} MB")
        return

    convert(checkpoint, args.output, args.dtype)


if __name__ == "__main__":
    main()
