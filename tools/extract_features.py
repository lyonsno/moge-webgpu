#!/usr/bin/env python3
"""
Extract real encoder features from PyTorch MoGe-2 for decoder validation.

Runs the encoder on a test image and saves:
  1. The encoder output feature map [1024, tokenH, tokenW]
  2. The CLS token [1024]
  3. The full MoGe-2 output (points, normals, mask, depth) as reference
  4. The resized input image as RGB float [3, H, W]

These are loaded as test fixtures in the browser to validate the WebGPU
decoder independently of the backbone.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output-dir", default="public/test_fixtures", help="Output directory")
    parser.add_argument("--model", default="Ruicheng/moge-2-vitl", help="HF model name")
    parser.add_argument("--resize", type=int, default=224, help="Resize input to this size")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    from moge.model import import_model_class_by_version
    MoGeModel = import_model_class_by_version("v2")
    model = MoGeModel.from_pretrained(args.model).to("mps").eval()

    # Load and preprocess image
    print(f"Loading image: {args.image}")
    bgr = cv2.imread(args.image)
    if bgr is None:
        print(f"Error: could not read {args.image}")
        sys.exit(1)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Resize
    h, w = rgb.shape[:2]
    size = args.resize
    rgb_resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)

    # To tensor [1, 3, H, W] float32 [0, 1]
    image_tensor = torch.tensor(rgb_resized / 255.0, dtype=torch.float32, device="mps").permute(2, 0, 1).unsqueeze(0)

    # Save input image as binary
    input_chw = image_tensor[0].cpu().numpy().astype(np.float32)  # [3, H, W]
    input_chw.tofile(str(output_dir / "input_image.bin"))
    print(f"Saved input image: {input_chw.shape} → input_image.bin")

    # Also save the resized image as a viewable png
    cv2.imwrite(str(output_dir / "input.png"), cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR))

    # Run encoder to get features
    print("Running encoder...")
    aspect_ratio = 1.0  # square input
    num_tokens = 2400  # default-ish
    base_h = round((num_tokens / aspect_ratio) ** 0.5)
    base_w = round((num_tokens * aspect_ratio) ** 0.5)

    with torch.inference_mode():
        features, cls_token = model.encoder(image_tensor, base_h, base_w, return_class_token=True)

    print(f"Encoder output: features {features.shape}, cls_token {cls_token.shape}")
    # features: [1, 1024, tokenH, tokenW]
    # cls_token: [1, 1024]

    tokenH = base_h
    tokenW = base_w

    # Save encoder features
    feat_np = features[0].cpu().numpy().astype(np.float32)  # [1024, tokenH, tokenW]
    feat_np.tofile(str(output_dir / "encoder_features.bin"))
    print(f"Saved encoder features: {feat_np.shape} → encoder_features.bin")

    cls_np = cls_token[0].cpu().numpy().astype(np.float32)  # [1024]
    cls_np.tofile(str(output_dir / "cls_token.bin"))
    print(f"Saved CLS token: {cls_np.shape} → cls_token.bin")

    # Now run full forward pass to get reference output
    print("Running full forward pass...")
    with torch.inference_mode():
        output = model.infer(image_tensor, num_tokens=num_tokens, use_fp16=False)

    # Save reference outputs
    for key in ["points", "depth", "mask", "normal"]:
        if key in output:
            val = output[key]
            if val is not None:
                arr = val[0].cpu().numpy().astype(np.float32) if val.dim() == 4 else val[0].cpu().numpy().astype(np.float32)
                arr.tofile(str(output_dir / f"ref_{key}.bin"))
                print(f"Saved ref_{key}: {arr.shape} → ref_{key}.bin")

    if "intrinsics" in output:
        intr = output["intrinsics"][0].cpu().numpy().astype(np.float32)
        intr.tofile(str(output_dir / "ref_intrinsics.bin"))
        print(f"Saved ref_intrinsics: {intr.shape}")

    # Save metadata
    meta = {
        "image": args.image,
        "resize": size,
        "tokenH": tokenH,
        "tokenW": tokenW,
        "num_tokens": num_tokens,
        "encoder_dim": 1024,
        "features_shape": list(feat_np.shape),
        "model": args.model,
    }
    for key in ["points", "depth", "mask", "normal"]:
        if key in output and output[key] is not None:
            arr = output[key][0].cpu().numpy()
            meta[f"ref_{key}_shape"] = list(arr.shape)
            meta[f"ref_{key}_range"] = [float(np.nanmin(arr[np.isfinite(arr)])) if np.any(np.isfinite(arr)) else 0,
                                         float(np.nanmax(arr[np.isfinite(arr)])) if np.any(np.isfinite(arr)) else 0]

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata: {json.dumps(meta, indent=2)}")

    print("\nDone! Use these fixtures to validate the WebGPU decoder.")


if __name__ == "__main__":
    main()
