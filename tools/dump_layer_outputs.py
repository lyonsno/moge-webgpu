#!/usr/bin/env python3
"""
Dump every intermediate tensor from PyTorch MoGe-2 for layer-by-layer
comparison against WebGPU outputs.

Saves binary files for:
  - Normalized image (after ImageNet mean/std)
  - Patch embed output (before and after pos embed)
  - Token state after each transformer block (0, 5, 11, 17, 23)
  - Output projection results (4 intermediate layers)
  - Summed encoder features
  - Neck level 0 input (features + UV)
  - Neck level outputs (0-4)
  - Points head final output
  - Post-processed depth

All tensors saved as fp32 flat binary with a JSON manifest.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def save_tensor(output_dir, name, tensor, manifest):
    """Save a tensor as flat binary and record in manifest."""
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().float().numpy()
    else:
        arr = np.asarray(tensor, dtype=np.float32)

    path = output_dir / f"{name}.bin"
    arr.tofile(str(path))
    manifest[name] = {
        "shape": list(arr.shape),
        "dtype": "float32",
        "range": [float(arr.min()), float(arr.max())],
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "file": f"{name}.bin",
        "size_bytes": arr.nbytes,
    }
    print(f"  {name}: shape={list(arr.shape)}, range=[{arr.min():.4f}, {arr.max():.4f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="public/layer_dumps")
    parser.add_argument("--model", default="Ruicheng/moge-2-vitl")
    parser.add_argument("--token-size", type=int, default=37,
                        help="Token grid size (37 matches pretrained pos_embed)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}

    device = "mps"
    patch_size = 14
    img_size = args.token_size * patch_size
    token_h = token_w = args.token_size

    # Load model
    print(f"Loading model: {args.model}")
    from moge.model import import_model_class_by_version
    MoGeModel = import_model_class_by_version("v2")
    model = MoGeModel.from_pretrained(args.model).to(device).eval()

    # Load and preprocess image
    print(f"Loading image: {args.image} → {img_size}x{img_size}")
    bgr = cv2.imread(args.image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # To tensor [1, 3, H, W] float [0, 1]
    img_tensor = torch.tensor(rgb_resized / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    # Normalize with ImageNet mean/std (same as DINOv2 encoder)
    mean = model.encoder.image_mean.to(device)
    std = model.encoder.image_std.to(device)
    img_norm = (img_tensor - mean) / std

    # Resize to patch-aligned size with antialiasing (matching upstream)
    img_14 = F.interpolate(img_norm, (token_h * 14, token_w * 14),
                           mode="bilinear", align_corners=False, antialias=True)

    save_tensor(output_dir, "input_normalized", img_14[0], manifest)

    print("\n--- Patch Embedding ---")
    with torch.inference_mode():
        backbone = model.encoder.backbone

        # Patch embed
        patch_out = backbone.patch_embed(img_14)  # [1, N, D]
        save_tensor(output_dir, "patch_embed_output", patch_out[0], manifest)

        # CLS + pos embed
        cls_token = backbone.cls_token.expand(1, -1, -1)
        tokens = torch.cat([cls_token, patch_out], dim=1)  # [1, N+1, D]

        # Position embedding (with interpolation if needed)
        pos = backbone.interpolate_pos_encoding(tokens, token_h, token_w)
        tokens = tokens + pos
        save_tensor(output_dir, "tokens_after_pos_embed", tokens[0], manifest)

        print("\n--- Transformer Blocks ---")
        x = backbone.norm(tokens) if hasattr(backbone, 'norm') and False else tokens

        # Run blocks and capture intermediates
        for i, block in enumerate(backbone.blocks):
            # Detailed sub-block dumps for block 0
            if i == 0:
                # norm1
                norm1_out = block.norm1(x)
                save_tensor(output_dir, "block_0_norm1", norm1_out[0], manifest)
                # QKV
                qkv = block.attn.qkv(norm1_out)
                save_tensor(output_dir, "block_0_qkv", qkv[0], manifest)
                # Attention output (full attn forward on norm1 output)
                attn_out = block.attn(norm1_out)
                save_tensor(output_dir, "block_0_attn_out", attn_out[0], manifest)
                # After ls1 + residual
                ls1_out = x + block.ls1(attn_out)
                save_tensor(output_dir, "block_0_after_ls1", ls1_out[0], manifest)
                # norm2
                norm2_out = block.norm2(ls1_out)
                save_tensor(output_dir, "block_0_norm2", norm2_out[0], manifest)
                # MLP
                mlp_out = block.mlp(norm2_out)
                save_tensor(output_dir, "block_0_mlp_out", mlp_out[0], manifest)
                # fc1 intermediate (before GELU)
                fc1_pre = block.mlp.fc1(norm2_out)
                save_tensor(output_dir, "block_0_fc1_pre_gelu", fc1_pre[0], manifest)
                # fc1 after GELU
                fc1_post = block.mlp.act(fc1_pre)
                save_tensor(output_dir, "block_0_fc1_post_gelu", fc1_post[0], manifest)

            x = block(x)
            if i in [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 23]:
                save_tensor(output_dir, f"block_{i}_output", x[0], manifest)

        # Final norm
        x_normed = backbone.norm(x)
        save_tensor(output_dir, "backbone_final_norm", x_normed[0], manifest)

        print("\n--- Encoder Output Projections ---")
        # Re-run to get intermediate layers (get_intermediate_layers)
        features_list = backbone.get_intermediate_layers(
            img_14, n=model.encoder.intermediate_layers, return_class_token=True
        )

        for idx, (feat, cls) in enumerate(features_list):
            save_tensor(output_dir, f"intermediate_layer_{idx}_feat", feat[0], manifest)
            save_tensor(output_dir, f"intermediate_layer_{idx}_cls", cls, manifest)

        # Project and sum (matching encoder.forward)
        projected_sum = None
        for idx, (feat, cls) in enumerate(features_list):
            # feat: [1, N, D] → permute to [1, D, N] → unflatten to [1, D, H, W]
            feat_chw = feat.permute(0, 2, 1).unflatten(2, (token_h, token_w)).contiguous()
            proj = model.encoder.output_projections[idx](feat_chw)
            save_tensor(output_dir, f"output_proj_{idx}", proj[0], manifest)

            if projected_sum is None:
                projected_sum = proj
            else:
                projected_sum = projected_sum + proj

        save_tensor(output_dir, "encoder_features_sum", projected_sum[0], manifest)

        # Also get CLS token from last layer
        cls_token_final = features_list[-1][1]
        save_tensor(output_dir, "cls_token_final", cls_token_final, manifest)

        print("\n--- Neck ---")
        from moge.utils.geometry_torch import normalized_view_plane_uv

        # Build neck inputs (matching v2.py forward)
        encoder_features = projected_sum
        neck_inputs = [encoder_features, None, None, None, None]

        for level in range(5):
            uv = normalized_view_plane_uv(
                width=token_w * 2**level, height=token_h * 2**level,
                aspect_ratio=1.0, dtype=torch.float32, device=device
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0)
            if neck_inputs[level] is None:
                neck_inputs[level] = uv
            else:
                neck_inputs[level] = torch.cat([neck_inputs[level], uv], dim=1)

            save_tensor(output_dir, f"neck_input_{level}", neck_inputs[level][0], manifest)

        # Run neck
        neck_outputs = model.neck(neck_inputs)
        for i, out in enumerate(neck_outputs):
            save_tensor(output_dir, f"neck_output_{i}", out[0], manifest)

        print("\n--- Points Head ---")
        points_outputs = model.points_head(neck_outputs)
        for i, out in enumerate(points_outputs):
            save_tensor(output_dir, f"points_head_output_{i}", out[0], manifest)

        # Final points output
        raw_points = points_outputs[-1]  # [1, 3, H, W]
        save_tensor(output_dir, "points_raw_final", raw_points[0], manifest)

        # Resize to original image size
        raw_points_resized = F.interpolate(
            raw_points, (img_size, img_size),
            mode="bilinear", align_corners=False, antialias=False
        )

        # Remap (exp)
        points_hwc = raw_points_resized.permute(0, 2, 3, 1)  # [1, H, W, 3]
        xy, z = points_hwc[..., :2], points_hwc[..., 2:]
        z_exp = torch.exp(z)
        points_remapped = torch.cat([xy * z_exp, z_exp], dim=-1)
        save_tensor(output_dir, "points_remapped", points_remapped[0], manifest)

        depth = points_remapped[0, :, :, 2]
        save_tensor(output_dir, "depth_final", depth, manifest)

    # Save manifest
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! {len(manifest)} tensors saved to {output_dir}/")
    print(f"Manifest: {output_dir}/manifest.json")


if __name__ == "__main__":
    main()
