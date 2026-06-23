# MoGe-WebGPU

Single-image depth and surface normal estimation running entirely in the browser via WebGPU compute shaders.

A complete port of [MoGe-2](https://github.com/microsoft/MoGe) (ViT-Large + ConvStack decoder) from PyTorch to WebGPU. No server, no WASM, no ONNX runtime — pure GPU compute shaders dispatched from JavaScript.

## What it does

Drop an image in the browser and get:
- **Depth map** — per-pixel depth estimation
- **Surface normals** — per-pixel 3D surface orientation (from dedicated normal head)
- **3D pointcloud** — interactive colored point cloud with orbit controls

All inference runs client-side on your GPU. ~2.5s on Apple M4 Max, ~660MB weight download on first load.

## Architecture

MoGe-2-ViT-Large-Normal (`Ruicheng/moge-2-vitl-normal`):

- **Encoder**: DINOv2 ViT-Large backbone (24 transformer blocks, 1024-dim)
  - Patch embedding (14x14 patches) + CLS token + position embeddings
  - 4 intermediate layer feature extraction (layers 5, 11, 17, 23)
  - Per-layer 1x1 conv projection + sum
- **Neck**: ConvStack (5-level multi-scale residual conv blocks with resamplers)
- **Points head**: ConvStack -> per-pixel xyz point map
- **Normal head**: ConvStack -> per-pixel surface normals
- **Mask head**: ConvStack -> per-pixel confidence mask
- **Scale head**: MLP (CLS token -> metric scale)

15 compute shaders: patch embedding, layer norm, multi-head self-attention (QKV projection, score computation, softmax, apply), linear projection, GELU MLP, layer scale, conv2d (replicate padding), conv1x1, conv_transpose2d, bilinear upsample, pixel shuffle, group norm, activations (ReLU/add/sigmoid).

## Setup

```bash
git clone https://github.com/lyonsno/moge-webgpu.git
cd moge-webgpu
npm install
```

### Download and convert weights

Requires a Python environment with PyTorch and huggingface_hub:

```bash
python tools/convert_weights.py \
  --model Ruicheng/moge-2-vitl-normal \
  --output public/weights.bin \
  --dtype fp16
```

This downloads the model from HuggingFace (~1.3GB PyTorch checkpoint) and converts it to a flat fp16 binary (~660MB) optimized for WebGPU buffer loading.

### Run

```bash
npx vite --port 5180
# Open http://localhost:5180/
```

## Browser requirements

- Chrome 113+ or Edge 113+ (WebGPU enabled)
- Firefox 141+ (WebGPU enabled via `dom.webgpu.enabled` in about:config)
- GPU with WebGPU support

## Tools

- `tools/convert_weights.py` — Convert HuggingFace PyTorch checkpoint to WebGPU binary format
- `tools/dump_layer_outputs.py` — Dump PyTorch reference tensors for validation
- `tools/compare_backbone.mjs` — Puppeteer-based automated backbone comparison harness
- `tools/visual_smoke.mjs` — Automated visual smoke test

## License

MIT (matching upstream MoGe-2 license)
