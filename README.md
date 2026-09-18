# MoGe-WebGPU

Single-image depth and surface normal estimation running entirely in the browser via WebGPU compute shaders.

**[Live demo →](https://lyonsno.github.io/moge-webgpu/)** — drop in an image and get depth, normals, and a 3D pointcloud. Heads up: the first load streams ~660MB of model weights (cached by your browser afterward).

A complete port of [MoGe-2](https://github.com/microsoft/MoGe) (ViT-Large + ConvStack decoder) from PyTorch to WebGPU. No server, no WASM, no ONNX runtime — pure GPU compute shaders dispatched from JavaScript, running on the [Kaminos WebGPU Inference Kit](https://github.com/lyonsno/kaminos/tree/main/webgpu-inference-kit) ([`@kaminos/webgpu-inference-kit`](https://www.npmjs.com/package/@kaminos/webgpu-inference-kit)) for cooperative scheduling, route receipts, and runtime telemetry.

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

## Cooperative scheduling

Long inference and a live page don't have to fight. Built on the
[Kaminos WebGPU Inference Kit](https://github.com/lyonsno/kaminos/tree/main/webgpu-inference-kit)'s
cooperative scheduling contract, the route can run in chunked GPU submissions
with browser yields between them, so the page keeps rendering while the model
works:

```js
await inference.run(imageData, {
  scheduler: { mode: 'cooperative', yieldMs: 4, vitBlockChunkSize: 1 },
});
```

- The DINOv2 backbone submits per transformer-block chunk; decoder and
  readback get their own submit/yield seams. Chunked submits are
  **bit-identical** to the monolithic run — same dispatch stream, persistent
  buffers.
- Cooperative submission keeps the page rendering during inference; it is
  not a sustained-frame-rate guarantee. Live-flame composition on M4 Max
  still shows perceptible periodic slowdown and remains under investigation.
- The run emits a kit-schema **scheduler verification receipt**: cooperative
  behavior is claimed only from genuinely observed submit/yield events, never
  from configuration or synthesized timing traces
  (`npm run test:cooperative-route`, plus pure-Node authority tests in
  `npm run test:scheduler-receipt-unit`).

This is what lets MoGe share one GPU (and one `GPUDevice`) with a live
renderer or simulation — the kit's core product target.

### Scheduler options

| Option | Default | Effect |
| --- | --- | --- |
| `mode` | — | `'cooperative'` enables chunked submits; omit for a single monolithic submit |
| `vitBlockChunkSize` | `1` | transformer blocks per backbone submit |
| `splitVitBlocks` | `false` | split each block into six attention/MLP segments, one submit each |
| `splitDecoderResBlocks` | `false` | split decoder ConvStack levels per residual conv, plus output-conv and resampler tails |
| `pacing` | `'strict-drain'` | `'bounded-prefix'` keeps up to `maxInFlightChunks` submits in flight and awaits only the oldest fence (GPU stays saturated; queued-ahead work stays bounded) |
| `maxInFlightChunks` | `2` | bounded-prefix depth |
| `yieldMs` | `4` | browser yield between chunks (`0` is a bare macrotask yield) |
| `admit` | — | optional async embedding-host callback after a submitted GPU chunk reaches the configured pacing boundary; receipt records callback identity and observed start/end events, never the function |

Finest granularity (`splitVitBlocks` + `splitDecoderResBlocks`) puts ~230
submissions in a run, the fattest being a single 3×3 conv at 296². Under
`strict-drain` that many waits stretch wall time badly; `bounded-prefix`
restores monolithic-class total time at the same granularity (measured 52.7s
→ 2.47s in the harness at identical chunking). In cooperative mode the CPU
preprocess and postprocess loops are also row-banded with yields.

Every chunk records submit, wait, yield and fence-retire events on a
`performance.now` clock in the scheduler receipt, labeled by chunk (e.g.
`block-7:attn-scores`, `neck:level-3:res-block-0:conv2`), so foreground hitches
can be attributed to the exact submission (`tools/probe_hitch_alignment.mjs`
aligns rAF frame gaps with submit→retire occupancy spans).

An embedding host may supply `admit({ phase, chunk, signal })` to require a
host-owned opportunity before the next GPU chunk is encoded. The callback is
allowed to fail the run when its external liveness predicate disappears. It
does not itself establish presentation or hardware-priority authority; the
host must retain and validate the observation that let the callback return.

## Embedding in a host application

`npm run build:lib` produces `dist-lib/moge-inference.js`: a single
self-contained ES module (WGSL inlined, no bundler needed on the host) that
runs inference on a device the host owns:

```js
import { MoGeInference, initGPU, borrowedDeviceBackendIdentity } from './moge-inference.js';

// Either let MoGe acquire a device (kit shared helper) …
const gpu = await initGPU();
// … or hand it one you already share with your renderer:
// const gpu = { device, adapter, backendIdentity: borrowedDeviceBackendIdentity({ adapter, device }) };

const inference = new MoGeInference(gpu);
await inference.init(progress => {});  // weights stream from HuggingFace if no local copy
await inference.warmUp();               // optional: one discarded cooperative run so the
                                        // first visible run is steady state (pipelines
                                        // dispatched, bind groups built, buffer pool filled)
const result = await inference.run(imageData, {
  scheduler: { mode: 'cooperative', splitVitBlocks: true, splitDecoderResBlocks: true, pacing: 'bounded-prefix' },
});
// When the host retires this instance (not after each inference):
await inference.dispose(); // releases MoGe resources, never destroys gpu.device
```

Device acquisition and backend identity come from the kit's shared
gpu-environment helpers (`requestBrowserWebGpuDevice`,
`createWebGpuBackendIdentity`; kit `^0.1.49`). Repeated `init()` calls share
one initialization. `run()` waits for an initialization already in progress;
call `init()` first. Each instance allows one run at a time (including warm-up
and debug comparisons); overlapping calls reject with an explicit busy error.
Separate instances may borrow the same device, with independent pools and
caches. Pooled intermediates are reused across runs; uploads and readbacks
remain transient allocations. Success and failure both drain submitted work
before recycling buffers. `dispose()` rejects new work, waits for accepted
initialization/inference, and releases this instance's resources. The host
retains ownership of the device and any renderer or other inference instance.
The first consumer is Kaminos' live-flame composition pages, which run MoGe
beside the pyro volume simulation.

## Quick start

```bash
git clone https://github.com/lyonsno/moge-webgpu.git
cd moge-webgpu
npm install
npm run dev
```

That's it — the app streams pre-converted fp16 weights (~660MB, cached by the browser) from [lyonsno/moge-webgpu on HuggingFace](https://huggingface.co/lyonsno/moge-webgpu) on first load. No Python required.

### Optional: convert weights locally

If you'd rather serve the weights yourself (or convert a different checkpoint), use a Python environment with PyTorch and huggingface_hub:

```bash
python tools/convert_weights.py \
  --model Ruicheng/moge-2-vitl-normal \
  --output public/weights.bin \
  --dtype fp16
```

This downloads the model from HuggingFace (~1.3GB PyTorch checkpoint) and converts it to a flat fp16 binary (~660MB) optimized for WebGPU buffer loading. A local `public/weights.bin` takes precedence over the hosted copy.

## Browser requirements

- Chrome 113+ or Edge 113+ (WebGPU enabled)
- Firefox 141+ (WebGPU enabled via `dom.webgpu.enabled` in about:config)
- GPU with WebGPU support

## Numerical verification

End-to-end depth output is compared against the PyTorch reference implementation on the same checkpoint (`tools/test_depth_parity.mjs`, references generated by `tools/dump_layer_outputs.py`):

- **End-to-end depth: 0.25% mean relative error, 0.06% scale-invariant relative RMS, metric scale within 0.24%** vs PyTorch fp32 (full 24-block ViT-L + ConvStack decoder pipeline, 268k pixels compared).
- Per-layer tensor parity (`tools/compare_backbone.mjs`): patch embed matches to rms 4e-5; the final transformer block (block 23) matches to rms 0.003 against an activation std of 5.35 — ~0.06% relative. The residual is fp16 weight-storage rounding; all GPU compute is fp32.
- The comparison references must come from the exact shipped checkpoint (`Ruicheng/moge-2-vitl-normal`). An earlier reference set generated from the sibling checkpoint (`moge-2-vitl`) produced phantom "divergence" of ~2% structure and ~15% scale — entirely a checkpoint mismatch, reproducible in pure numpy with no GPU involved. `tools/dump_layer_outputs.py` regenerates references; `tools/probe_scale_head.mjs` isolates the metric-scale head if the depth scale ever drifts again.

## Tools

- `tools/convert_weights.py` — Convert HuggingFace PyTorch checkpoint to WebGPU binary format
- `tools/dump_layer_outputs.py` — Dump PyTorch reference tensors for validation (defaults to the shipped checkpoint)
- `tools/compare_backbone.mjs` — Puppeteer-based automated backbone comparison harness
- `tools/test_depth_parity.mjs` — End-to-end depth parity vs PyTorch (`npm run test:depth-parity`)
- `tools/test_cooperative_route.mjs` — Cooperative run under a rAF monitor: verified receipt, observed yields, bit-identical depth vs monolithic (`npm run test:cooperative-route`)
- `tools/probe_hitch_alignment.mjs` — Attributes foreground frame gaps to the scheduler occupancy span they overlap
- `tools/test_scheduler_receipt_unit.mjs`, `tools/test_kit_gpu_environment_unit.mjs`, `tools/test_buffer_pool_unit.mjs` — pure-Node contract tests (no GPU)
- `tools/visual_smoke.mjs`, `tools/smoke_live_flame_page.mjs` — Automated visual smoke tests (the latter for the Kaminos composition page, with a fire-colour witness)

Measurement note: headless-Chrome harness frame timings are compositor-quantized
and only relative evidence; on-device HUD telemetry in the composition page is
the measurement of record for smoothness.

## License

MIT (matching upstream MoGe-2 license)
