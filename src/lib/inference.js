/**
 * inference.js — MoGe-2 inference pipeline in WebGPU compute.
 *
 * Architecture (from upstream MoGe-2 v2.py + modules.py):
 *
 *   1. DINOv2 ViT-Large encoder → feature map + CLS token
 *      [STUBBED: random activations until ViT kernels arrive from voxel-attention-defibrillator]
 *
 *   2. Feature map + UV coords at 5 scales → neck ConvStack → multi-scale features
 *
 *   3. points_head ConvStack → xyz point map (H×W×3)
 *   4. normal_head ConvStack → normal vectors (H×W×3)
 *   5. mask_head ConvStack → confidence mask (H×W×1)
 *   6. scale_head MLP → metric scale from CLS token
 *
 *   7. Post-processing: remap, focal recovery, mask application
 */

import { createStorageBuffer, createEmptyBuffer, readBuffer } from './gpu.js';
import {
  dispatchConv2d,
  dispatchConv1x1,
  dispatchActivation,
  dispatchGroupNorm,
  dispatchPixelShuffle,
  dispatchUpsample,
} from './shader_ops.js';


/**
 * ResidualConvBlock dispatch:
 *   GroupNorm → ReLU → Conv3x3 → GroupNorm → ReLU → Conv3x3 + Skip
 */
function dispatchResidualConvBlock(device, encoder, inputBuf, weights, params) {
  const { C, hiddenC, H, W, normType } = params;
  const numGroups = normType === 'group_norm' ? Math.floor(C / 32) : 1;
  const hiddenGroups = normType === 'group_norm' ? Math.floor(hiddenC / 32) : 1;

  // GroupNorm 1
  let x = dispatchGroupNorm(device, encoder, inputBuf, weights.norm1_scale, weights.norm1_bias,
    { C, H, W, numGroups });

  // ReLU
  x = dispatchActivation(device, encoder, x, null, C * H * W, 0);

  // Conv3x3 (C → hiddenC)
  let convOut = dispatchConv2d(device, encoder, x, weights.conv1_weight, weights.conv1_bias,
    { inC: C, inH: H, inW: W, outC: hiddenC, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });

  // GroupNorm 2
  x = dispatchGroupNorm(device, encoder, convOut.buffer, weights.norm2_scale, weights.norm2_bias,
    { C: hiddenC, H, W, numGroups: hiddenGroups });

  // ReLU
  x = dispatchActivation(device, encoder, x, null, hiddenC * H * W, 0);

  // Conv3x3 (hiddenC → C)
  convOut = dispatchConv2d(device, encoder, x, weights.conv2_weight, weights.conv2_bias,
    { inC: hiddenC, inH: H, inW: W, outC: C, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });

  // Skip connection: if in_channels != out_channels, apply 1x1 conv
  let skip;
  if (weights.skip_weight) {
    skip = dispatchConv1x1(device, encoder, inputBuf, weights.skip_weight, null,
      { inC: C, outC: C, H, W }).buffer;
  } else {
    skip = inputBuf;
  }

  // Add skip connection
  const out = dispatchActivation(device, encoder, convOut.buffer, skip, C * H * W, 2);
  return out;
}

/**
 * Resampler dispatch: Conv3x3 → PixelShuffle → Conv3x3
 * (for 'pixel_shuffle' type, which is MoGe-2's default)
 */
function dispatchResampler(device, encoder, inputBuf, weights, params) {
  const { inC, outC, H, W, scaleFactor, type } = params;

  if (type === 'pixel_shuffle') {
    // Conv3x3: inC → outC * scaleFactor^2
    const expandedC = outC * scaleFactor * scaleFactor;
    const conv1 = dispatchConv2d(device, encoder, inputBuf, weights.conv1_weight, weights.conv1_bias,
      { inC, inH: H, inW: W, outC: expandedC, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });

    // PixelShuffle
    const shuffled = dispatchPixelShuffle(device, encoder, conv1.buffer,
      { inC: expandedC, inH: H, inW: W, scaleFactor });

    // Conv3x3: outC → outC
    const conv2 = dispatchConv2d(device, encoder, shuffled.buffer, weights.conv2_weight, weights.conv2_bias,
      { inC: outC, inH: shuffled.H, inW: shuffled.W, outC, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });

    return { buffer: conv2.buffer, H: shuffled.H, W: shuffled.W };
  } else if (type === 'bilinear' || type === 'nearest') {
    // Upsample
    const mode = type === 'bilinear' ? 1 : 0;
    const upsampled = dispatchUpsample(device, encoder, inputBuf,
      { C: inC, inH: H, inW: W, outH: H * scaleFactor, outW: W * scaleFactor, mode });

    // Conv3x3: inC → outC
    const conv = dispatchConv2d(device, encoder, upsampled.buffer, weights.conv1_weight, weights.conv1_bias,
      { inC, inH: upsampled.H, inW: upsampled.W, outC, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });

    return { buffer: conv.buffer, H: upsampled.H, W: upsampled.W };
  }

  throw new Error(`Unsupported resampler type: ${type}`);
}

/**
 * ConvStack dispatch:
 *   For each level:
 *     1. input_block (1x1 conv to project features)
 *     2. Add to running feature (skip from previous level)
 *     3. res_blocks (N × ResidualConvBlock)
 *     4. output_block (1x1 conv to project to output dim)
 *     5. resampler (between levels)
 */
function dispatchConvStack(device, encoder, inFeatures, weights, config) {
  const { levels, resBlockCount } = config;
  const outFeatures = [];
  let x = null;

  for (let i = 0; i < levels.length; i++) {
    const level = levels[i];
    const { dimIn, dimResBlock, dimOut, H, W } = level;

    // input_block: 1x1 conv (project input to dimResBlock)
    let projected;
    if (dimIn != null && inFeatures[i] != null) {
      projected = dispatchConv1x1(device, encoder, inFeatures[i],
        weights.levels[i].input_weight, weights.levels[i].input_bias,
        { inC: dimIn, outC: dimResBlock, H, W });
    }

    // Add to running state
    if (i === 0) {
      x = projected.buffer;
    } else if (projected) {
      x = dispatchActivation(device, encoder, x, projected.buffer, dimResBlock * H * W, 2);
    }

    // res_blocks
    const numBlocks = Array.isArray(resBlockCount) ? resBlockCount[i] : resBlockCount;
    for (let j = 0; j < numBlocks; j++) {
      x = dispatchResidualConvBlock(device, encoder, x, weights.levels[i].res_blocks[j], {
        C: dimResBlock, hiddenC: dimResBlock, H, W, normType: 'group_norm',
      });
    }

    // output_block: 1x1 conv
    if (dimOut != null) {
      const out = dispatchConv1x1(device, encoder, x,
        weights.levels[i].output_weight, weights.levels[i].output_bias,
        { inC: dimResBlock, outC: dimOut, H, W });
      outFeatures.push({ buffer: out.buffer, C: dimOut, H, W });
    } else {
      outFeatures.push({ buffer: x, C: dimResBlock, H, W });
    }

    // resampler (between levels, not after last)
    if (i < levels.length - 1) {
      const resampled = dispatchResampler(device, encoder, x,
        weights.levels[i].resampler, {
          inC: dimResBlock, outC: levels[i + 1].dimResBlock,
          H, W, scaleFactor: 2, type: level.resamplerType || 'pixel_shuffle',
        });
      x = resampled.buffer;
    }
  }

  return outFeatures;
}


export class MoGeInference {
  constructor(gpu) {
    this.device = gpu.device;
    this.weights = null;
    this.modelConfig = null;
  }

  async init() {
    // For now: stub weights (random). Real weight loading comes later.
    this.modelConfig = this._getDefaultConfig();
    this.weights = this._createStubWeights();
  }

  _getDefaultConfig() {
    // MoGe-2 ViT-Large config from upstream
    // Encoder output dim after projection: typically 512 or 1024
    // We'll use the actual config once weights are loaded
    return {
      encoderDim: 1024,
      neckConfig: {
        // 5 levels, progressively higher resolution
        levels: [
          { dimIn: 1024 + 2, dimResBlock: 512, dimOut: null, resamplerType: 'pixel_shuffle' },
          { dimIn: 2, dimResBlock: 256, dimOut: null, resamplerType: 'pixel_shuffle' },
          { dimIn: 2, dimResBlock: 128, dimOut: null, resamplerType: 'pixel_shuffle' },
          { dimIn: 2, dimResBlock: 64, dimOut: null, resamplerType: 'pixel_shuffle' },
          { dimIn: 2, dimResBlock: 32, dimOut: null, resamplerType: 'pixel_shuffle' },
        ],
        resBlockCount: 1,
      },
      pointsHeadConfig: {
        levels: [
          { dimIn: 512, dimResBlock: 256, dimOut: null, resamplerType: 'pixel_shuffle' },
          { dimIn: 256, dimResBlock: 128, dimOut: null, resamplerType: 'pixel_shuffle' },
          { dimIn: 128, dimResBlock: 64, dimOut: null, resamplerType: 'pixel_shuffle' },
          { dimIn: 64, dimResBlock: 32, dimOut: 3, resamplerType: 'pixel_shuffle' },
        ],
        resBlockCount: 1,
      },
      inputSize: 224,
      patchSize: 14,
    };
  }

  _createStubWeights() {
    // Create random weight buffers matching the model architecture.
    // This lets us validate the dispatch chain end-to-end before real weights.
    const d = this.device;
    const rand = (n) => {
      const data = new Float32Array(n);
      for (let i = 0; i < n; i++) data[i] = (Math.random() - 0.5) * 0.1;
      return createStorageBuffer(d, data);
    };
    const zeros = (n) => createStorageBuffer(d, new Float32Array(n));

    // Helper to create ResidualConvBlock weights
    const makeResBlock = (C, hiddenC) => ({
      norm1_scale: rand(C),
      norm1_bias: zeros(C),
      conv1_weight: rand(hiddenC * C * 3 * 3),
      conv1_bias: zeros(hiddenC),
      norm2_scale: rand(hiddenC),
      norm2_bias: zeros(hiddenC),
      conv2_weight: rand(C * hiddenC * 3 * 3),
      conv2_bias: zeros(C),
      skip_weight: null,
    });

    // Helper to create Resampler weights (pixel_shuffle type)
    const makeResampler = (inC, outC, r = 2) => ({
      conv1_weight: rand(outC * r * r * inC * 3 * 3),
      conv1_bias: zeros(outC * r * r),
      conv2_weight: rand(outC * outC * 3 * 3),
      conv2_bias: zeros(outC),
    });

    // Stub: just create enough structure for the neck
    const cfg = this.modelConfig;
    const neckWeights = {
      levels: cfg.neckConfig.levels.map((level, i) => {
        const prevDim = i === 0 ? level.dimResBlock : cfg.neckConfig.levels[i - 1].dimResBlock;
        return {
          input_weight: rand(level.dimResBlock * level.dimIn),
          input_bias: zeros(level.dimResBlock),
          res_blocks: [makeResBlock(level.dimResBlock, level.dimResBlock)],
          output_weight: level.dimOut ? rand(level.dimOut * level.dimResBlock) : null,
          output_bias: level.dimOut ? zeros(level.dimOut) : null,
          resampler: i < cfg.neckConfig.levels.length - 1
            ? makeResampler(level.dimResBlock, cfg.neckConfig.levels[i + 1].dimResBlock)
            : null,
        };
      }),
    };

    return { neck: neckWeights };
  }

  /**
   * Run inference on an image.
   * Returns { depth, normals, points, colors } as Float32Arrays.
   */
  async run(imageData) {
    const { width, height } = imageData;
    const device = this.device;

    // Convert RGBA image to RGB float [0, 1] in CHW format
    const rgbData = new Float32Array(3 * width * height);
    for (let i = 0; i < width * height; i++) {
      rgbData[0 * width * height + i] = imageData.data[i * 4 + 0] / 255;
      rgbData[1 * width * height + i] = imageData.data[i * 4 + 1] / 255;
      rgbData[2 * width * height + i] = imageData.data[i * 4 + 2] / 255;
    }

    // --- STUB BACKBONE ---
    // Until ViT kernels arrive, generate random features at the right shape.
    // DINOv2 ViT-Large with patch_size=14, input 224x224 → 16x16 tokens → feature map [1024, 16, 16]
    const tokenH = Math.floor(height / 14);
    const tokenW = Math.floor(width / 14);
    const encoderDim = this.modelConfig.encoderDim;

    const stubFeatures = new Float32Array(encoderDim * tokenH * tokenW);
    for (let i = 0; i < stubFeatures.length; i++) {
      stubFeatures[i] = (Math.random() - 0.5) * 0.5;
    }
    const featureBuf = createStorageBuffer(device, stubFeatures);

    // Generate UV coordinates at each scale
    // Level 0: tokenH × tokenW, Level 1: 2*tokenH × 2*tokenW, etc.
    const uvFeatures = [];
    for (let level = 0; level < 5; level++) {
      const h = tokenH * (2 ** level);
      const w = tokenW * (2 ** level);
      const uv = new Float32Array(2 * h * w);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          uv[0 * h * w + y * w + x] = (x + 0.5) / w * 2 - 1; // u: [-1, 1]
          uv[1 * h * w + y * w + x] = (y + 0.5) / h * 2 - 1; // v: [-1, 1]
        }
      }
      uvFeatures.push(createStorageBuffer(device, uv));
    }

    // --- STUB OUTPUT ---
    // For now, generate plausible-looking depth from the input image itself
    // (grayscale as rough depth proxy) so we can test the visualization pipeline.
    const depth = new Float32Array(width * height);
    const normals = new Float32Array(3 * width * height);
    const points = new Float32Array(3 * width * height);
    const colors = new Float32Array(3 * width * height);

    // Simple stub: depth from luminance
    for (let i = 0; i < width * height; i++) {
      const r = imageData.data[i * 4 + 0] / 255;
      const g = imageData.data[i * 4 + 1] / 255;
      const b = imageData.data[i * 4 + 2] / 255;

      depth[i] = 1.0 + (0.299 * r + 0.587 * g + 0.114 * b) * 4.0;

      const y = Math.floor(i / width);
      const x = i % width;

      // Fake normals from depth gradient
      const idx = (cx, cy) => Math.max(0, Math.min(height - 1, cy)) * width + Math.max(0, Math.min(width - 1, cx));
      const dzdx = (depth[idx(x + 1, y)] || depth[i]) - (depth[idx(x - 1, y)] || depth[i]);
      const dzdy = (depth[idx(x, y + 1)] || depth[i]) - (depth[idx(x, y - 1)] || depth[i]);
      const len = Math.sqrt(dzdx * dzdx + dzdy * dzdy + 1);
      normals[i * 3 + 0] = -dzdx / len;
      normals[i * 3 + 1] = -dzdy / len;
      normals[i * 3 + 2] = 1.0 / len;

      // 3D points from depth + pixel coords (simple pinhole)
      const fx = width * 0.8;
      const fy = height * 0.8;
      points[i * 3 + 0] = (x - width / 2) / fx * depth[i];
      points[i * 3 + 1] = (y - height / 2) / fy * depth[i];
      points[i * 3 + 2] = depth[i];

      colors[i * 3 + 0] = r;
      colors[i * 3 + 1] = g;
      colors[i * 3 + 2] = b;
    }

    // Clean up GPU buffers
    featureBuf.destroy();
    uvFeatures.forEach(b => b.destroy());

    return { depth, normals, points, colors, width, height };
  }
}
