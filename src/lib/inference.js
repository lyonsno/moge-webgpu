/**
 * inference.js — MoGe-2 inference pipeline in WebGPU compute.
 *
 * Architecture (from upstream configs/train/v2.json + modules.py):
 *
 *   Encoder: DINOv2 ViT-Large (dinov2_vitl14)
 *     - intermediate_layers: [5, 11, 17, 23] → sum of projected features → [1024, tokenH, tokenW]
 *     - Also returns CLS token [1024]
 *     [STUBBED: random features until ViT kernels arrive from voxel-attention-defibrillator]
 *
 *   Neck ConvStack:
 *     dim_in: [1026, 2, 2, 2, 2]  (encoder features + 2 UV channels per level)
 *     dim_res_blocks: [1024, 256, 128, 64, 32]
 *     num_res_blocks: [0, 2, 2, 2, 0]
 *     resamplers: [conv_transpose, conv_transpose, conv_transpose, bilinear]
 *     norm: none
 *
 *   Points/Normal/Mask heads (each a ConvStack):
 *     dim_in: [1024, 256, 128, 64, 32]  (from neck outputs)
 *     dim_res_blocks: [1024, 256, 128, 64, 32]
 *     num_res_blocks: [0, 1, 1, 1, 0]
 *     dim_out: [null, null, null, null, 3/3/1]
 *     resamplers: [conv_transpose, conv_transpose, conv_transpose, bilinear]
 *
 *   Scale head: MLP [1024 → 1024 → 1024 → 1] with ReLU between layers
 *
 *   Post-processing: exp remap, focal recovery, force projection, mask
 */

import { createStorageBuffer, createEmptyBuffer, readBuffer } from './gpu.js';
import {
  dispatchConv2d,
  dispatchConv1x1,
  dispatchConvTranspose2d,
  dispatchActivation,
  dispatchGroupNorm,
  dispatchPixelShuffle,
  dispatchUpsample,
} from './shader_ops.js';
import { loadWeights } from './weights.js';


// --- Model config from upstream v2.json ---
const MODEL_CONFIG = {
  encoder: {
    backbone: 'dinov2_vitl14',
    intermediateLayers: [5, 11, 17, 23],
    dimOut: 1024,
  },
  neck: {
    dimIn: [1026, 2, 2, 2, 2],
    dimResBlocks: [1024, 256, 128, 64, 32],
    dimOut: [null, null, null, null, null],
    numResBlocks: [0, 2, 2, 2, 0],
    resamplers: ['conv_transpose', 'conv_transpose', 'conv_transpose', 'bilinear'],
    resBlockInNorm: 'none',
    resBlockHiddenNorm: 'none',
  },
  pointsHead: {
    dimIn: [1024, 256, 128, 64, 32],
    dimResBlocks: [1024, 256, 128, 64, 32],
    dimOut: [null, null, null, null, 3],
    numResBlocks: [0, 1, 1, 1, 0],
    resamplers: ['conv_transpose', 'conv_transpose', 'conv_transpose', 'bilinear'],
    resBlockInNorm: 'none',
    resBlockHiddenNorm: 'none',
  },
  normalHead: {
    dimIn: [1024, 256, 128, 64, 32],
    dimResBlocks: [1024, 256, 128, 64, 32],
    dimOut: [null, null, null, null, 3],
    numResBlocks: [0, 1, 1, 1, 0],
    resamplers: ['conv_transpose', 'conv_transpose', 'conv_transpose', 'bilinear'],
    resBlockInNorm: 'none',
    resBlockHiddenNorm: 'none',
  },
  maskHead: {
    dimIn: [1024, 256, 128, 64, 32],
    dimResBlocks: [1024, 256, 128, 64, 32],
    dimOut: [null, null, null, null, 1],
    numResBlocks: [0, 1, 1, 1, 0],
    resamplers: ['conv_transpose', 'conv_transpose', 'conv_transpose', 'bilinear'],
    resBlockInNorm: 'none',
    resBlockHiddenNorm: 'none',
  },
  scaleHead: { dims: [1024, 1024, 1024, 1] },
  remapOutput: 'exp',
  numTokensRange: [1200, 3600],
  patchSize: 14,
};


/**
 * ResidualConvBlock dispatch:
 *   [Norm →] Activation → Conv3x3 → [Norm →] Activation → Conv3x3 + Skip
 *
 * When inNorm='none', the norm layers are identity (MoGe-2 default).
 */
function dispatchResidualConvBlock(device, encoder, inputBuf, weights, params) {
  const { inC, outC, hiddenC, H, W, inNorm, hiddenNorm } = params;

  let x = inputBuf;

  // Norm 1 (skip if 'none')
  if (inNorm !== 'none') {
    const numGroups = inNorm === 'group_norm' ? Math.floor(inC / 32) : 1;
    x = dispatchGroupNorm(device, encoder, x, weights.norm1_scale, weights.norm1_bias,
      { C: inC, H, W, numGroups });
  }

  // ReLU
  x = dispatchActivation(device, encoder, x, null, inC * H * W, 0);

  // Conv3x3 (inC → hiddenC)
  let convOut = dispatchConv2d(device, encoder, x, weights.conv1_weight, weights.conv1_bias,
    { inC, inH: H, inW: W, outC: hiddenC, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });

  x = convOut.buffer;

  // Norm 2 (skip if 'none')
  if (hiddenNorm !== 'none') {
    const numGroups = hiddenNorm === 'group_norm' ? Math.floor(hiddenC / 32) : 1;
    x = dispatchGroupNorm(device, encoder, x, weights.norm2_scale, weights.norm2_bias,
      { C: hiddenC, H, W, numGroups });
  }

  // ReLU
  x = dispatchActivation(device, encoder, x, null, hiddenC * H * W, 0);

  // Conv3x3 (hiddenC → outC)
  convOut = dispatchConv2d(device, encoder, x, weights.conv2_weight, weights.conv2_bias,
    { inC: hiddenC, inH: H, inW: W, outC, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });

  // Skip connection
  let skip;
  if (inC !== outC && weights.skip_weight) {
    skip = dispatchConv1x1(device, encoder, inputBuf, weights.skip_weight, null,
      { inC, outC, H, W }).buffer;
  } else {
    skip = inputBuf;
  }

  // Add
  const out = dispatchActivation(device, encoder, convOut.buffer, skip, outC * H * W, 2);
  return out;
}

/**
 * Resampler dispatch.
 * Type determines the upsampling method:
 *   conv_transpose: ConvTranspose2d(inC, outC, k=2, s=2) → Conv2d(outC, outC, 3, pad=1)
 *   bilinear: Upsample(2x, bilinear) → Conv2d(inC, outC, 3, pad=1)
 */
function dispatchResampler(device, encoder, inputBuf, weights, params) {
  const { inC, outC, H, W, type } = params;

  if (type === 'conv_transpose') {
    // ConvTranspose2d: inC → outC, kernel=2, stride=2
    const deconv = dispatchConvTranspose2d(device, encoder, inputBuf,
      weights.deconv_weight, weights.deconv_bias,
      { inC, inH: H, inW: W, outC, stride: 2 });

    // Conv2d: outC → outC, 3x3, pad=1
    const conv = dispatchConv2d(device, encoder, deconv.buffer,
      weights.conv_weight, weights.conv_bias,
      { inC: outC, inH: deconv.H, inW: deconv.W, outC, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });

    return { buffer: conv.buffer, H: deconv.H, W: deconv.W };
  } else if (type === 'bilinear') {
    // Bilinear upsample 2x
    const upsampled = dispatchUpsample(device, encoder, inputBuf,
      { C: inC, inH: H, inW: W, outH: H * 2, outW: W * 2, mode: 1 });

    // Conv2d: inC → outC, 3x3, pad=1
    const conv = dispatchConv2d(device, encoder, upsampled.buffer,
      weights.conv_weight, weights.conv_bias,
      { inC, inH: upsampled.H, inW: upsampled.W, outC, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });

    return { buffer: conv.buffer, H: upsampled.H, W: upsampled.W };
  }

  throw new Error(`Unsupported resampler type: ${type}`);
}

/**
 * ConvStack dispatch — the core multi-scale decoder.
 *
 * For each level i:
 *   1. input_block: 1x1 conv (dimIn[i] → dimResBlocks[i])
 *   2. Add to running feature from previous level
 *   3. res_blocks: numResBlocks[i] × ResidualConvBlock
 *   4. output_block: 1x1 conv (dimResBlocks[i] → dimOut[i]) if dimOut[i] != null
 *   5. resampler: upsample x2 for next level
 *
 * Returns list of output features per level.
 */
function dispatchConvStack(device, encoder, inFeatures, weights, config) {
  const { dimIn, dimResBlocks, dimOut, numResBlocks, resamplers, resBlockInNorm, resBlockHiddenNorm } = config;
  const numLevels = dimResBlocks.length;
  const outFeatures = [];
  let x = null;

  for (let i = 0; i < numLevels; i++) {
    const H = inFeatures[i].H;
    const W = inFeatures[i].W;

    // input_block: 1x1 conv
    let projected = null;
    if (dimIn[i] != null && inFeatures[i].buffer != null) {
      projected = dispatchConv1x1(device, encoder, inFeatures[i].buffer,
        weights.levels[i].input_weight, weights.levels[i].input_bias,
        { inC: dimIn[i], outC: dimResBlocks[i], H, W });
    }

    // Add to running state or initialize
    if (i === 0) {
      x = projected.buffer;
    } else if (projected) {
      x = dispatchActivation(device, encoder, x, projected.buffer, dimResBlocks[i] * H * W, 2);
    }

    // res_blocks
    for (let j = 0; j < numResBlocks[i]; j++) {
      x = dispatchResidualConvBlock(device, encoder, x, weights.levels[i].res_blocks[j], {
        inC: dimResBlocks[i], outC: dimResBlocks[i], hiddenC: dimResBlocks[i],
        H, W, inNorm: resBlockInNorm, hiddenNorm: resBlockHiddenNorm,
      });
    }

    // output_block: 1x1 conv if dimOut specified
    if (dimOut[i] != null) {
      const out = dispatchConv1x1(device, encoder, x,
        weights.levels[i].output_weight, weights.levels[i].output_bias,
        { inC: dimResBlocks[i], outC: dimOut[i], H, W });
      outFeatures.push({ buffer: out.buffer, C: dimOut[i], H, W });
    } else {
      outFeatures.push({ buffer: x, C: dimResBlocks[i], H, W });
    }

    // resampler between levels
    if (i < numLevels - 1 && resamplers[i]) {
      const resampled = dispatchResampler(device, encoder, x,
        weights.levels[i].resampler, {
          inC: dimResBlocks[i], outC: dimResBlocks[i + 1],
          H, W, type: resamplers[i],
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
  }

  async init(onProgress) {
    try {
      this.weights = await loadWeights(this.device, '/weights.bin', onProgress);
      this.useRealWeights = true;
      console.log('Loaded real MoGe-2 weights');
    } catch (e) {
      console.warn('Failed to load real weights, using stubs:', e.message);
      this.weights = this._createStubWeights();
      this.useRealWeights = false;
    }
  }

  _createStubWeights() {
    const d = this.device;
    const rand = (n) => {
      const data = new Float32Array(n);
      for (let i = 0; i < n; i++) data[i] = (Math.random() - 0.5) * 0.02;
      return createStorageBuffer(d, data);
    };
    const zeros = (n) => createStorageBuffer(d, new Float32Array(n));
    const ones = (n) => {
      const data = new Float32Array(n);
      data.fill(1.0);
      return createStorageBuffer(d, data);
    };

    const makeResBlock = (C, hiddenC) => ({
      norm1_scale: ones(C),
      norm1_bias: zeros(C),
      conv1_weight: rand(hiddenC * C * 3 * 3),
      conv1_bias: zeros(hiddenC),
      norm2_scale: ones(hiddenC),
      norm2_bias: zeros(hiddenC),
      conv2_weight: rand(C * hiddenC * 3 * 3),
      conv2_bias: zeros(C),
      skip_weight: null,
    });

    const makeResampler = (inC, outC, type) => {
      if (type === 'conv_transpose') {
        return {
          deconv_weight: rand(inC * outC * 2 * 2),  // [inC, outC, 2, 2]
          deconv_bias: zeros(outC),
          conv_weight: rand(outC * outC * 3 * 3),
          conv_bias: zeros(outC),
        };
      } else {
        // bilinear: no deconv, just conv after upsample
        return {
          conv_weight: rand(outC * inC * 3 * 3),
          conv_bias: zeros(outC),
        };
      }
    };

    const makeConvStackWeights = (config) => ({
      levels: config.dimResBlocks.map((dimRB, i) => ({
        input_weight: config.dimIn[i] != null ? rand(dimRB * config.dimIn[i]) : null,
        input_bias: config.dimIn[i] != null ? zeros(dimRB) : null,
        res_blocks: Array.from({ length: config.numResBlocks[i] },
          () => makeResBlock(dimRB, dimRB)),
        output_weight: config.dimOut[i] != null ? rand(config.dimOut[i] * dimRB) : null,
        output_bias: config.dimOut[i] != null ? zeros(config.dimOut[i]) : null,
        resampler: i < config.dimResBlocks.length - 1 && config.resamplers[i]
          ? makeResampler(dimRB, config.dimResBlocks[i + 1], config.resamplers[i])
          : null,
      })),
    });

    return {
      neck: makeConvStackWeights(MODEL_CONFIG.neck),
      pointsHead: makeConvStackWeights(MODEL_CONFIG.pointsHead),
      normalHead: makeConvStackWeights(MODEL_CONFIG.normalHead),
      maskHead: makeConvStackWeights(MODEL_CONFIG.maskHead),
    };
  }

  /**
   * Run full inference.
   */
  async run(imageData) {
    const { width, height } = imageData;
    const device = this.device;

    const tokenH = Math.floor(height / MODEL_CONFIG.patchSize);
    const tokenW = Math.floor(width / MODEL_CONFIG.patchSize);

    // --- STUB BACKBONE ---
    // Random features at [encoderDim, tokenH, tokenW]
    const encoderDim = MODEL_CONFIG.encoder.dimOut;
    const stubEncoderData = new Float32Array(encoderDim * tokenH * tokenW);
    for (let i = 0; i < stubEncoderData.length; i++) {
      stubEncoderData[i] = (Math.random() - 0.5) * 0.5;
    }
    const encoderFeatures = createStorageBuffer(device, stubEncoderData);

    // Build input features for neck: encoder features + UV coords concatenated
    // Level 0: [1024 + 2, tokenH, tokenW]
    // Levels 1-4: [2, tokenH*2^i, tokenW*2^i] (UV only)
    const neckInputs = [];
    for (let level = 0; level < 5; level++) {
      const h = tokenH * (2 ** level);
      const w = tokenW * (2 ** level);
      const dimIn = MODEL_CONFIG.neck.dimIn[level];

      const data = new Float32Array(dimIn * h * w);
      if (level === 0) {
        // Copy encoder features then append UV
        for (let c = 0; c < encoderDim; c++) {
          for (let s = 0; s < tokenH * tokenW; s++) {
            data[c * h * w + s] = stubEncoderData[c * tokenH * tokenW + s];
          }
        }
        // UV channels at end
        const aspect = width / height;
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            data[encoderDim * h * w + y * w + x] = ((x + 0.5) / w * 2 - 1) * aspect;
            data[(encoderDim + 1) * h * w + y * w + x] = (y + 0.5) / h * 2 - 1;
          }
        }
      } else {
        // UV only
        const aspect = width / height;
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            data[0 * h * w + y * w + x] = ((x + 0.5) / w * 2 - 1) * aspect;
            data[1 * h * w + y * w + x] = (y + 0.5) / h * 2 - 1;
          }
        }
      }

      neckInputs.push({ buffer: createStorageBuffer(device, data), H: h, W: w });
    }

    // --- NECK ---
    const commandEncoder = device.createCommandEncoder();

    const neckOutputs = dispatchConvStack(device, commandEncoder, neckInputs, this.weights.neck, MODEL_CONFIG.neck);

    // --- HEADS ---
    // Points head takes neck outputs as input
    const pointsInputs = neckOutputs.map((f, i) => ({
      buffer: f.buffer,
      H: f.H,
      W: f.W,
    }));
    const pointsOutputs = dispatchConvStack(device, commandEncoder, pointsInputs, this.weights.pointsHead, MODEL_CONFIG.pointsHead);

    // Normal head
    const normalInputs = neckOutputs.map(f => ({ buffer: f.buffer, H: f.H, W: f.W }));
    const normalOutputs = dispatchConvStack(device, commandEncoder, normalInputs, this.weights.normalHead, MODEL_CONFIG.normalHead);

    // Mask head
    const maskInputs = neckOutputs.map(f => ({ buffer: f.buffer, H: f.H, W: f.W }));
    const maskOutputs = dispatchConvStack(device, commandEncoder, maskInputs, this.weights.maskHead, MODEL_CONFIG.maskHead);

    // Submit all compute work
    device.queue.submit([commandEncoder.finish()]);

    // Read back the final level outputs
    const lastPointsFeature = pointsOutputs[pointsOutputs.length - 1];
    const lastNormalFeature = normalOutputs[normalOutputs.length - 1];
    const lastMaskFeature = maskOutputs[maskOutputs.length - 1];

    const [pointsRaw, normalsRaw, maskRaw] = await Promise.all([
      readBuffer(device, lastPointsFeature.buffer, lastPointsFeature.C * lastPointsFeature.H * lastPointsFeature.W * 4),
      readBuffer(device, lastNormalFeature.buffer, lastNormalFeature.C * lastNormalFeature.H * lastNormalFeature.W * 4),
      readBuffer(device, lastMaskFeature.buffer, lastMaskFeature.C * lastMaskFeature.H * lastMaskFeature.W * 4),
    ]);

    const outH = lastPointsFeature.H;
    const outW = lastPointsFeature.W;

    // Remap points: exp remap (xy * exp(z), exp(z))
    const points = new Float32Array(3 * outH * outW);
    const depth = new Float32Array(outH * outW);
    const colors = new Float32Array(3 * outH * outW);

    for (let i = 0; i < outH * outW; i++) {
      // pointsRaw is in CHW: [3, outH, outW]
      let px = pointsRaw[0 * outH * outW + i];
      let py = pointsRaw[1 * outH * outW + i];
      let pz = pointsRaw[2 * outH * outW + i];

      // exp remap
      const expZ = Math.exp(Math.min(pz, 10)); // clamp to avoid overflow
      px = px * expZ;
      py = py * expZ;
      pz = expZ;

      points[i * 3 + 0] = px;
      points[i * 3 + 1] = py;
      points[i * 3 + 2] = pz;
      depth[i] = pz;

      // Color from input image (resample to output resolution)
      const oy = Math.floor(i / outW);
      const ox = i % outW;
      const srcY = Math.floor(oy * height / outH);
      const srcX = Math.floor(ox * width / outW);
      const srcIdx = srcY * width + srcX;
      colors[i * 3 + 0] = imageData.data[srcIdx * 4 + 0] / 255;
      colors[i * 3 + 1] = imageData.data[srcIdx * 4 + 1] / 255;
      colors[i * 3 + 2] = imageData.data[srcIdx * 4 + 2] / 255;
    }

    // Normals: normalize vectors (CHW → per-pixel vec3)
    const normals = new Float32Array(3 * outH * outW);
    for (let i = 0; i < outH * outW; i++) {
      let nx = normalsRaw[0 * outH * outW + i];
      let ny = normalsRaw[1 * outH * outW + i];
      let nz = normalsRaw[2 * outH * outW + i];
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
      normals[i * 3 + 0] = nx / len;
      normals[i * 3 + 1] = ny / len;
      normals[i * 3 + 2] = nz / len;
    }

    // Clean up input buffers
    encoderFeatures.destroy();
    neckInputs.forEach(f => f.buffer.destroy());

    return { depth, normals, points, colors, width: outW, height: outH };
  }
}
