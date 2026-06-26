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
  dispatchReluConv2d,
  dispatchConv1x1,
  dispatchConvTranspose2d,
  dispatchActivation,
  dispatchGroupNorm,
  dispatchPixelShuffle,
  dispatchUpsample,
} from './shader_ops.js';
import { loadWeights } from './weights.js';
import { DINOv2Backbone } from './backbone.js';

function createGpuTimestampProfile(device, count) {
  if (!device.features?.has?.('timestamp-query')) return null;
  const byteSize = count * 8;
  return {
    route: 'timestamp-query',
    count,
    querySet: device.createQuerySet({ type: 'timestamp', count }),
    resolveBuffer: device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    }),
    readBuffer: device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
  };
}

function writeGpuTimestamp(profile, encoder, index) {
  if (profile) encoder.writeTimestamp(profile.querySet, index);
}

function resolveGpuTimestamps(profile, encoder) {
  if (!profile) return;
  encoder.resolveQuerySet(profile.querySet, 0, profile.count, profile.resolveBuffer, 0);
  encoder.copyBufferToBuffer(profile.resolveBuffer, 0, profile.readBuffer, 0, profile.count * 8);
}

async function readGpuTimestamps(profile) {
  if (!profile) return null;
  await profile.readBuffer.mapAsync(GPUMapMode.READ);
  const values = Array.from(new BigUint64Array(profile.readBuffer.getMappedRange().slice(0)));
  profile.readBuffer.unmap();
  profile.querySet.destroy?.();
  profile.resolveBuffer.destroy();
  profile.readBuffer.destroy();
  return values;
}

function timestampDeltaMs(values, start, end) {
  if (!values || values.length <= end) return null;
  return Number(values[end] - values[start]) / 1e6;
}

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

  // ReLU -> Conv3x3 (inC -> hiddenC). MoGe-2 decoder uses norm='none',
  // so this is the hot path worth fusing.
  let convOut;
  if (inNorm === 'none') {
    convOut = dispatchReluConv2d(device, encoder, x, weights.conv1_weight, weights.conv1_bias,
      { inC, inH: H, inW: W, outC: hiddenC, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });
  } else {
    x = dispatchActivation(device, encoder, x, null, inC * H * W, 0);
    convOut = dispatchConv2d(device, encoder, x, weights.conv1_weight, weights.conv1_bias,
      { inC, inH: H, inW: W, outC: hiddenC, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });
  }

  x = convOut.buffer;

  // Norm 2 (skip if 'none')
  if (hiddenNorm !== 'none') {
    const numGroups = hiddenNorm === 'group_norm' ? Math.floor(hiddenC / 32) : 1;
    x = dispatchGroupNorm(device, encoder, x, weights.norm2_scale, weights.norm2_bias,
      { C: hiddenC, H, W, numGroups });
  }

  // ReLU -> Conv3x3 (hiddenC -> outC)
  if (hiddenNorm === 'none') {
    convOut = dispatchReluConv2d(device, encoder, x, weights.conv2_weight, weights.conv2_bias,
      { inC: hiddenC, inH: H, inW: W, outC, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });
  } else {
    x = dispatchActivation(device, encoder, x, null, hiddenC * H * W, 0);
    convOut = dispatchConv2d(device, encoder, x, weights.conv2_weight, weights.conv2_bias,
      { inC: hiddenC, inH: H, inW: W, outC, kH: 3, kW: 3, padH: 1, padW: 1, strideH: 1, strideW: 1 });
  }

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

async function dispatchConvStackProfiledByLevel(device, encoder, inFeatures, weights, config) {
  const { dimIn, dimResBlocks, dimOut, numResBlocks, resamplers, resBlockInNorm, resBlockHiddenNorm } = config;
  const numLevels = dimResBlocks.length;
  const outFeatures = [];
  const levels = [];
  let x = null;

  for (let i = 0; i < numLevels; i++) {
    const H = inFeatures[i].H;
    const W = inFeatures[i].W;

    let projected = null;
    if (dimIn[i] != null && inFeatures[i].buffer != null) {
      projected = dispatchConv1x1(device, encoder, inFeatures[i].buffer,
        weights.levels[i].input_weight, weights.levels[i].input_bias,
        { inC: dimIn[i], outC: dimResBlocks[i], H, W });
    }

    if (i === 0) {
      x = projected.buffer;
    } else if (projected) {
      x = dispatchActivation(device, encoder, x, projected.buffer, dimResBlocks[i] * H * W, 2);
    }

    for (let j = 0; j < numResBlocks[i]; j++) {
      x = dispatchResidualConvBlock(device, encoder, x, weights.levels[i].res_blocks[j], {
        inC: dimResBlocks[i], outC: dimResBlocks[i], hiddenC: dimResBlocks[i],
        H, W, inNorm: resBlockInNorm, hiddenNorm: resBlockHiddenNorm,
      });
    }

    const outputC = dimOut[i] != null ? dimOut[i] : dimResBlocks[i];
    if (dimOut[i] != null) {
      const out = dispatchConv1x1(device, encoder, x,
        weights.levels[i].output_weight, weights.levels[i].output_bias,
        { inC: dimResBlocks[i], outC: dimOut[i], H, W });
      outFeatures.push({ buffer: out.buffer, C: dimOut[i], H, W });
    } else {
      outFeatures.push({ buffer: x, C: dimResBlocks[i], H, W });
    }

    if (i < numLevels - 1 && resamplers[i]) {
      const resampled = dispatchResampler(device, encoder, x,
        weights.levels[i].resampler, {
          inC: dimResBlocks[i], outC: dimResBlocks[i + 1],
          H, W, type: resamplers[i],
        });
      x = resampled.buffer;
    }

    const waitStart = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const submitWaitMs = performance.now() - waitStart;
    levels.push({
      level: i,
      submitWaitMs,
      shape: [outputC, H, W],
      dimIn: dimIn[i],
      dimResBlocks: dimResBlocks[i],
      numResBlocks: numResBlocks[i],
      resampler: i < numLevels - 1 ? (resamplers[i] || null) : null,
    });
    encoder = device.createCommandEncoder();
  }

  return { outFeatures, encoder, levels };
}

async function submitProfileStage(device, encoder, stages, stage) {
  const waitStart = performance.now();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  stages.push({
    ...stage,
    submitWaitMs: performance.now() - waitStart,
  });
  return device.createCommandEncoder();
}

async function dispatchConvStackProfiledInternals(device, encoder, inFeatures, weights, config, targetLevel) {
  const { dimIn, dimResBlocks, dimOut, numResBlocks, resamplers, resBlockInNorm, resBlockHiddenNorm } = config;
  const numLevels = dimResBlocks.length;
  const outFeatures = [];
  const stages = [];
  let x = null;

  for (let i = 0; i < numLevels; i++) {
    const H = inFeatures[i].H;
    const W = inFeatures[i].W;
    const isTarget = i === targetLevel;

    let projected = null;
    if (dimIn[i] != null && inFeatures[i].buffer != null) {
      projected = dispatchConv1x1(device, encoder, inFeatures[i].buffer,
        weights.levels[i].input_weight, weights.levels[i].input_bias,
        { inC: dimIn[i], outC: dimResBlocks[i], H, W });
    }

    if (i === 0) {
      x = projected.buffer;
    } else if (projected) {
      x = dispatchActivation(device, encoder, x, projected.buffer, dimResBlocks[i] * H * W, 2);
    }

    if (isTarget) {
      encoder = await submitProfileStage(device, encoder, stages, {
        name: 'inputAdd',
        shape: [dimResBlocks[i], H, W],
      });
    }

    for (let j = 0; j < numResBlocks[i]; j++) {
      x = dispatchResidualConvBlock(device, encoder, x, weights.levels[i].res_blocks[j], {
        inC: dimResBlocks[i], outC: dimResBlocks[i], hiddenC: dimResBlocks[i],
        H, W, inNorm: resBlockInNorm, hiddenNorm: resBlockHiddenNorm,
      });
      if (isTarget) {
        encoder = await submitProfileStage(device, encoder, stages, {
          name: `resBlock${j}`,
          shape: [dimResBlocks[i], H, W],
        });
      }
    }

    const outputC = dimOut[i] != null ? dimOut[i] : dimResBlocks[i];
    if (dimOut[i] != null) {
      const out = dispatchConv1x1(device, encoder, x,
        weights.levels[i].output_weight, weights.levels[i].output_bias,
        { inC: dimResBlocks[i], outC: dimOut[i], H, W });
      outFeatures.push({ buffer: out.buffer, C: dimOut[i], H, W });
    } else {
      outFeatures.push({ buffer: x, C: dimResBlocks[i], H, W });
    }

    if (isTarget) {
      if (dimOut[i] != null) {
        encoder = await submitProfileStage(device, encoder, stages, {
          name: 'output',
          shape: [outputC, H, W],
        });
      } else {
        stages.push({
          name: 'output',
          submitWaitMs: 0,
          shape: [outputC, H, W],
          noOp: true,
        });
      }
    }

    if (i < numLevels - 1 && resamplers[i]) {
      const resampled = dispatchResampler(device, encoder, x,
        weights.levels[i].resampler, {
          inC: dimResBlocks[i], outC: dimResBlocks[i + 1],
          H, W, type: resamplers[i],
        });
      x = resampled.buffer;
      if (isTarget) {
        encoder = await submitProfileStage(device, encoder, stages, {
          name: 'resampler',
          shape: [dimResBlocks[i + 1], resampled.H, resampled.W],
          resampler: resamplers[i],
        });
      }
    }

    if (!isTarget) {
      const waitStart = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      encoder = device.createCommandEncoder();
    }
  }

  return { outFeatures, encoder, stages };
}


export class MoGeInference {
  constructor(gpu) {
    this.device = gpu.device;
    this.weights = null;
    this.backbone = null;
  }

  async init(onProgress) {
    try {
      this.weights = await loadWeights(this.device, '/weights.bin', onProgress);
      this.useRealWeights = true;
      console.log('Loaded real MoGe-2 weights');

      // Initialize backbone
      this.backbone = new DINOv2Backbone(this.device);
      this.backbone.init();
      console.log('DINOv2 backbone initialized');

      // Expose for console debugging
      window.__mogeInference = this;
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
   * Try to load test fixture (real encoder features from PyTorch).
   */
  async _loadFixture() {
    try {
      const metaResp = await fetch('/test_fixtures/metadata.json');
      if (!metaResp.ok) return null;
      const meta = await metaResp.json();

      const [featBuf, clsBuf] = await Promise.all([
        fetch('/test_fixtures/encoder_features.bin').then(r => r.arrayBuffer()),
        fetch('/test_fixtures/cls_token.bin').then(r => r.arrayBuffer()),
      ]);

      console.log(`Loaded test fixture: tokenH=${meta.tokenH}, tokenW=${meta.tokenW}`);
      return {
        features: new Float32Array(featBuf),
        clsToken: new Float32Array(clsBuf),
        tokenH: meta.tokenH,
        tokenW: meta.tokenW,
        meta,
      };
    } catch (e) {
      console.warn('No test fixture available:', e.message);
      return null;
    }
  }

  /**
   * Run backbone comparison against PyTorch reference tensors.
   * Usage from console: await window.__mogeInference.runBackboneCompare()
   */
  async runBackboneCompare() {
    if (!this.backbone || !this.useRealWeights) {
      console.error('Backbone not initialized or using stub weights');
      return;
    }
    const device = this.device;
    const tokenH = 37, tokenW = 37;

    // Load normalized input from layer dumps (same image used for PyTorch reference)
    const resp = await fetch('/layer_dumps/input_normalized.bin');
    const inputData = new Float32Array(await resp.arrayBuffer());
    console.log(`Loaded reference input: [3, ${tokenH * 14}, ${tokenW * 14}], ${inputData.length} floats`);

    const imageBuf = createStorageBuffer(device, inputData);
    await this.backbone.debugCompare(imageBuf, this.weights, tokenH, tokenW);
    imageBuf.destroy();
  }

  /**
   * Detailed sub-block analysis of transformer block 0.
   * Usage from console: await window.__mogeInference.runBlock0Compare()
   */
  async runBlock0Compare() {
    if (!this.backbone || !this.useRealWeights) {
      console.error('Backbone not initialized');
      return;
    }
    const device = this.device;
    const tokenH = 37, tokenW = 37;
    const resp = await fetch('/layer_dumps/input_normalized.bin');
    const inputData = new Float32Array(await resp.arrayBuffer());
    const imageBuf = createStorageBuffer(device, inputData);
    await this.backbone.debugBlock0(imageBuf, this.weights, tokenH, tokenW);
    imageBuf.destroy();
  }

  /**
   * Run full inference.
   */
  async run(imageData, options = {}) {
    const totalStart = performance.now();
    const phaseTimings = {};
    const { width, height } = imageData;
    const device = this.device;
    const encoderDim = MODEL_CONFIG.encoder.dimOut;

    // Determine token grid size
    // Use 37x37 to match the pretrained position embedding grid (1370 tokens = 1 CLS + 37*37)
    // This avoids needing position embedding interpolation for now
    const tokenH = 37;
    const tokenW = 37;

    // --- Encoder ---
    let encoderData;
    let clsTokenData = null;
    let useBackbone = this.backbone && this.useRealWeights;

    const preprocessStart = performance.now();

    // Prepare image for backbone: normalize with ImageNet mean/std, resize to tokenH*14 x tokenW*14
    const imgH = tokenH * MODEL_CONFIG.patchSize;
    const imgW = tokenW * MODEL_CONFIG.patchSize;

    // Convert input image to CHW float [0,1], then normalize
    const imageMean = [0.485, 0.456, 0.406];
    const imageStd = [0.229, 0.224, 0.225];
    const normalizedImage = new Float32Array(3 * imgH * imgW);

    // Resize imageData to imgH x imgW using a temp canvas
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = imgW;
    tmpCanvas.height = imgH;
    const tmpCtx = tmpCanvas.getContext('2d');
    // Create ImageBitmap from the original imageData
    const origCanvas = document.createElement('canvas');
    origCanvas.width = width;
    origCanvas.height = height;
    origCanvas.getContext('2d').putImageData(imageData, 0, 0);
    tmpCtx.drawImage(origCanvas, 0, 0, imgW, imgH);
    const resizedData = tmpCtx.getImageData(0, 0, imgW, imgH);

    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < imgH * imgW; i++) {
        const pixel = resizedData.data[i * 4 + c] / 255.0;
        normalizedImage[c * imgH * imgW + i] = (pixel - imageMean[c]) / imageStd[c];
      }
    }
    phaseTimings.preprocessMs = performance.now() - preprocessStart;

    let backboneFeatureBuf = null;
    let backboneClsTokenBuf = null;
    let imageBuf = null;
    const tempUploadBuffers = [];
    const profileStagedGpu = !!options.profileStagedGpu;
    const profileDecoderSubstages = !!options.profileDecoderSubstages;
    const profileNeckLevels = !!options.profileNeckLevels;
    const profileNeckInternalsLevel = Number.isInteger(options.profileNeckInternals?.level)
      ? options.profileNeckInternals.level
      : null;
    const stagedGpuPhaseTimings = profileStagedGpu
      ? { route: 'staged-submits' }
      : null;
    const decoderSubstageTimings = profileDecoderSubstages
      ? { route: 'decoder-staged-submits' }
      : null;
    const neckLevelTimings = profileNeckLevels
      ? { route: 'neck-level-staged-submits', levels: [] }
      : null;
    const neckInternalTimings = profileNeckInternalsLevel !== null
      ? { route: 'neck-internal-staged-submits', level: profileNeckInternalsLevel, stages: [] }
      : null;
    let commandEncoder = device.createCommandEncoder();
    const gpuTimestampProfile = options.profileGpuTimestamps
      ? createGpuTimestampProfile(device, 4)
      : null;
    writeGpuTimestamp(gpuTimestampProfile, commandEncoder, 0);

    if (useBackbone) {
      imageBuf = createStorageBuffer(device, normalizedImage);

      const backboneEncodeStart = performance.now();
      const { featureBuf, clsTokenBuf } = this.backbone.encode(
        commandEncoder, imageBuf, this.weights, tokenH, tokenW
      );
      phaseTimings.backboneEncodeMs = performance.now() - backboneEncodeStart;

      // Keep feature and CLS buffers on GPU — no readback here
      backboneFeatureBuf = featureBuf;
      backboneClsTokenBuf = clsTokenBuf;

      if (profileStagedGpu) {
        const waitStart = performance.now();
        device.queue.submit([commandEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        stagedGpuPhaseTimings.backboneSubmitWaitMs = performance.now() - waitStart;
        commandEncoder = device.createCommandEncoder();
      }
    } else {
      phaseTimings.backboneEncodeMs = 0;
      if (profileStagedGpu) stagedGpuPhaseTimings.backboneSubmitWaitMs = 0;
      // Try fixture, then fall back to random
      const fixture = await this._loadFixture();
      if (fixture) {
        encoderData = fixture.features;
        console.log(`Using fixture encoder features`);
      } else {
        encoderData = new Float32Array(encoderDim * tokenH * tokenW);
        for (let i = 0; i < encoderData.length; i++) {
          encoderData[i] = (Math.random() - 0.5) * 0.5;
        }
        console.log(`Using stub encoder features`);
      }
    }
    writeGpuTimestamp(gpuTimestampProfile, commandEncoder, 1);

    // Try loading PyTorch neck inputs for decoder validation
    const USE_PYTORCH_NECK_INPUTS = false; // Set true to bypass backbone for decoder-only validation
    if (USE_PYTORCH_NECK_INPUTS) {
      try {
        const neckInputs = [];
        for (let level = 0; level < 5; level++) {
          const resp = await fetch(`/layer_dumps/neck_input_${level}.bin`);
          if (!resp.ok) throw new Error(`No neck_input_${level}`);
          const buf = await resp.arrayBuffer();
          const data = new Float32Array(buf);
          const h = tokenH * (2 ** level);
          const w = tokenW * (2 ** level);
          neckInputs.push({ buffer: createStorageBuffer(device, data), H: h, W: w });
        }
        console.log('Using PyTorch neck inputs from layer dumps');
        window.__mogeDebug = window.__mogeDebug || {};
        window.__mogeDebug.neckSource = 'pytorch_layer_dumps';

        // Run decoder with PyTorch neck inputs
        const decoderEncoder = device.createCommandEncoder();
        const neckOutputs = dispatchConvStack(device, decoderEncoder, neckInputs, this.weights.neck, MODEL_CONFIG.neck);
        const pointsInputs = neckOutputs.map(f => ({ buffer: f.buffer, H: f.H, W: f.W }));
        const pointsOutputs = dispatchConvStack(device, decoderEncoder, pointsInputs, this.weights.pointsHead, MODEL_CONFIG.pointsHead);
        const maskInputs = neckOutputs.map(f => ({ buffer: f.buffer, H: f.H, W: f.W }));
        const maskOutputs = dispatchConvStack(device, decoderEncoder, maskInputs, this.weights.maskHead, MODEL_CONFIG.maskHead);
        device.queue.submit([decoderEncoder.finish()]);

        // Compare each decoder level against PyTorch reference
        for (let i = 0; i < neckOutputs.length; i++) {
          const no = neckOutputs[i];
          const gpuData = await readBuffer(device, no.buffer, no.C * no.H * no.W * 4);
          try {
            const refResp = await fetch(`/layer_dumps/neck_output_${i}.bin`);
            if (!refResp.ok) continue;
            const refData = new Float32Array(await refResp.arrayBuffer());
            let maxErr = 0, sumSq = 0, nanCount = 0;
            const n = Math.min(gpuData.length, refData.length);
            for (let j = 0; j < n; j++) {
              if (isNaN(gpuData[j])) { nanCount++; continue; }
              const err = Math.abs(gpuData[j] - refData[j]);
              sumSq += err * err;
              if (err > maxErr) maxErr = err;
            }
            console.log(`Neck ${i} [${no.C},${no.H},${no.W}]: maxErr=${maxErr.toFixed(4)} rmsErr=${Math.sqrt(sumSq/n).toFixed(4)} NaN=${nanCount}`);
          } catch(e) {}
        }
        // Compare bilinear upsample intermediate
        try {
          const upRefResp = await fetch('/layer_dumps/neck_resampler3_upsampled.bin');
          if (upRefResp.ok) {
            const upRef = new Float32Array(await upRefResp.arrayBuffer());
            console.log(`Bilinear upsample ref: ${upRef.length} elements, range=[${Math.min(...Array.from(upRef.slice(0,1000))).toFixed(4)}, ${Math.max(...Array.from(upRef.slice(0,1000))).toFixed(4)}]`);
            // We need to read our upsample output before the conv2d... but the resampler combines them.
            // Instead, let's run a standalone upsample on neck_output_3 and compare
            const neck3Resp = await fetch('/layer_dumps/neck_output_3.bin');
            const neck3Data = new Float32Array(await neck3Resp.arrayBuffer());
            const neck3Buf = createStorageBuffer(device, neck3Data);
            const testEncoder = device.createCommandEncoder();
            const upResult = dispatchUpsample(device, testEncoder, neck3Buf,
              { C: 64, inH: 296, inW: 296, outH: 592, outW: 592, mode: 1 });
            device.queue.submit([testEncoder.finish()]);
            const upGpu = await readBuffer(device, upResult.buffer, 64 * 592 * 592 * 4);
            let upMaxErr = 0, upSumSq = 0;
            for (let j = 0; j < upGpu.length; j++) {
              const err = Math.abs(upGpu[j] - upRef[j]);
              upSumSq += err * err;
              if (err > upMaxErr) upMaxErr = err;
            }
            console.log(`Bilinear upsample comparison: maxErr=${upMaxErr.toFixed(6)} rmsErr=${Math.sqrt(upSumSq/upGpu.length).toFixed(6)}`);
            neck3Buf.destroy();
            upResult.buffer.destroy();
          }
        } catch(e) { console.log('Upsample comparison skipped:', e.message); }

        for (let i = 0; i < pointsOutputs.length; i++) {
          const po = pointsOutputs[i];
          const gpuData = await readBuffer(device, po.buffer, po.C * po.H * po.W * 4);
          try {
            const refResp = await fetch(`/layer_dumps/points_head_output_${i}.bin`);
            if (!refResp.ok) continue;
            const refData = new Float32Array(await refResp.arrayBuffer());
            let maxErr = 0, sumSq = 0, nanCount = 0;
            const n = Math.min(gpuData.length, refData.length);
            for (let j = 0; j < n; j++) {
              if (isNaN(gpuData[j])) { nanCount++; continue; }
              const err = Math.abs(gpuData[j] - refData[j]);
              sumSq += err * err;
              if (err > maxErr) maxErr = err;
            }
            console.log(`Points ${i} [${po.C},${po.H},${po.W}]: maxErr=${maxErr.toFixed(4)} rmsErr=${Math.sqrt(sumSq/n).toFixed(4)} NaN=${nanCount}`);
          } catch(e) {}
        }

        const lastPoints = pointsOutputs[pointsOutputs.length - 1];
        const pointsRaw = await readBuffer(device, lastPoints.buffer, lastPoints.C * lastPoints.H * lastPoints.W * 4);
        const outH = lastPoints.H, outW = lastPoints.W;

        // Same post-processing as below
        const points = new Float32Array(3 * outH * outW);
        const depth = new Float32Array(outH * outW);
        const colors = new Float32Array(3 * outH * outW);
        const normals = new Float32Array(3 * outH * outW);

        for (let i = 0; i < outH * outW; i++) {
          let px = pointsRaw[0 * outH * outW + i];
          let py = pointsRaw[1 * outH * outW + i];
          let pz = pointsRaw[2 * outH * outW + i];
          const expZ = Math.exp(Math.min(pz, 10));
          px = px * expZ; py = py * expZ; pz = expZ;
          points[i * 3] = px; points[i * 3 + 1] = py; points[i * 3 + 2] = pz;
          depth[i] = pz;
          const oy = Math.floor(i / outW), ox = i % outW;
          const srcY = Math.min(Math.floor(oy * height / outH), height - 1);
          const srcX = Math.min(Math.floor(ox * width / outW), width - 1);
          const srcIdx = srcY * width + srcX;
          colors[i * 3] = imageData.data[srcIdx * 4] / 255;
          colors[i * 3 + 1] = imageData.data[srcIdx * 4 + 1] / 255;
          colors[i * 3 + 2] = imageData.data[srcIdx * 4 + 2] / 255;
        }
        for (let y = 0; y < outH; y++) {
          for (let x = 0; x < outW; x++) {
            const i = y * outW + x;
            const ir = y * outW + Math.min(x + 1, outW - 1);
            const ib = Math.min(y + 1, outH - 1) * outW + x;
            const dxX = points[ir*3]-points[i*3], dxY = points[ir*3+1]-points[i*3+1], dxZ = points[ir*3+2]-points[i*3+2];
            const dyX = points[ib*3]-points[i*3], dyY = points[ib*3+1]-points[i*3+1], dyZ = points[ib*3+2]-points[i*3+2];
            let nx = dxY*dyZ-dxZ*dyY, ny = dxZ*dyX-dxX*dyZ, nz = dxX*dyY-dxY*dyX;
            const len = Math.sqrt(nx*nx+ny*ny+nz*nz)||1;
            normals[i*3]=nx/len; normals[i*3+1]=ny/len; normals[i*3+2]=nz/len;
          }
        }

        let pMin=Infinity,pMax=-Infinity;
        for(let i=0;i<pointsRaw.length;i++){if(pointsRaw[i]<pMin)pMin=pointsRaw[i];if(pointsRaw[i]>pMax)pMax=pointsRaw[i];}
        window.__mogeDebug.pointsDiag = `shape=[${lastPoints.C},${outH},${outW}], range=[${pMin.toFixed(4)},${pMax.toFixed(4)}]`;
        let dMin=Infinity,dMax=-Infinity;
        for(let i=0;i<depth.length;i++){if(depth[i]<dMin)dMin=depth[i];if(depth[i]>dMax)dMax=depth[i];}
        window.__mogeDebug.depthRange = `[${dMin.toFixed(4)},${dMax.toFixed(4)}]`;

        neckInputs.forEach(f => f.buffer.destroy());
        return { depth, normals, points, colors, width: outW, height: outH };
      } catch (e) {
        console.warn('PyTorch neck inputs not available, using backbone:', e.message);
      }
    }

    // Build neck input features: encoder features + UV coords at 5 scales
    const aspect = width / height;
    const neckInputs = [];

    function makeUV(h, w, aspect) {
      const spanX = aspect / Math.sqrt(1 + aspect * aspect);
      const spanY = 1 / Math.sqrt(1 + aspect * aspect);
      const uv = new Float32Array(2 * h * w);
      for (let y = 0; y < h; y++) {
        const v = -spanY * (h - 1) / h + (2 * spanY * (h - 1) / h) * y / (h - 1 || 1);
        for (let x = 0; x < w; x++) {
          const u = -spanX * (w - 1) / w + (2 * spanX * (w - 1) / w) * x / (w - 1 || 1);
          uv[0 * h * w + y * w + x] = u;
          uv[1 * h * w + y * w + x] = v;
        }
      }
      return uv;
    }

    for (let level = 0; level < 5; level++) {
      const h = tokenH * (2 ** level);
      const w = tokenW * (2 ** level);
      const dimIn = MODEL_CONFIG.neck.dimIn[level];

      if (level === 0 && backboneFeatureBuf) {
        // Zero-copy: concatenate backbone GPU buffer with UV on GPU
        const totalSize = dimIn * h * w * 4;
        const combinedBuf = createEmptyBuffer(device, totalSize);
        // Copy backbone features [1024, tokenH, tokenW] from GPU buffer
        const featureBytes = encoderDim * h * w * 4;
        commandEncoder.copyBufferToBuffer(backboneFeatureBuf, 0, combinedBuf, 0, featureBytes);
        // Upload UV coords to the remaining 2*h*w region
        const uv = makeUV(h, w, aspect);
        const uvBuf = createStorageBuffer(device, uv);
        commandEncoder.copyBufferToBuffer(uvBuf, 0, combinedBuf, featureBytes, uv.byteLength);
        tempUploadBuffers.push(uvBuf);
        neckInputs.push({ buffer: combinedBuf, H: h, W: w });
      } else if (level === 0 && encoderData) {
        // CPU fallback path (stub/fixture features)
        const data = new Float32Array(dimIn * h * w);
        const uv = makeUV(h, w, aspect);
        data.set(encoderData, 0);
        data.set(uv, encoderDim * h * w);
        neckInputs.push({ buffer: createStorageBuffer(device, data), H: h, W: w });
      } else {
        // UV only [2, h, w]
        const uv = makeUV(h, w, aspect);
        neckInputs.push({ buffer: createStorageBuffer(device, uv), H: h, W: w });
      }
    }
    writeGpuTimestamp(gpuTimestampProfile, commandEncoder, 2);
    if (profileStagedGpu) {
      const waitStart = performance.now();
      device.queue.submit([commandEncoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      stagedGpuPhaseTimings.neckInputSubmitWaitMs = performance.now() - waitStart;
      commandEncoder = device.createCommandEncoder();
    }

    // Submit backbone (if used) and start decoder in one encoder
    // No separate submit — backbone and decoder buffer copies share this encoder
    let decoderEncoder = commandEncoder;
    if (profileNeckLevels || neckInternalTimings) {
      const waitStart = performance.now();
      device.queue.submit([decoderEncoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const preNeckSubmitWaitMs = performance.now() - waitStart;
      if (profileNeckLevels) neckLevelTimings.preNeckSubmitWaitMs = preNeckSubmitWaitMs;
      if (neckInternalTimings) neckInternalTimings.preNeckSubmitWaitMs = preNeckSubmitWaitMs;
      decoderEncoder = device.createCommandEncoder();
    }

    // Neck
    const neckAndHeadsEncodeStart = performance.now();
    let neckOutputs;
    if (neckInternalTimings) {
      const profiledNeck = await dispatchConvStackProfiledInternals(
        device, decoderEncoder, neckInputs, this.weights.neck, MODEL_CONFIG.neck, profileNeckInternalsLevel
      );
      neckOutputs = profiledNeck.outFeatures;
      decoderEncoder = profiledNeck.encoder;
      neckInternalTimings.stages = profiledNeck.stages;
      neckInternalTimings.totalLevelInternalMs = profiledNeck.stages.reduce((sum, stage) => sum + stage.submitWaitMs, 0);
    } else if (profileNeckLevels) {
      const profiledNeck = await dispatchConvStackProfiledByLevel(
        device, decoderEncoder, neckInputs, this.weights.neck, MODEL_CONFIG.neck
      );
      neckOutputs = profiledNeck.outFeatures;
      decoderEncoder = profiledNeck.encoder;
      neckLevelTimings.levels = profiledNeck.levels;
      neckLevelTimings.totalNeckLevelMs = profiledNeck.levels.reduce((sum, level) => sum + level.submitWaitMs, 0);
    } else {
      neckOutputs = dispatchConvStack(device, decoderEncoder, neckInputs, this.weights.neck, MODEL_CONFIG.neck);
    }
    if (profileDecoderSubstages) {
      const waitStart = performance.now();
      device.queue.submit([decoderEncoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      decoderSubstageTimings.neckSubmitWaitMs = performance.now() - waitStart;
      decoderEncoder = device.createCommandEncoder();
    }

    // Points head
    const pointsInputs = neckOutputs.map(f => ({ buffer: f.buffer, H: f.H, W: f.W }));
    const pointsOutputs = dispatchConvStack(device, decoderEncoder, pointsInputs, this.weights.pointsHead, MODEL_CONFIG.pointsHead);
    if (profileDecoderSubstages) {
      const waitStart = performance.now();
      device.queue.submit([decoderEncoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      decoderSubstageTimings.pointsHeadSubmitWaitMs = performance.now() - waitStart;
      decoderEncoder = device.createCommandEncoder();
    }

    // Normal head
    const normalInputs = neckOutputs.map(f => ({ buffer: f.buffer, H: f.H, W: f.W }));
    const normalOutputs = dispatchConvStack(device, decoderEncoder, normalInputs, this.weights.normalHead, MODEL_CONFIG.normalHead);
    if (profileDecoderSubstages) {
      const waitStart = performance.now();
      device.queue.submit([decoderEncoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      decoderSubstageTimings.normalHeadSubmitWaitMs = performance.now() - waitStart;
      decoderEncoder = device.createCommandEncoder();
    }

    // Mask head
    const maskInputs = neckOutputs.map(f => ({ buffer: f.buffer, H: f.H, W: f.W }));
    const maskOutputs = dispatchConvStack(device, decoderEncoder, maskInputs, this.weights.maskHead, MODEL_CONFIG.maskHead);
    if (profileDecoderSubstages) {
      const waitStart = performance.now();
      device.queue.submit([decoderEncoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      decoderSubstageTimings.maskHeadSubmitWaitMs = performance.now() - waitStart;
    }
    phaseTimings.neckAndHeadsEncodeMs = performance.now() - neckAndHeadsEncodeStart;
    if (!profileDecoderSubstages) {
      writeGpuTimestamp(gpuTimestampProfile, decoderEncoder, 3);
      resolveGpuTimestamps(gpuTimestampProfile, decoderEncoder);
    }

    // Submit entire pipeline (backbone + decoder in one encoder)
    if (profileDecoderSubstages) {
      // Decoder substages were already submitted above.
    } else if (profileStagedGpu) {
      const waitStart = performance.now();
      device.queue.submit([decoderEncoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      stagedGpuPhaseTimings.decoderSubmitWaitMs = performance.now() - waitStart;
    } else {
      device.queue.submit([decoderEncoder.finish()]);
    }
    if (imageBuf) imageBuf.destroy();
    tempUploadBuffers.forEach(buf => buf.destroy());

    // Read back final outputs (only the necessary ones)
    const lastPoints = pointsOutputs[pointsOutputs.length - 1];
    const lastNormals = normalOutputs[normalOutputs.length - 1];
    const lastMask = maskOutputs[maskOutputs.length - 1];

    const gpuReadbackStart = performance.now();
    const [pointsRaw, normalsRaw, maskRaw] = await Promise.all([
      readBuffer(device, lastPoints.buffer, lastPoints.C * lastPoints.H * lastPoints.W * 4),
      readBuffer(device, lastNormals.buffer, lastNormals.C * lastNormals.H * lastNormals.W * 4),
      readBuffer(device, lastMask.buffer, lastMask.C * lastMask.H * lastMask.W * 4),
    ]);
    phaseTimings.gpuReadbackMs = performance.now() - gpuReadbackStart;
    if (profileStagedGpu) stagedGpuPhaseTimings.outputReadbackMs = phaseTimings.gpuReadbackMs;
    const gpuTimestamps = await readGpuTimestamps(gpuTimestampProfile);

    const outH = lastPoints.H;
    const outW = lastPoints.W;

    // Diagnostic: check raw output ranges
    let pMin = Infinity, pMax = -Infinity, pNan = 0, pZero = 0;
    for (let i = 0; i < pointsRaw.length; i++) {
      if (isNaN(pointsRaw[i])) pNan++;
      if (pointsRaw[i] === 0) pZero++;
      if (isFinite(pointsRaw[i])) {
        pMin = Math.min(pMin, pointsRaw[i]);
        pMax = Math.max(pMax, pointsRaw[i]);
      }
    }
    const pointsDiag = `shape=[${lastPoints.C}, ${outH}, ${outW}], range=[${pMin.toFixed(4)}, ${pMax.toFixed(4)}], NaN=${pNan}, zeros=${pZero}/${pointsRaw.length}`;
    console.log(`Points raw: ${pointsDiag}`);
    window.__mogeDebug = window.__mogeDebug || {};
    window.__mogeDebug.pointsDiag = pointsDiag;
    window.__mogeDebug.outputSize = `${outW}x${outH}`;

    let mMin = Infinity, mMax = -Infinity;
    for (let i = 0; i < maskRaw.length; i++) {
      if (isFinite(maskRaw[i])) {
        mMin = Math.min(mMin, maskRaw[i]);
        mMax = Math.max(mMax, maskRaw[i]);
      }
    }
    console.log(`Mask raw: range=[${mMin.toFixed(4)}, ${mMax.toFixed(4)}]`);

    // Read CLS token for scale head (deferred from backbone to avoid mid-pipeline sync)
    if (backboneClsTokenBuf && !clsTokenData) {
      const clsReadbackStart = performance.now();
      clsTokenData = await readBuffer(device, backboneClsTokenBuf, encoderDim * 4);
      phaseTimings.clsReadbackMs = performance.now() - clsReadbackStart;
    } else {
      phaseTimings.clsReadbackMs = 0;
    }

    // Scale head: CLS token → metric scale via MLP (1024→1024→1024→1) + exp
    const scaleHeadStart = performance.now();
    let metricScale = 1.0;
    if (clsTokenData && this.weights.scaleHead) {
      let x = clsTokenData;
      for (let li = 0; li < this.weights.scaleHead.layers.length; li++) {
        const { weight, bias, inDim, outDim } = this.weights.scaleHead.layers[li];
        const out = new Float32Array(outDim);
        for (let o = 0; o < outDim; o++) {
          let sum = bias[o];
          for (let k = 0; k < inDim; k++) {
            sum += x[k] * weight[k * outDim + o];
          }
          // ReLU between layers (not after the last)
          out[o] = (li < this.weights.scaleHead.layers.length - 1) ? Math.max(0, sum) : sum;
        }
        x = out;
      }
      metricScale = Math.exp(x[0]);
      console.log(`Scale head: raw=${x[0].toFixed(4)}, metric_scale=${metricScale.toFixed(4)}`);
    }
    phaseTimings.scaleHeadMs = performance.now() - scaleHeadStart;

    // Post-processing: exp remap
    const postprocessStart = performance.now();
    const points = new Float32Array(3 * outH * outW);
    const depth = new Float32Array(outH * outW);
    const colors = new Float32Array(3 * outH * outW);

    for (let i = 0; i < outH * outW; i++) {
      let px = pointsRaw[0 * outH * outW + i];
      let py = pointsRaw[1 * outH * outW + i];
      let pz = pointsRaw[2 * outH * outW + i];

      // exp remap: xy = xy * exp(z), z = exp(z)
      const expZ = Math.exp(Math.min(pz, 10));
      px = px * expZ;
      py = py * expZ;
      pz = expZ;

      points[i * 3 + 0] = px;
      points[i * 3 + 1] = py;
      points[i * 3 + 2] = pz;
      depth[i] = pz;

      // Color from input image
      const oy = Math.floor(i / outW);
      const ox = i % outW;
      const srcY = Math.min(Math.floor(oy * height / outH), height - 1);
      const srcX = Math.min(Math.floor(ox * width / outW), width - 1);
      const srcIdx = srcY * width + srcX;
      colors[i * 3 + 0] = imageData.data[srcIdx * 4 + 0] / 255;
      colors[i * 3 + 1] = imageData.data[srcIdx * 4 + 1] / 255;
      colors[i * 3 + 2] = imageData.data[srcIdx * 4 + 2] / 255;
    }

    // Normals from normal_head (moge-2-vitl-normal model)
    // normalsRaw is CHW planar [3, outH, outW], needs L2 normalization per pixel
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

    // Apply metric scale
    if (metricScale !== 1.0) {
      for (let i = 0; i < points.length; i++) points[i] *= metricScale;
      for (let i = 0; i < depth.length; i++) depth[i] *= metricScale;
    }

    let dMin = Infinity, dMax = -Infinity;
    for (let i = 0; i < depth.length; i++) {
      if (isFinite(depth[i])) {
        dMin = Math.min(dMin, depth[i]);
        dMax = Math.max(dMax, depth[i]);
      }
    }

    // Clean up
    neckInputs.forEach(f => f.buffer.destroy());

    phaseTimings.postprocessMs = performance.now() - postprocessStart;
    phaseTimings.totalMs = performance.now() - totalStart;
    window.__mogeDebug = window.__mogeDebug || {};
    window.__mogeDebug.depthRange = `[${dMin.toFixed(4)}, ${dMax.toFixed(4)}]`;
    window.__mogeDebug.phaseTimings = phaseTimings;
    if (profileStagedGpu) {
      stagedGpuPhaseTimings.totalProfiledGpuMs =
        stagedGpuPhaseTimings.backboneSubmitWaitMs +
        stagedGpuPhaseTimings.neckInputSubmitWaitMs +
        stagedGpuPhaseTimings.decoderSubmitWaitMs +
        stagedGpuPhaseTimings.outputReadbackMs;
      window.__mogeDebug.stagedGpuPhaseTimings = stagedGpuPhaseTimings;
    }
    if (profileDecoderSubstages) {
      decoderSubstageTimings.totalDecoderSubstageMs =
        decoderSubstageTimings.neckSubmitWaitMs +
        decoderSubstageTimings.pointsHeadSubmitWaitMs +
        decoderSubstageTimings.normalHeadSubmitWaitMs +
        decoderSubstageTimings.maskHeadSubmitWaitMs;
      window.__mogeDebug.decoderSubstageTimings = decoderSubstageTimings;
    }
    if (profileNeckLevels) {
      window.__mogeDebug.neckLevelTimings = neckLevelTimings;
    }
    if (neckInternalTimings) {
      window.__mogeDebug.neckInternalTimings = neckInternalTimings;
    }
    if (gpuTimestamps) {
      window.__mogeDebug.gpuPhaseTimings = {
        route: 'timestamp-query',
        timestampUnit: 'nanoseconds',
        timestamps: gpuTimestamps.map(String),
        backboneGpuMs: timestampDeltaMs(gpuTimestamps, 0, 1),
        neckInputGpuMs: timestampDeltaMs(gpuTimestamps, 1, 2),
        decoderGpuMs: timestampDeltaMs(gpuTimestamps, 2, 3),
        mainPassGpuMs: timestampDeltaMs(gpuTimestamps, 0, 3),
      };
    } else if (options.profileGpuTimestamps) {
      window.__mogeDebug.gpuPhaseTimings = {
        route: 'unavailable',
        reason: 'timestamp-query feature was not present on the GPUDevice',
      };
    }

    return { depth, normals, points, colors, width: outW, height: outH, metricScale };
  }
}
