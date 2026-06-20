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
import { DINOv2Backbone } from './backbone.js';


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
  // NOTE: moge-2-vitl does NOT have a normal_head.
  // Normals are computed from the point map via finite differences.
  // moge-2-vits-normal has a normal_head but we're targeting vitl.
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
   * Run full inference.
   */
  async run(imageData) {
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
    let useBackbone = this.backbone && this.useRealWeights;

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

    const commandEncoder = device.createCommandEncoder();

    if (useBackbone) {
      console.log(`Running DINOv2 backbone: image [3, ${imgH}, ${imgW}] → tokens [${tokenH}x${tokenW}]`);
      const imageBuf = createStorageBuffer(device, normalizedImage);

      const { featureBuf } = this.backbone.encode(
        commandEncoder, imageBuf, this.weights, tokenH, tokenW
      );

      // Submit backbone compute, read back features
      device.queue.submit([commandEncoder.finish()]);

      const numPatches = tokenH * tokenW;
      encoderData = await readBuffer(device, featureBuf, encoderDim * numPatches * 4);
      const backboneRange = `[${Math.min(...encoderData.slice(0, 100)).toFixed(3)}, ${Math.max(...encoderData.slice(0, 100)).toFixed(3)}]`;
      console.log(`Backbone output: [${encoderDim}, ${tokenH}, ${tokenW}], range=${backboneRange}`);
      // Write diagnostics to DOM for external reading
      window.__mogeDebug = window.__mogeDebug || {};
      window.__mogeDebug.backboneRange = backboneRange;
      window.__mogeDebug.backboneShape = [encoderDim, tokenH, tokenW];

      imageBuf.destroy();
    } else {
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

    // Build neck input features: encoder features + UV coords at 5 scales
    const aspect = width / height;
    const neckInputs = [];
    for (let level = 0; level < 5; level++) {
      const h = tokenH * (2 ** level);
      const w = tokenW * (2 ** level);
      const dimIn = MODEL_CONFIG.neck.dimIn[level];

      const data = new Float32Array(dimIn * h * w);
      if (level === 0) {
        // Encoder features [1024, tokenH, tokenW] + UV [2, tokenH, tokenW]
        data.set(encoderData, 0);
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            data[encoderDim * h * w + y * w + x] = ((x + 0.5) / w * 2 - 1) * aspect;
            data[(encoderDim + 1) * h * w + y * w + x] = (y + 0.5) / h * 2 - 1;
          }
        }
      } else {
        // UV only [2, h, w]
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            data[0 * h * w + y * w + x] = ((x + 0.5) / w * 2 - 1) * aspect;
            data[1 * h * w + y * w + x] = (y + 0.5) / h * 2 - 1;
          }
        }
      }
      neckInputs.push({ buffer: createStorageBuffer(device, data), H: h, W: w });
    }

    // --- Decoder dispatch ---
    const decoderEncoder = device.createCommandEncoder();

    // Neck
    const neckOutputs = dispatchConvStack(device, decoderEncoder, neckInputs, this.weights.neck, MODEL_CONFIG.neck);

    // Points head
    const pointsInputs = neckOutputs.map(f => ({ buffer: f.buffer, H: f.H, W: f.W }));
    const pointsOutputs = dispatchConvStack(device, decoderEncoder, pointsInputs, this.weights.pointsHead, MODEL_CONFIG.pointsHead);

    // Mask head
    const maskInputs = neckOutputs.map(f => ({ buffer: f.buffer, H: f.H, W: f.W }));
    const maskOutputs = dispatchConvStack(device, decoderEncoder, maskInputs, this.weights.maskHead, MODEL_CONFIG.maskHead);

    // Submit decoder
    device.queue.submit([decoderEncoder.finish()]);

    // Read back
    const lastPoints = pointsOutputs[pointsOutputs.length - 1];
    const lastMask = maskOutputs[maskOutputs.length - 1];

    const [pointsRaw, maskRaw] = await Promise.all([
      readBuffer(device, lastPoints.buffer, lastPoints.C * lastPoints.H * lastPoints.W * 4),
      readBuffer(device, lastMask.buffer, lastMask.C * lastMask.H * lastMask.W * 4),
    ]);

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

    let mMin = Infinity, mMax = -Infinity;
    for (let i = 0; i < maskRaw.length; i++) {
      if (isFinite(maskRaw[i])) {
        mMin = Math.min(mMin, maskRaw[i]);
        mMax = Math.max(mMax, maskRaw[i]);
      }
    }
    console.log(`Mask raw: range=[${mMin.toFixed(4)}, ${mMax.toFixed(4)}]`);

    // Post-processing: exp remap
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

    // Compute normals from point map via finite differences (cross product)
    // No normal_head in moge-2-vitl — normals are derived from geometry
    const normals = new Float32Array(3 * outH * outW);
    for (let y = 0; y < outH; y++) {
      for (let x = 0; x < outW; x++) {
        const i = y * outW + x;
        const x1 = Math.min(x + 1, outW - 1);
        const y1 = Math.min(y + 1, outH - 1);
        const ir = y * outW + x1;
        const ib = y1 * outW + x;

        // dP/dx and dP/dy
        const dxX = points[ir * 3 + 0] - points[i * 3 + 0];
        const dxY = points[ir * 3 + 1] - points[i * 3 + 1];
        const dxZ = points[ir * 3 + 2] - points[i * 3 + 2];
        const dyX = points[ib * 3 + 0] - points[i * 3 + 0];
        const dyY = points[ib * 3 + 1] - points[i * 3 + 1];
        const dyZ = points[ib * 3 + 2] - points[i * 3 + 2];

        // Cross product
        let nx = dxY * dyZ - dxZ * dyY;
        let ny = dxZ * dyX - dxX * dyZ;
        let nz = dxX * dyY - dxY * dyX;
        const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
        normals[i * 3 + 0] = nx / len;
        normals[i * 3 + 1] = ny / len;
        normals[i * 3 + 2] = nz / len;
      }
    }

    // Clean up
    neckInputs.forEach(f => f.buffer.destroy());

    return { depth, normals, points, colors, width: outW, height: outH };
  }
}
