/**
 * backbone.js — DINOv2 ViT-Large backbone dispatch for MoGe-2.
 *
 * Architecture:
 *   1. Patch embedding: image → [N+1, 1024] tokens (14×14 patches + CLS)
 *   2. 24 transformer blocks, each:
 *      a. LayerNorm1 → Attention (QKV → scores → softmax → apply → proj) → LayerScale1 + residual
 *      b. LayerNorm2 → SwiGLU FFN (w12 → gate → w3) → LayerScale2 + residual
 *   3. Extract intermediate features at layers [5, 11, 17, 23]
 *   4. Project each intermediate feature with 1x1 conv and sum
 *
 * Produces: [1024, tokenH, tokenW] feature map + [1024] CLS token
 */

import { createStorageBuffer, createEmptyBuffer, readBuffer } from './gpu.js';

import patchEmbedWGSL from '../shaders/patch_embed_dinov2.wgsl?raw';
import layerNormWGSL from '../shaders/layernorm_vit.wgsl?raw';
import attentionWGSL from '../shaders/attention.wgsl?raw';
import linearWGSL from '../shaders/linear.wgsl?raw';
import linearGeluWGSL from '../shaders/linear_gelu.wgsl?raw';
import layerscaleWGSL from '../shaders/layerscale.wgsl?raw';
import transposeWGSL from '../shaders/transpose_nd.wgsl?raw';

const MAX_WG = 65535;
function splitWG(total) {
  if (total <= MAX_WG) return [total, 1];
  return [MAX_WG, Math.ceil(total / MAX_WG)];
}
function ceilDiv(a, b) { return Math.ceil(a / b); }

function makeUniform(device, data) {
  const buf = device.createBuffer({
    size: Math.max(data.byteLength, 16),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data.buffer || data));
  buf.unmap();
  return buf;
}

// DINOv2 ViT-Large config
const VIT_CONFIG = {
  dim: 1024,
  numHeads: 16,
  headDim: 64,
  numLayers: 24,
  patchSize: 14,
  channels: 3,
  intermediateLayers: [5, 11, 17, 23],
  // Standard GELU MLP (not SwiGLU — verified from checkpoint weights)
  mlpHiddenDim: 4096,
  scale: 1.0 / Math.sqrt(64),
  eps: 1e-6,
};

export class DINOv2Backbone {
  constructor(device) {
    this.device = device;
    this.pipelines = {};
  }

  init() {
    const device = this.device;
    const make = (code, entry) => device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code }), entryPoint: entry },
    });

    this.pipelines.patchEmbed = make(patchEmbedWGSL, 'main');
    this.pipelines.layerNorm = make(layerNormWGSL, 'main');
    this.pipelines.attnScores = make(attentionWGSL, 'computeScores');
    this.pipelines.attnSoftmax = make(attentionWGSL, 'softmax');
    this.pipelines.attnApply = make(attentionWGSL, 'applyAttn');
    this.pipelines.linear = make(linearWGSL, 'main');
    this.pipelines.linearGelu = make(linearGeluWGSL, 'main');
    this.pipelines.layerScale = make(layerscaleWGSL, 'main');
    this.pipelines.transpose = make(transposeWGSL, 'main');
  }

  /**
   * Run the DINOv2 backbone.
   * @param {GPUCommandEncoder} encoder
   * @param {GPUBuffer} imageBuf - [3, imgH, imgW] normalized CHW image
   * @param {Object} weights - encoder weights from weight loader
   * @param {number} tokenH
   * @param {number} tokenW
   * @returns {{ featureBuf: GPUBuffer, clsTokenBuf: GPUBuffer }}
   */
  async encode(encoder, imageBuf, weights, tokenH, tokenW) {
    const device = this.device;
    const D = VIT_CONFIG.dim;
    const numPatches = tokenH * tokenW;
    const N = numPatches + 1; // +1 for CLS
    const T = N * D; // total token elements

    // --- Patch embedding ---
    const tokenBuf = createEmptyBuffer(device, T * 4);
    this._encodePatchEmbed(encoder, imageBuf, weights, tokenBuf, tokenH, tokenW);

    // --- Transformer blocks ---
    // Intermediate feature buffers for extraction
    const intermediateFeatures = [];
    let currentTokens = tokenBuf;

    // Working buffers (reused across layers)
    const normBuf = createEmptyBuffer(device, T * 4);
    const qBuf = createEmptyBuffer(device, T * 4);
    const kBuf = createEmptyBuffer(device, T * 4);
    const vBuf = createEmptyBuffer(device, T * 4);
    const scoreBuf = createEmptyBuffer(device, VIT_CONFIG.numHeads * N * N * 4);
    const attnOutBuf = createEmptyBuffer(device, T * 4);
    const projOutBuf = createEmptyBuffer(device, T * 4);
    const hiddenBuf = createEmptyBuffer(device, N * VIT_CONFIG.mlpHiddenDim * 4);
    const ffnOutBuf = createEmptyBuffer(device, T * 4);
    // Two token buffers for ping-pong (avoid read/write race on same buffer)
    let tokenBufA = tokenBuf;
    let tokenBufB = createEmptyBuffer(device, T * 4);

    for (let l = 0; l < VIT_CONFIG.numLayers; l++) {
      // LayerNorm1
      this._encodeLayerNorm(encoder, currentTokens, normBuf, weights, `encoder.backbone.blocks.${l}.norm1`, N);

      // Attention: QKV projections
      this._encodeQKV(encoder, normBuf, qBuf, kBuf, vBuf, weights, l, N);

      // Attention scores
      this._encodeAttnScores(encoder, qBuf, kBuf, scoreBuf, N);

      // Softmax
      this._encodeAttnSoftmax(encoder, scoreBuf, N);

      // Apply attention
      this._encodeAttnApply(encoder, scoreBuf, vBuf, attnOutBuf, N);

      // Output projection
      this._encodeLinear(encoder, attnOutBuf, projOutBuf, weights, `encoder.backbone.blocks.${l}.attn.proj`, N, D, D);

      // LayerScale1 + residual: output = currentTokens + ls1.gamma * projOutBuf
      // Write to the OTHER buffer to avoid read/write race
      const attnResidualOut = (currentTokens === tokenBufA) ? tokenBufB : tokenBufA;
      this._encodeLayerScaleResidual(encoder, projOutBuf, currentTokens, attnResidualOut, weights, `encoder.backbone.blocks.${l}.ls1`, T, D);
      currentTokens = attnResidualOut;

      // LayerNorm2
      this._encodeLayerNorm(encoder, currentTokens, normBuf, weights, `encoder.backbone.blocks.${l}.norm2`, N);

      // GELU MLP: fc1 (linear+GELU) then fc2 (linear)
      this._encodeLinearGelu(encoder, normBuf, hiddenBuf, weights, `encoder.backbone.blocks.${l}.mlp.fc1`, N, D, VIT_CONFIG.mlpHiddenDim);
      this._encodeLinear(encoder, hiddenBuf, ffnOutBuf, weights, `encoder.backbone.blocks.${l}.mlp.fc2`, N, VIT_CONFIG.mlpHiddenDim, D);

      // LayerScale2 + residual: write to the other buffer
      const ffnResidualOut = (currentTokens === tokenBufA) ? tokenBufB : tokenBufA;
      this._encodeLayerScaleResidual(encoder, ffnOutBuf, currentTokens, ffnResidualOut, weights, `encoder.backbone.blocks.${l}.ls2`, T, D);
      currentTokens = ffnResidualOut;

      // Debug: read back token state after each layer to trace where signal dies
      if (l === 0 || l === 5 || l === 11 || l === 23) {
        // Submit what we have so far and read back
        device.queue.submit([encoder.finish()]);
        const debugData = await readBuffer(device, currentTokens, Math.min(T * 4, 4096));
        let dMin = Infinity, dMax = -Infinity;
        for (let i = 0; i < debugData.length; i++) {
          if (debugData[i] < dMin) dMin = debugData[i];
          if (debugData[i] > dMax) dMax = debugData[i];
        }
        console.log(`Layer ${l} tokens: range=[${dMin.toFixed(4)}, ${dMax.toFixed(4)}]`);
        window.__mogeDebug = window.__mogeDebug || {};
        window.__mogeDebug[`layer${l}`] = `[${dMin.toFixed(4)}, ${dMax.toFixed(4)}]`;
        // Start a new encoder for subsequent work
        encoder = device.createCommandEncoder();
      }

      // Capture intermediate features at specified layers
      if (VIT_CONFIG.intermediateLayers.includes(l)) {
        // Copy current token state for feature extraction
        const snapBuf = createEmptyBuffer(device, T * 4, GPUBufferUsage.COPY_DST);
        encoder.copyBufferToBuffer(currentTokens, 0, snapBuf, 0, T * 4);
        intermediateFeatures.push({ buffer: snapBuf, layerIdx: l });
      }
    }

    // --- Project and sum intermediate features ---
    // Each intermediate feature gets a 1x1 conv projection, then all are summed
    const featureBuf = createEmptyBuffer(device, D * numPatches * 4);
    let sumBuf = null;

    for (let i = 0; i < intermediateFeatures.length; i++) {
      const { buffer: snapBuf } = intermediateFeatures[i];

      // Upstream flow:
      //   feat [N, D] → skip CLS → [numPatches, D] → permute → [D, numPatches]
      //   → unflatten → [D, tokenH, tokenW] → 1x1 conv → [D, tokenH, tokenW]
      //
      // Step 1: Linear projection on [numPatches, D] → [numPatches, D]
      const projBuf = createEmptyBuffer(device, D * numPatches * 4);
      this._encodeOutputProjection(encoder, snapBuf, projBuf, weights, i, N, numPatches);

      // Step 2: Transpose [numPatches, D] → [D, numPatches] (= [D, tokenH, tokenW] in CHW)
      const transposedBuf = createEmptyBuffer(device, D * numPatches * 4);
      this._encodeTranspose(encoder, projBuf, transposedBuf, numPatches, D);

      if (sumBuf === null) {
        sumBuf = transposedBuf;
      } else {
        this._encodeAdd(encoder, sumBuf, transposedBuf, D * numPatches);
      }
    }

    return {
      featureBuf: sumBuf || featureBuf,
      clsTokenBuf: currentTokens, // CLS is at position 0 in the final token buffer
      tokenH,
      tokenW,
    };
  }

  // --- Private dispatch methods ---

  _encodePatchEmbed(encoder, imageBuf, weights, outputBuf, tokenH, tokenW) {
    const device = this.device;
    const D = VIT_CONFIG.dim;
    const ps = VIT_CONFIG.patchSize;
    const numTokens = tokenH * tokenW + 1;
    const totalWG = ceilDiv(numTokens * D, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new Uint32Array([tokenH * ps, tokenW * ps, ps, tokenH, tokenW, 3, D, numTokens, wgX]);
    const paramsBuf = makeUniform(device, paramsData);

    const bg = device.createBindGroup({
      layout: this.pipelines.patchEmbed.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: imageBuf } },
        { binding: 2, resource: { buffer: weights.encoder.patchEmbed.weight } },
        { binding: 3, resource: { buffer: weights.encoder.patchEmbed.bias } },
        { binding: 4, resource: { buffer: weights.encoder.clsToken } },
        { binding: 5, resource: { buffer: weights.encoder.posEmbed } },
        { binding: 6, resource: { buffer: outputBuf } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.patchEmbed);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeLayerNorm(enc, input, output, weights, prefix, N) {
    const device = this.device;
    const D = VIT_CONFIG.dim;

    const paramsData = new ArrayBuffer(16);
    const v = new DataView(paramsData);
    v.setUint32(0, N, true);
    v.setUint32(4, D, true);
    v.setFloat32(8, VIT_CONFIG.eps, true);
    const paramsBuf = makeUniform(device, new Uint8Array(paramsData));

    // Get weight/bias buffers by name
    const gammaKey = `${prefix}.weight`;
    const betaKey = `${prefix}.bias`;
    const gamma = weights.encoder.blockWeights?.[gammaKey] || weights.encoder[gammaKey];
    const beta = weights.encoder.blockWeights?.[betaKey] || weights.encoder[betaKey];

    if (!gamma || !beta) {
      console.warn(`Missing LayerNorm weights: ${prefix}`);
      return;
    }

    const bg = device.createBindGroup({
      layout: this.pipelines.layerNorm.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: gamma } },
        { binding: 3, resource: { buffer: beta } },
        { binding: 4, resource: { buffer: output } },
      ],
    });

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.layerNorm);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(N); // one workgroup per token
    pass.end();
  }

  _encodeQKV(enc, input, qBuf, kBuf, vBuf, weights, layerIdx, N) {
    const D = VIT_CONFIG.dim;
    // QKV is stored as a single [3*D, D] weight matrix
    // We do three separate linear projections using buffer offsets
    const prefix = `encoder.backbone.blocks.${layerIdx}.attn.qkv`;
    const qkvWeight = weights.encoder.blockWeights?.[`${prefix}.weight`];
    const qkvBias = weights.encoder.blockWeights?.[`${prefix}.bias`];

    if (!qkvWeight || !qkvBias) {
      console.warn(`Missing QKV weights for layer ${layerIdx}`);
      return;
    }

    const wSize = D * D * 4; // bytes for one D×D weight matrix
    const bSize = D * 4;

    // Q
    this._encodeLinearWithOffsets(enc, input, qkvWeight, 0, wSize, qkvBias, 0, bSize, qBuf, N, D, D);
    // K
    this._encodeLinearWithOffsets(enc, input, qkvWeight, wSize, wSize, qkvBias, bSize, bSize, kBuf, N, D, D);
    // V
    this._encodeLinearWithOffsets(enc, input, qkvWeight, 2 * wSize, wSize, qkvBias, 2 * bSize, bSize, vBuf, N, D, D);
  }

  _encodeLinear(enc, input, output, weights, prefix, numRows, inDim, outDim) {
    const device = this.device;
    const totalWG = ceilDiv(numRows * outDim, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new Uint32Array([numRows, inDim, outDim, wgX]);
    const paramsBuf = makeUniform(device, paramsData);

    const weight = weights.encoder.blockWeights?.[`${prefix}.weight`];
    const bias = weights.encoder.blockWeights?.[`${prefix}.bias`];

    if (!weight || !bias) {
      console.warn(`Missing linear weights: ${prefix}`);
      return;
    }

    const bg = device.createBindGroup({
      layout: this.pipelines.linear.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: weight } },
        { binding: 3, resource: { buffer: bias } },
        { binding: 4, resource: { buffer: output } },
      ],
    });

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.linear);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeLinearWithOffsets(enc, input, weight, wOffset, wSize, bias, bOffset, bSize, output, numRows, inDim, outDim) {
    const device = this.device;
    const totalWG = ceilDiv(numRows * outDim, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new Uint32Array([numRows, inDim, outDim, wgX]);
    const paramsBuf = makeUniform(device, paramsData);

    const bg = device.createBindGroup({
      layout: this.pipelines.linear.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: weight, offset: wOffset, size: wSize } },
        { binding: 3, resource: { buffer: bias, offset: bOffset, size: bSize } },
        { binding: 4, resource: { buffer: output } },
      ],
    });

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.linear);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeAttnScores(enc, qBuf, kBuf, scoreBuf, N) {
    const device = this.device;
    const { numHeads, dim, headDim, scale } = VIT_CONFIG;
    const total = numHeads * N * N;
    const totalWG = ceilDiv(total, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new ArrayBuffer(24);
    const v = new DataView(paramsData);
    v.setUint32(0, N, true);
    v.setUint32(4, dim, true);
    v.setUint32(8, numHeads, true);
    v.setUint32(12, headDim, true);
    v.setFloat32(16, scale, true);
    v.setUint32(20, wgX, true);
    const paramsBuf = makeUniform(device, new Uint8Array(paramsData));

    const bg = device.createBindGroup({
      layout: this.pipelines.attnScores.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: qBuf } },
        { binding: 2, resource: { buffer: kBuf } },
        { binding: 3, resource: { buffer: scoreBuf } },
      ],
    });

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.attnScores);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeAttnSoftmax(enc, scoreBuf, N) {
    const device = this.device;
    const totalRows = VIT_CONFIG.numHeads * N;
    const totalWG = ceilDiv(totalRows, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new Uint32Array([N, VIT_CONFIG.numHeads, wgX]);
    const paramsBuf = makeUniform(device, paramsData);

    const bg = device.createBindGroup({
      layout: this.pipelines.attnSoftmax.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: scoreBuf } },
      ],
    });

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.attnSoftmax);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeAttnApply(enc, scoreBuf, vBuf, output, N) {
    const device = this.device;
    const D = VIT_CONFIG.dim;
    const totalWG = ceilDiv(N * D, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new Uint32Array([N, D, VIT_CONFIG.numHeads, VIT_CONFIG.headDim, wgX]);
    const paramsBuf = makeUniform(device, paramsData);

    const bg = device.createBindGroup({
      layout: this.pipelines.attnApply.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: scoreBuf } },
        { binding: 2, resource: { buffer: vBuf } },
        { binding: 3, resource: { buffer: output } },
      ],
    });

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.attnApply);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeLayerScaleResidual(enc, input, residual, output, weights, prefix, count, D) {
    const device = this.device;
    const totalWG = ceilDiv(count, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new Uint32Array([count, D, wgX]);
    const paramsBuf = makeUniform(device, paramsData);

    const gamma = weights.encoder.blockWeights?.[`${prefix}.gamma`];
    if (!gamma) {
      console.warn(`Missing LayerScale gamma: ${prefix}`);
      return;
    }

    const bg = device.createBindGroup({
      layout: this.pipelines.layerScale.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: gamma } },
        { binding: 3, resource: { buffer: residual } },
        { binding: 4, resource: { buffer: output } },
      ],
    });

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.layerScale);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeLinearGelu(enc, input, output, weights, prefix, numRows, inDim, outDim) {
    const device = this.device;
    const totalWG = ceilDiv(numRows * outDim, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new Uint32Array([numRows, inDim, outDim, wgX]);
    const paramsBuf = makeUniform(device, paramsData);

    const weight = weights.encoder.blockWeights?.[`${prefix}.weight`];
    const bias = weights.encoder.blockWeights?.[`${prefix}.bias`];

    if (!weight || !bias) {
      console.warn(`Missing linear+GELU weights: ${prefix}`);
      return;
    }

    const bg = device.createBindGroup({
      layout: this.pipelines.linearGelu.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: weight } },
        { binding: 3, resource: { buffer: bias } },
        { binding: 4, resource: { buffer: output } },
      ],
    });

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.linearGelu);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeOutputProjection(enc, tokensBuf, outputBuf, weights, projIdx, N, numPatches) {
    // Extract patch tokens (skip CLS), reshape to CHW, and project with 1x1 conv
    // This is a simplified version — the upstream code does permute + unflatten + conv
    // For now, we treat it as a linear: [numPatches, D] → [numPatches, D]
    // using the output_projections weight
    const device = this.device;
    const D = VIT_CONFIG.dim;
    const totalWG = ceilDiv(numPatches * D, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new Uint32Array([numPatches, D, D, wgX]);
    const paramsBuf = makeUniform(device, paramsData);

    const proj = weights.encoder.outputProjections[projIdx];

    const bg = device.createBindGroup({
      layout: this.pipelines.linear.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        // Skip CLS token (offset by D*4 bytes)
        { binding: 1, resource: { buffer: tokensBuf, offset: D * 4, size: numPatches * D * 4 } },
        { binding: 2, resource: { buffer: proj.weight } },
        { binding: 3, resource: { buffer: proj.bias } },
        { binding: 4, resource: { buffer: outputBuf } },
      ],
    });

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.linear);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeTranspose(enc, input, output, rows, cols) {
    const device = this.device;
    const total = rows * cols;
    const totalWG = ceilDiv(total, 256);
    const [wgX, wgY] = splitWG(totalWG);

    const paramsData = new Uint32Array([rows, cols, wgX]);
    const paramsBuf = makeUniform(device, paramsData);

    const bg = device.createBindGroup({
      layout: this.pipelines.transpose.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: output } },
      ],
    });

    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.transpose);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  _encodeAdd(enc, dst, src, count) {
    // Simple element-wise add using the activation shader
    // We reuse layerScale with gamma=1 and dst as residual... or just inline
    // For simplicity, use a copy + add pattern
    const device = this.device;
    const totalWG = ceilDiv(count, 256);
    const [wgX, wgY] = splitWG(totalWG);

    // Use layerScale with gamma=all ones? No, that creates a dependency.
    // Just dispatch activation add (op=2)
    // Import not available here, so let's create a simple add inline
    const addModule = device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read_write> dst: array<f32>;
        @group(0) @binding(1) var<storage, read> src: array<f32>;
        struct P { count: u32, numWgX: u32 }
        @group(0) @binding(2) var<uniform> p: P;

        @compute @workgroup_size(256)
        fn main(@builtin(workgroup_id) wgid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
          let idx = (wgid.x + wgid.y * p.numWgX) * 256u + lid.x;
          if (idx >= p.count) { return; }
          dst[idx] = dst[idx] + src[idx];
        }
      `,
    });
    const addPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: addModule, entryPoint: 'main' },
    });

    const paramsBuf = makeUniform(device, new Uint32Array([count, wgX]));

    const bg = device.createBindGroup({
      layout: addPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dst } },
        { binding: 1, resource: { buffer: src } },
        { binding: 2, resource: { buffer: paramsBuf } },
      ],
    });

    const pass = enc.beginComputePass();
    pass.setPipeline(addPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }
}

export { VIT_CONFIG };
