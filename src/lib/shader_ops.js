/**
 * shader_ops.js — WebGPU compute dispatch wrappers for each shader.
 *
 * Each function creates a pipeline, binds buffers, and dispatches.
 * Pipelines are cached by device for reuse.
 */

import conv2dWGSL from '../shaders/conv2d.wgsl?raw';
import conv1x1WGSL from '../shaders/conv1x1.wgsl?raw';
import activationsWGSL from '../shaders/activations.wgsl?raw';
import groupnormWGSL from '../shaders/groupnorm.wgsl?raw';
import pixelshuffleWGSL from '../shaders/pixelshuffle.wgsl?raw';
import upsampleWGSL from '../shaders/upsample.wgsl?raw';

import { createStorageBuffer, createEmptyBuffer } from './gpu.js';

const pipelineCache = new Map();

function getOrCreatePipeline(device, key, code, entryPoint) {
  if (pipelineCache.has(key)) return pipelineCache.get(key);
  const module = device.createShaderModule({ code });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint },
  });
  pipelineCache.set(key, pipeline);
  return pipeline;
}

function ceil(a, b) { return Math.ceil(a / b); }

/**
 * Dispatch conv2d (3x3 or arbitrary kernel).
 * Returns output buffer [outC, outH, outW].
 */
export function dispatchConv2d(device, encoder, inputBuf, weightBuf, biasBuf, params) {
  const { inC, inH, inW, outC, kH, kW, padH, padW, strideH, strideW } = params;
  const outH = Math.floor((inH + 2 * padH - kH) / strideH) + 1;
  const outW = Math.floor((inW + 2 * padW - kW) / strideW) + 1;
  const hasBias = biasBuf ? 1 : 0;

  const pipeline = getOrCreatePipeline(device, 'conv2d', conv2dWGSL, 'conv2d_main');

  const uniformData = new Uint32Array([inC, inH, inW, outC, outH, outW, kH, kW, padH, padW, strideH, strideW, hasBias]);
  const uniformBuf = createStorageBuffer(device, uniformData, GPUBufferUsage.UNIFORM);
  // Override usage for uniform
  const uniformBuf2 = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint32Array(uniformBuf2.getMappedRange()).set(uniformData);
  uniformBuf2.unmap();
  uniformBuf.destroy();

  const dummyBias = biasBuf || createStorageBuffer(device, new Float32Array([0]));
  const outputBuf = createEmptyBuffer(device, outC * outH * outW * 4);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf2 } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: weightBuf } },
      { binding: 3, resource: { buffer: dummyBias } },
      { binding: 4, resource: { buffer: outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(ceil(outW, 16), ceil(outH, 16), outC);
  pass.end();

  return { buffer: outputBuf, outC, outH, outW };
}

/**
 * Dispatch 1x1 conv.
 */
export function dispatchConv1x1(device, encoder, inputBuf, weightBuf, biasBuf, params) {
  const { inC, outC, H, W } = params;
  const hasBias = biasBuf ? 1 : 0;

  const pipeline = getOrCreatePipeline(device, 'conv1x1', conv1x1WGSL, 'conv1x1_main');

  const uniformData = new Uint32Array([inC, outC, H, W, hasBias]);
  const uniformBuf = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint32Array(uniformBuf.getMappedRange()).set(uniformData);
  uniformBuf.unmap();

  const dummyBias = biasBuf || createStorageBuffer(device, new Float32Array([0]));
  const outputBuf = createEmptyBuffer(device, outC * H * W * 4);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: weightBuf } },
      { binding: 3, resource: { buffer: dummyBias } },
      { binding: 4, resource: { buffer: outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(ceil(outC * H * W, 256));
  pass.end();

  return { buffer: outputBuf, C: outC, H, W };
}

/**
 * Dispatch element-wise activation.
 * op: 0=relu, 1=silu, 2=add, 3=add_relu, 4=sigmoid
 */
export function dispatchActivation(device, encoder, inputA, inputB, count, op) {
  const pipeline = getOrCreatePipeline(device, 'activation', activationsWGSL, 'activation_main');

  const uniformData = new Uint32Array([count, op]);
  const uniformBuf = device.createBuffer({
    size: 8,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint32Array(uniformBuf.getMappedRange()).set(uniformData);
  uniformBuf.unmap();

  const dummyB = inputB || createStorageBuffer(device, new Float32Array([0]));
  const outputBuf = createEmptyBuffer(device, count * 4);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputA } },
      { binding: 2, resource: { buffer: dummyB } },
      { binding: 3, resource: { buffer: outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(ceil(count, 256));
  pass.end();

  return outputBuf;
}

/**
 * Dispatch GroupNorm (two-pass: stats then normalize).
 */
export function dispatchGroupNorm(device, encoder, inputBuf, scaleBuf, biasBuf, params) {
  const { C, H, W, numGroups, eps = 1e-5 } = params;

  const statsPipeline = getOrCreatePipeline(device, 'gn_stats', groupnormWGSL, 'groupnorm_stats');
  const normPipeline = getOrCreatePipeline(device, 'gn_norm', groupnormWGSL, 'groupnorm_normalize');

  // Uniform: C, H, W, numGroups, eps (f32)
  const uniformArr = new ArrayBuffer(20);
  const u32View = new Uint32Array(uniformArr, 0, 4);
  const f32View = new Float32Array(uniformArr, 16, 1);
  u32View.set([C, H, W, numGroups]);
  f32View[0] = eps;

  const uniformBuf = device.createBuffer({
    size: 20,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(uniformBuf.getMappedRange()).set(new Uint8Array(uniformArr));
  uniformBuf.unmap();

  const statsBuf = createEmptyBuffer(device, numGroups * 2 * 4);
  const outputBuf = createEmptyBuffer(device, C * H * W * 4);

  // Pass 1: compute stats
  const statsBindGroup = device.createBindGroup({
    layout: statsPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: scaleBuf } },
      { binding: 3, resource: { buffer: biasBuf } },
      { binding: 4, resource: { buffer: outputBuf } },
      { binding: 5, resource: { buffer: statsBuf } },
    ],
  });

  const pass1 = encoder.beginComputePass();
  pass1.setPipeline(statsPipeline);
  pass1.setBindGroup(0, statsBindGroup);
  pass1.dispatchWorkgroups(ceil(numGroups, 256));
  pass1.end();

  // Pass 2: normalize
  const normBindGroup = device.createBindGroup({
    layout: normPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: scaleBuf } },
      { binding: 3, resource: { buffer: biasBuf } },
      { binding: 4, resource: { buffer: outputBuf } },
      { binding: 5, resource: { buffer: statsBuf } },
    ],
  });

  const pass2 = encoder.beginComputePass();
  pass2.setPipeline(normPipeline);
  pass2.setBindGroup(0, normBindGroup);
  pass2.dispatchWorkgroups(ceil(C * H * W, 256));
  pass2.end();

  return outputBuf;
}

/**
 * Dispatch PixelShuffle.
 */
export function dispatchPixelShuffle(device, encoder, inputBuf, params) {
  const { inC, inH, inW, scaleFactor } = params;
  const outC = inC / (scaleFactor * scaleFactor);
  const outH = inH * scaleFactor;
  const outW = inW * scaleFactor;

  const pipeline = getOrCreatePipeline(device, 'pixelshuffle', pixelshuffleWGSL, 'pixelshuffle_main');

  const uniformData = new Uint32Array([inC, inH, inW, outC, scaleFactor]);
  const uniformBuf = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint32Array(uniformBuf.getMappedRange()).set(uniformData);
  uniformBuf.unmap();

  const outputBuf = createEmptyBuffer(device, outC * outH * outW * 4);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(ceil(outC * outH * outW, 256));
  pass.end();

  return { buffer: outputBuf, C: outC, H: outH, W: outW };
}

/**
 * Dispatch bilinear/nearest upsample.
 */
export function dispatchUpsample(device, encoder, inputBuf, params) {
  const { C, inH, inW, outH, outW, mode = 1 } = params; // mode: 0=nearest, 1=bilinear

  const pipeline = getOrCreatePipeline(device, 'upsample', upsampleWGSL, 'upsample_main');

  const uniformData = new Uint32Array([C, inH, inW, outH, outW, mode]);
  const uniformBuf = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint32Array(uniformBuf.getMappedRange()).set(uniformData);
  uniformBuf.unmap();

  const outputBuf = createEmptyBuffer(device, C * outH * outW * 4);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: inputBuf } },
      { binding: 2, resource: { buffer: outputBuf } },
    ],
  });

  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(ceil(C * outH * outW, 256));
  pass.end();

  return { buffer: outputBuf, C, H: outH, W: outW };
}
