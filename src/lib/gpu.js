/**
 * WebGPU initialization and device management.
 *
 * Device acquisition and backend identity delegate to the kit's shared
 * gpu-environment helpers (adopted 2026-09-15); buffer utilities stay local.
 */
import {
  createWebGpuBackendIdentity,
  createWebGpuDeviceRequest,
  requestBrowserWebGpuDevice,
} from '@kaminos/webgpu-inference-kit';

// Kept for embedding hosts that merge inference limits into a shared-device
// request; mirrors the kit's copied limit-key set.
export const INFERENCE_LIMIT_KEYS = [
  'maxBufferSize',
  'maxStorageBufferBindingSize',
  'maxComputeWorkgroupStorageSize',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY',
];

export function inferenceLimits(limits) {
  const out = {};
  for (const key of INFERENCE_LIMIT_KEYS) {
    if (Number.isFinite(limits?.[key])) out[key] = limits[key];
  }
  return out;
}

function adapterName(adapter) {
  const info = adapter?.info || {};
  return info.description
    || [info.vendor, info.architecture, info.device].filter(Boolean).join(' ')
    || 'unknown-webgpu-adapter';
}

/**
 * Device request via the kit's shared GPU-environment helper: identical
 * limit-copying and timestamp-query preference semantics for every kit port.
 */
export function buildMogeDeviceRequest(adapter, options = {}) {
  return createWebGpuDeviceRequest(adapter, options);
}

/**
 * Backend identity for a HOST-OWNED (borrowed/shared) device — the embedding
 * path, where the host application created the GPUDevice (possibly shared
 * with a renderer) and MoGe must still report a kit-valid identity.
 */
export function borrowedDeviceBackendIdentity({ adapter, device, browser, requestedFeatures } = {}) {
  const requested = requestedFeatures
    ?? (adapter?.features?.has?.('timestamp-query') ? ['timestamp-query'] : []);
  return createWebGpuBackendIdentity({
    adapterName: adapterName(adapter),
    browser: browser ?? globalThis.navigator?.userAgent ?? null,
    requestedFeatures: requested,
    effectiveFeatures: device?.features || requested,
    limits: device?.limits || adapter?.limits || {},
    timestampQuery: requested.includes('timestamp-query') ? 'requested' : 'unavailable',
  });
}

export async function initGPU() {
  if (!navigator.gpu) {
    throw new Error('WebGPU is not supported in this browser. Try Chrome 113+ or Edge 113+.');
  }

  // Owned-device acquisition through the kit's shared helper: adapter request,
  // device request (limit copy + timestamp preference), and backend identity
  // all come from @kaminos/webgpu-inference-kit gpu-environment.
  const { adapter, device, backendIdentity } = await requestBrowserWebGpuDevice(navigator.gpu, {
    adapterOptions: { powerPreference: 'high-performance' },
    label: 'moge-webgpu-inference',
  });

  device.lost.then((info) => {
    console.error('WebGPU device lost:', info.message);
    if (info.reason !== 'destroyed') {
      // Could attempt recovery here
    }
  });

  return { adapter, device, backendIdentity };
}

/**
 * Create a storage buffer initialized with data.
 */
export function createStorageBuffer(device, data, usage = 0) {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | usage,
    mappedAtCreation: true,
  });
  new (data.constructor)(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

/**
 * Create an empty storage buffer.
 */
export function createEmptyBuffer(device, size, usage = 0) {
  return device.createBuffer({
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | usage,
    mappedAtCreation: false,
  });
}

/**
 * Read back buffer contents to CPU.
 */
export async function readBuffer(device, buffer, size) {
  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return result;
}
