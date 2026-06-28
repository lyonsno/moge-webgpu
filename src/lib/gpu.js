/**
 * WebGPU initialization and device management.
 */

const INFERENCE_LIMIT_KEYS = [
  'maxBufferSize',
  'maxStorageBufferBindingSize',
  'maxComputeWorkgroupStorageSize',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY',
];

function featureList(features) {
  if (!features) return [];
  return Array.from(features).map(String).sort();
}

function inferenceLimits(limits) {
  const out = {};
  for (const key of INFERENCE_LIMIT_KEYS) {
    if (Number.isFinite(limits?.[key])) out[key] = limits[key];
  }
  return out;
}

function adapterName(adapter) {
  const info = adapter.info || {};
  return info.description
    || [info.vendor, info.architecture, info.device].filter(Boolean).join(' ')
    || 'unknown-webgpu-adapter';
}

export async function initGPU() {
  if (!navigator.gpu) {
    throw new Error('WebGPU is not supported in this browser. Try Chrome 113+ or Edge 113+.');
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  });
  if (!adapter) {
    throw new Error('No WebGPU adapter found. Your GPU may not support WebGPU.');
  }

  const requiredFeatures = [];
  if (adapter.features.has('timestamp-query')) {
    requiredFeatures.push('timestamp-query');
  }
  const requiredLimits = inferenceLimits(adapter.limits);

  // Request max limits for large model inference
  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits,
  });

  device.lost.then((info) => {
    console.error('WebGPU device lost:', info.message);
    if (info.reason !== 'destroyed') {
      // Could attempt recovery here
    }
  });

  const deviceFeatures = featureList(device.features || adapter.features);
  return {
    adapter,
    device,
    backendIdentity: {
      kind: 'webgpu-local',
      runtime: 'browser',
      adapterName: adapterName(adapter),
      browser: navigator.userAgent || 'unknown-browser',
      requestedFeatures: [...requiredFeatures],
      features: deviceFeatures,
      limits: requiredLimits,
      timestampQuery: requiredFeatures.includes('timestamp-query') ? 'requested' : 'unavailable',
    },
  };
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
