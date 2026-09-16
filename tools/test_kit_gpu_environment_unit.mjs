#!/usr/bin/env node
/**
 * Pure-Node unit tests: MoGe's device initialization consumes the kit's
 * shared GPU-environment helpers rather than reimplementing them.
 *
 * Pins the adoption contract:
 *   1. buildMogeDeviceRequest delegates to the kit's createWebGpuDeviceRequest
 *      (limit copying, timestamp-query preference semantics).
 *   2. borrowedDeviceBackendIdentity produces a kit-valid backend identity for
 *      host-owned (shared) devices, validated by the kit's own validator.
 *   3. Timestamp-unavailable adapters degrade identically to the kit contract.
 */
import assert from 'node:assert/strict';
import {
  createWebGpuDeviceRequest,
  validateWebGpuBackendIdentity,
} from '@kaminos/webgpu-inference-kit';
import { buildMogeDeviceRequest, borrowedDeviceBackendIdentity } from '../src/lib/gpu.js';

const LIMITS = {
  maxBufferSize: 4294967296,
  maxStorageBufferBindingSize: 4294967292,
  maxComputeWorkgroupStorageSize: 32768,
  maxComputeInvocationsPerWorkgroup: 1024,
  maxComputeWorkgroupSizeX: 1024,
  maxComputeWorkgroupSizeY: 1024,
  maxStorageBuffersPerShaderStage: 10, // not a kit-copied key: must be dropped
};

function mockAdapter({ timestamp = true } = {}) {
  return {
    features: new Set(timestamp ? ['timestamp-query'] : []),
    limits: LIMITS,
    info: { description: 'mock-apple-gpu' },
  };
}

// 1. Delegation: moge's request === kit's request for the same adapter.
{
  const adapter = mockAdapter();
  const ours = buildMogeDeviceRequest(adapter);
  const kits = createWebGpuDeviceRequest(adapter);
  assert.deepEqual(ours, kits, 'buildMogeDeviceRequest must match the kit helper output exactly');
  assert.ok(ours.requiredFeatures.includes('timestamp-query'));
  assert.equal(ours.requiredLimits.maxBufferSize, LIMITS.maxBufferSize);
  assert.equal(ours.requiredLimits.maxStorageBuffersPerShaderStage, undefined,
    'non-kit limit keys must not be silently requested');
}

// 2. Borrowed-device identity validates under the kit's validator.
{
  const adapter = mockAdapter();
  const device = { features: new Set(['timestamp-query']), limits: LIMITS };
  const identity = borrowedDeviceBackendIdentity({
    adapter, device, browser: 'test-agent', requestedFeatures: ['timestamp-query'],
  });
  const verdict = validateWebGpuBackendIdentity(identity);
  assert.ok(verdict.ok, `identity must validate: ${JSON.stringify(verdict.errors)}`);
  assert.equal(identity.kind, 'webgpu-local');
  assert.equal(identity.adapterName, 'mock-apple-gpu');
  assert.equal(identity.timestampQuery, 'requested');
}

// 3. No-timestamp adapter: degraded identically to kit semantics.
{
  const adapter = mockAdapter({ timestamp: false });
  const ours = buildMogeDeviceRequest(adapter);
  assert.deepEqual(ours.requiredFeatures, []);
  assert.equal(ours.timestampQuery, 'unavailable');
  const identity = borrowedDeviceBackendIdentity({
    adapter, device: { features: new Set(), limits: LIMITS }, browser: 'test-agent',
  });
  const verdict = validateWebGpuBackendIdentity(identity);
  assert.ok(verdict.ok, `no-timestamp identity must validate: ${JSON.stringify(verdict.errors)}`);
  assert.equal(identity.timestampQuery, 'unavailable');
}

// Adapter support does not establish what a borrowed device enabled.
{
  const identity = borrowedDeviceBackendIdentity({
    adapter: mockAdapter(), device: { features: new Set(), limits: LIMITS },
  });
  assert.deepEqual(identity.features, []);
  assert.deepEqual(identity.requestedFeatures, []);
  assert.equal(identity.timestampQuery, 'disabled');
  assert.ok(validateWebGpuBackendIdentity(identity).ok);
}
{
  const identity = borrowedDeviceBackendIdentity({
    adapter: mockAdapter(), device: { features: new Set(['timestamp-query']), limits: LIMITS },
  });
  assert.deepEqual(identity.requestedFeatures, [], 'host request provenance must not be invented');
  assert.equal(identity.timestampQuery, 'available');
  assert.ok(validateWebGpuBackendIdentity(identity).ok);
}
assert.throws(() => borrowedDeviceBackendIdentity({ adapter: mockAdapter(), device: { limits: LIMITS } }),
  /device.*features/i);
assert.throws(() => borrowedDeviceBackendIdentity({ adapter: mockAdapter(), device: { features: new Set() } }),
  /device.*limits/i);
assert.throws(() => borrowedDeviceBackendIdentity({
  adapter: mockAdapter(), device: { features: new Set(), limits: LIMITS }, requestedFeatures: ['timestamp-query'],
}), /requested.*enabled/i);

console.log('kit gpu-environment adoption contracts: PASS');
