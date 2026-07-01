#!/usr/bin/env node
/**
 * Pure contract checks for MoGE's kit-compatible route request/result envelopes.
 */

import assert from 'node:assert/strict';

import { WEBGPU_INFERENCE_KIT_VERSION } from '@kaminos/webgpu-inference-kit';

import {
  MOGE_DEPTH_NORMAL_ROUTE_ID,
  createMogeRouteInvocationRequest,
  createMogeRouteWorkerResult,
  validateMogeRouteWorkerResult,
} from '../src/lib/route_boundary.js';

const routeReceipt = {
  sourceArtifact: {
    artifactId: 'image:test-fixture-input',
    sha256: 'sha256:test-fixture-input',
    shape: [518, 518, 4],
  },
  outputs: {
    depth: { artifactId: 'depth:test-fixture-input', sha256: 'sha256:depth-output' },
    normal: { artifactId: 'normal:test-fixture-input', sha256: 'sha256:normal-output' },
    pointMap: { artifactId: 'pointmap:test-fixture-input', sha256: 'sha256:pointmap-output' },
  },
  model: {
    revision: 'local-moge-2-vitl-normal',
    weightsHash: 'sha256:weights-fixture',
    dtype: 'fp16',
  },
  kernel: {
    profile: 'conv-transpose2d-stride2',
    commit: '15d2dea',
  },
};

const request = createMogeRouteInvocationRequest({ routeReceipt, outH: 592, outW: 592 });
assert.equal(request.schema, 'kaminos.webgpu-route-request.v0');
assert.equal(request.routeId, MOGE_DEPTH_NORMAL_ROUTE_ID);
assert.equal(request.inputs[0].sha256, 'sha256:test-fixture-input');
assert.deepEqual(request.outputs.map(output => output.role), ['depth', 'normal', 'pointmap']);

const receipt = {
  schema: 'kaminos.webgpu-route-receipt.v0',
  requestedRouteId: MOGE_DEPTH_NORMAL_ROUTE_ID,
  effectiveRouteId: MOGE_DEPTH_NORMAL_ROUTE_ID,
  status: 'real',
  fallbackReason: null,
  backend: {
    kind: 'webgpu-local',
    runtime: 'browser',
    adapterName: 'Apple M4 Max',
    features: ['timestamp-query'],
    requestedFeatures: ['timestamp-query'],
    limits: { maxBufferSize: 4294967296 },
    timestampQuery: 'requested',
  },
  model: {
    id: 'Ruicheng/moge-2-vitl-normal',
    revision: 'local-moge-2-vitl-normal',
    weightsHash: 'sha256:weights-fixture',
    dtype: 'fp16',
  },
  kernel: {
    kitVersion: WEBGPU_INFERENCE_KIT_VERSION,
    profile: 'conv-transpose2d-stride2',
    commit: '15d2dea',
  },
  inputs: request.inputs,
  outputs: [
    { role: 'depth', artifactId: 'depth:test-fixture-input', sha256: 'sha256:depth-output', shape: [592, 592], status: 'real' },
    { role: 'normal', artifactId: 'normal:test-fixture-input', sha256: 'sha256:normal-output', shape: [3, 592, 592], status: 'real' },
    { role: 'pointmap', artifactId: 'pointmap:test-fixture-input', sha256: 'sha256:pointmap-output', shape: [3, 592, 592], status: 'real' },
  ],
  timings: {
    source: 'queue-submit-wait',
    totalMs: 1853.4,
    stages: [
      { name: 'backbone', ms: 997.6 },
      { name: 'decoder-heads', ms: 854.3 },
      { name: 'output-readback', ms: 1.9 },
    ],
  },
};

const result = createMogeRouteWorkerResult({ request, receipt });
assert.equal(result.schema, 'kaminos.webgpu-route-result.v0');
assert.equal(result.authoritative, true);
assert.equal(validateMogeRouteWorkerResult(result).ok, true);
assert.match(
  validateMogeRouteWorkerResult({
    ...result,
    receipt: { ...receipt, kernel: { ...receipt.kernel, kitVersion: '' } },
  }).errors.join('\n'),
  /kernel.kitVersion/,
);

const partialReceipt = {
  ...receipt,
  status: 'partial',
  inputs: [{ ...receipt.inputs[0], sha256: null, hashStatus: 'not-hashed-browser-runtime' }],
  outputs: [{ ...receipt.outputs[0], sha256: null, hashStatus: 'not-hashed-browser-runtime', status: 'partial' }],
};
const partial = createMogeRouteWorkerResult({ request, receipt: partialReceipt });
assert.equal(partial.authoritative, false);
assert.equal(partial.validation.ok, false);
assert.match(partial.validation.errors.join('\n'), /sha256|status/);

const routeMismatch = createMogeRouteWorkerResult({
  request,
  receipt: { ...receipt, effectiveRouteId: 'fixture.moge.depth-normal.v0' },
});
assert.equal(routeMismatch.authoritative, false);
assert.equal(validateMogeRouteWorkerResult(routeMismatch).ok, false);
assert.match(routeMismatch.validation.errors.join('\n'), /effectiveRouteId/);

const wallClockOnly = createMogeRouteWorkerResult({
  request,
  receipt: {
    ...receipt,
    timings: {
      source: 'wall-clock',
      totalMs: 1853.4,
      stages: [{ name: 'total', ms: 1853.4 }],
    },
  },
});
assert.equal(wallClockOnly.authoritative, false);
assert.equal(validateMogeRouteWorkerResult(wallClockOnly).ok, false);
assert.match(wallClockOnly.validation.errors.join('\n'), /queue-submit-wait|staged/i);

console.log('route boundary contracts passed');
