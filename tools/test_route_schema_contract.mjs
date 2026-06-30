#!/usr/bin/env node
/**
 * Compare MoGE's local route-boundary schema mirror against the kit contract.
 */

import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  createMogeRouteInvocationRequest,
  createMogeRouteSchemaContract,
  createMogeRouteWorkerResult,
} from '../src/lib/route_boundary.js';

const candidateKitEntries = [
  process.env.KAMINOS_WEBGPU_INFERENCE_KIT_SRC,
  '../kaminos/webgpu-inference-kit/src/index.js',
  '/private/tmp/kaminos-cranial-webgpu-inference-kit-0628/webgpu-inference-kit/src/index.js',
].filter(Boolean);

function resolveKitEntry() {
  for (const candidate of candidateKitEntries) {
    const resolved = resolve(candidate);
    if (existsSync(resolved)) return resolved;
  }
  throw new Error(`Unable to locate @kaminos/webgpu-inference-kit src entry. Tried: ${candidateKitEntries.join(', ')}`);
}

const kit = await import(pathToFileURL(resolveKitEntry()).href);
const kitContract = kit.createWebGpuRouteSchemaContract();
const mogeContract = createMogeRouteSchemaContract();

assert.equal(mogeContract.schema, kitContract.schema);
assert.equal(mogeContract.definitionSchema, kitContract.definitionSchema);
assert.equal(mogeContract.requestSchema, kitContract.requestSchema);
assert.equal(mogeContract.resultSchema, kitContract.resultSchema);
assert.equal(mogeContract.receiptSchema, kitContract.receiptSchema);
assert.equal(mogeContract.runtimeProfileSchema, kitContract.runtimeProfileSchema);
assert.equal(mogeContract.evidenceClassificationSchema, kitContract.evidenceClassificationSchema);
assert.equal(mogeContract.schedulerSchema, kitContract.schedulerSchema);
assert.equal(mogeContract.backpressureSchema, kitContract.backpressureSchema);
assert.deepEqual(mogeContract.authoritativeReceiptStatuses, kitContract.authoritativeReceiptStatuses);
assert.deepEqual(mogeContract.nonAuthoritativeReceiptStatuses, kitContract.nonAuthoritativeReceiptStatuses);
assert.equal(mogeContract.routes.mogeDepthNormal.routeId, 'moge.depth-normal.webgpu-local.v0');
assert.deepEqual(mogeContract.routes.mogeDepthNormal.requiredOutputRoles, ['depth', 'normal', 'pointmap']);
assert.deepEqual(mogeContract.routes.mogeDepthNormal.authoritativeTimingStages, ['backbone', 'decoder-heads', 'output-readback']);

const request = createMogeRouteInvocationRequest({
  routeReceipt: {
    sourceArtifact: {
      artifactId: 'image:bunnycake',
      sha256: 'sha256:input',
      shape: [518, 518, 4],
    },
    outputs: {
      depth: { artifactId: 'depth:bunnycake', sha256: 'sha256:depth' },
      normal: { artifactId: 'normal:bunnycake', sha256: 'sha256:normal' },
      pointMap: { artifactId: 'pointmap:bunnycake', sha256: 'sha256:pointmap' },
    },
    model: {
      revision: 'local-vitl-normal',
      weightsHash: 'sha256:weights',
      dtype: 'fp16',
    },
    kernel: {
      profile: 'conv-transpose2d-stride2',
      commit: 'a1bf4d3',
    },
  },
  outH: 592,
  outW: 592,
  requestId: 'req:moge-conformance',
});

const receipt = {
  schema: mogeContract.receiptSchema,
  requestedRouteId: mogeContract.routes.mogeDepthNormal.routeId,
  effectiveRouteId: mogeContract.routes.mogeDepthNormal.routeId,
  status: 'real',
  fallbackReason: null,
  backend: {
    kind: 'webgpu-local',
    runtime: 'browser',
    adapterName: 'Apple M4 Max',
    browser: 'Chrome Headless',
    requestedFeatures: ['timestamp-query'],
    features: ['shader-f16', 'timestamp-query'],
    limits: {
      maxBufferSize: 4294967296,
      maxStorageBufferBindingSize: 2147483648,
    },
    timestampQuery: 'requested',
  },
  model: {
    id: 'Ruicheng/moge-2-vitl-normal',
    revision: 'local-vitl-normal',
    weightsHash: 'sha256:weights',
    dtype: 'fp16',
  },
  kernel: {
    kitVersion: '0.0.0',
    profile: 'conv-transpose2d-stride2',
    commit: 'a1bf4d3',
  },
  inputs: request.inputs,
  outputs: [
    { role: 'depth', artifactId: 'depth:bunnycake', sha256: 'sha256:depth', shape: [592, 592], status: 'real' },
    { role: 'normal', artifactId: 'normal:bunnycake', sha256: 'sha256:normal', shape: [3, 592, 592], status: 'real' },
    { role: 'pointmap', artifactId: 'pointmap:bunnycake', sha256: 'sha256:pointmap', shape: [3, 592, 592], status: 'real' },
  ],
  timings: {
    source: 'queue-submit-wait',
    totalMs: 1852,
    stages: [
      { name: 'backbone', ms: 1000 },
      { name: 'decoder-heads', ms: 850 },
      { name: 'output-readback', ms: 2 },
    ],
    profile: {
      schema: 'kaminos.webgpu-staged-profile.v0',
      route: 'staged-submits',
      timingSource: 'queue-submit-wait',
      requiredStages: ['backbone', 'decoder-heads', 'output-readback'],
      stages: [
        { name: 'backbone', ms: 1000 },
        { name: 'decoder-heads', ms: 850 },
        { name: 'output-readback', ms: 2 },
      ],
      stageNames: ['backbone', 'decoder-heads', 'output-readback'],
      totalMs: 1852,
    },
  },
  createdAt: new Date().toISOString(),
};

const workerResult = createMogeRouteWorkerResult({ request, receipt });
assert.equal(workerResult.authoritative, true);
const classification = kit.classifyWebGpuRouteWorkerResultEvidence(workerResult, {
  expectedRouteId: mogeContract.routes.mogeDepthNormal.routeId,
  now: receipt.createdAt,
});
assert.equal(classification.schema, mogeContract.evidenceClassificationSchema);
assert.equal(classification.classification, 'authoritative-live-webgpu');
assert.equal(classification.authoritative, true);

console.log('route schema contract passed');
