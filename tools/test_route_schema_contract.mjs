#!/usr/bin/env node
/**
 * Compare MoGE's local route-boundary schema mirror against the kit contract.
 */

import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  createMogeRouteSchemaContract,
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
assert.deepEqual(mogeContract.authoritativeReceiptStatuses, kitContract.authoritativeReceiptStatuses);
assert.deepEqual(mogeContract.nonAuthoritativeReceiptStatuses, kitContract.nonAuthoritativeReceiptStatuses);
assert.equal(mogeContract.routes.mogeDepthNormal.routeId, 'moge.depth-normal.webgpu-local.v0');
assert.deepEqual(mogeContract.routes.mogeDepthNormal.requiredOutputRoles, ['depth', 'normal', 'pointmap']);
assert.deepEqual(mogeContract.routes.mogeDepthNormal.authoritativeTimingStages, ['backbone', 'decoder-heads', 'output-readback']);

console.log('route schema contract passed');
