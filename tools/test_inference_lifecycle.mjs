import assert from 'node:assert/strict';
import { MoGeInference } from '../dist-lib/moge-inference.js';
import { createResourceScope } from '../src/lib/resource_scope.js';
import { bufferPoolFor } from '../src/lib/gpu.js';

const deferred = () => {
  let resolve;
  const promise = new Promise(r => { resolve = r; });
  return { promise, resolve };
};

// Actual init entry: pause the real weight URL lookup before GPU allocation.
const lookup = deferred();
let lookups = 0;
globalThis.fetch = () => { lookups++; return lookup.promise; };
globalThis.window = {};
const initializing = new MoGeInference({ device: { queue: { onSubmittedWorkDone: async () => {} } } });
initializing._createStubWeights = () => ({});
const firstInit = initializing.init();
const secondInit = initializing.init();
try {
  assert.equal(lookups, 1, 'overlapping init must share one weight-load operation');
} finally {
  lookup.resolve({ ok: false, headers: { get: () => 'text/html' } });
  await Promise.allSettled([firstInit, secondInit]);
}
console.log('PASS overlapping initialization uses one load');

function mockDevice() {
  const buffers = [];
  return {
    buffers,
    queue: { onSubmittedWorkDone: async () => {} },
    destroy() { assert.fail('must never destroy the borrowed device'); },
    createBuffer({ size, usage }) {
      assert.equal(this.buffers, buffers, 'native method receiver must be the host device');
      const buffer = { size, usage, destroyed: false, destroy() { this.destroyed = true; } };
      buffers.push(buffer);
      return buffer;
    },
  };
}

// Scope/pool mechanics, independently of inference numerics. Two scopes on
// one real device must neither release nor destroy each other's allocations.
const host = mockDevice();
const scopeA = createResourceScope(host), scopeB = createResourceScope(host);
scopeA.beginRun(); scopeB.beginRun();
const poolA = bufferPoolFor(scopeA.device), poolB = bufferPoolFor(scopeB.device);
const a = poolA.acquire(16, 7), b = poolB.acquire(16, 7);
poolA.releaseAll(); scopeA.endRun();
assert.deepEqual(poolB.stats(), { inUse: 1, free: 0 });
assert.equal(poolA.acquire(16, 7), a);
assert.notEqual(poolB.acquire(16, 7), b);
assert.equal(a.destroyed, false, 'pooled allocation survives a successful run');
scopeA.dispose();
assert.equal(a.destroyed, true);
assert.equal(b.destroyed, false, 'another instance survives disposal');
scopeB.dispose();
console.log('PASS independent pools on one borrowed device');

// Exercise the actual public lifecycle wrappers with a deterministic workload,
// not a mock wrapper. Real WebGPU conformance is covered by the browser route.
class ControlledInference extends MoGeInference {
  async _initialize() {
    this.initCalls = (this.initCalls || 0) + 1;
    this.weight = this._device.createBuffer({ size: 16, usage: 7 });
    await this.initGate?.promise;
    this.useRealWeights = true;
  }
  async _run() {
    this.runCalls = (this.runCalls || 0) + 1;
    this.temporary = this._device.createBuffer({ size: 32, usage: 7 });
    this.started?.resolve();
    await this.runGate?.promise;
    if (this.failRun) throw new Error('injected failure after allocation');
    return 42;
  }
}

const waiting = new ControlledInference({ device: mockDevice() });
await assert.rejects(waiting.run(), /Call init/);
waiting.initGate = deferred();
const init = waiting.init();
assert.equal(waiting.init(), init);
const waitingRun = waiting.run();
await assert.rejects(waiting.run(), /already running/);
assert.equal(waiting.runCalls, undefined);
waiting.initGate.resolve();
assert.equal(await waitingRun, 42);
assert.equal(waiting.initCalls, 1);
assert.equal(waiting.temporary.destroyed, true);
assert.equal(waiting.weight.destroyed, false);
await waiting.dispose();
assert.equal(waiting.weight.destroyed, true);
console.log('PASS run awaits init, overlap rejects, successful transients retire');

const device = mockDevice();
const failed = new ControlledInference({ device });
await failed.init();
const fence = deferred(), fenceEntered = deferred();
device.queue.onSubmittedWorkDone = () => { fenceEntered.resolve(); return fence.promise; };
failed.failRun = true;
const rejected = assert.rejects(failed.run(), /injected failure/);
await fenceEntered.promise;
assert.equal(failed.temporary.destroyed, false, 'failed run must wait for submitted work');
await assert.rejects(failed.run(), /already running/);
fence.resolve();
await rejected;
assert.equal(failed.temporary.destroyed, true);
failed.failRun = false;
assert.equal(await failed.run(), 42, 'failure must not poison a healthy instance');
await failed.dispose();
console.log('PASS exception cleanup waits for fence and permits retry');

const retiring = new ControlledInference({ device: mockDevice() });
await retiring.init();
retiring.runGate = deferred(); retiring.started = deferred();
const active = retiring.run();
await retiring.started.promise;
const disposal = retiring.dispose();
assert.equal(retiring.dispose(), disposal);
await assert.rejects(retiring.run(), /disposed/);
await assert.rejects(retiring.init(), /disposed/);
await assert.rejects(retiring.warmUp(), /disposed/);
await assert.rejects(retiring.runBackboneCompare(), /disposed/);
await assert.rejects(retiring.runBlock0Compare(), /disposed/);
assert.equal(retiring.weight.destroyed, false);
assert.equal(retiring.temporary.destroyed, false);
retiring.runGate.resolve();
assert.equal(await active, 42);
await disposal;
assert.equal(retiring.weight.destroyed, true);
assert.equal(retiring.temporary.destroyed, true);
console.log('PASS disposal waits for accepted run and rejects every new entry');

const loading = new ControlledInference({ device: mockDevice() });
loading.initGate = deferred();
const loadingInit = loading.init();
const loadingDisposal = loading.dispose();
assert.equal(loading.weight.destroyed, false);
loading.initGate.resolve();
await loadingInit; await loadingDisposal;
assert.equal(loading.weight.destroyed, true);
console.log('PASS disposal during initialization');

const brokenInit = new ControlledInference({ device: mockDevice() });
brokenInit._initialize = async function () {
  this.partialWeight = this._device.createBuffer({ size: 16, usage: 7 });
  throw new Error('failed initialization');
};
await assert.rejects(brokenInit.init(), /failed initialization/);
assert.equal(brokenInit.partialWeight.destroyed, true);
await brokenInit.dispose();
console.log('PASS failed initialization releases partial resources');
