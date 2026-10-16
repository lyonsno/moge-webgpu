import assert from 'node:assert/strict';
import { test } from 'node:test';
import { copyFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { MoGeInference } from '../dist-lib/moge-inference.js';

const hostedUrl = 'https://huggingface.co/lyonsno/moge-webgpu/resolve/main/weights.bin';
const customUrl = 'https://example.invalid/models/moge.bin';

function model(options) {
  const device = {
    queue: { onSubmittedWorkDone: async () => {} },
    destroy() { assert.fail('model must not destroy the application device'); },
  };
  const inference = new MoGeInference({ device }, options);
  inference._createStubWeights = () => ({ diagnostic: true });
  return inference;
}

async function withFetch(fetch, run) {
  const previous = globalThis.fetch;
  globalThis.fetch = fetch;
  try { await run(); }
  finally { globalThis.fetch = previous; }
}

test('explicit weights are fetched directly and an HTTP failure rejects init', async () => {
  const requests = [];
  await withFetch(async (url, options) => {
    requests.push({ url, method: options?.method || 'GET' });
    return new Response('missing weights', { status: 404 });
  }, async () => {
    const inference = model({ weightsUrl: customUrl });
    try {
      await assert.rejects(inference.init(), /Failed to fetch weights: 404/);
      assert.deepEqual(requests, [{ url: customUrl, method: 'GET' }]);
      assert.equal(inference.weights, null);
      assert.equal(inference.useRealWeights, false);
    } finally { await inference.dispose(); }
  });
});

test('default weights do not probe an application-relative file or read page flags', async () => {
  const previousLocation = globalThis.location;
  globalThis.location = { search: '?forceStub=1' };
  const requests = [];
  const failure = new Error('offline');
  try {
    await withFetch(async (url, options) => {
      requests.push({ url, method: options?.method || 'GET' });
      throw failure;
    }, async () => {
      const inference = model();
      try {
        await assert.rejects(inference.init(), error => error === failure);
        assert.deepEqual(requests, [{ url: hostedUrl, method: 'GET' }]);
        assert.equal(inference.weights, null);
      } finally { await inference.dispose(); }
    });
  } finally {
    if (previousLocation === undefined) delete globalThis.location;
    else globalThis.location = previousLocation;
  }
});

test('invalid weights reject instead of producing a usable diagnostic model', async () => {
  const progress = [];
  await withFetch(async () => new Response(new Uint8Array(16), {
    headers: { 'content-length': '16' },
  }), async () => {
    const inference = model({ weightsUrl: customUrl });
    try {
      await assert.rejects(inference.init((received, total) => progress.push([received, total])), /magic|header|version/i);
      assert.deepEqual(progress, [[16, 16]]);
      assert.equal(inference.weights, null);
      assert.equal(inference.useRealWeights, false);
    } finally { await inference.dispose(); }
  });
});

test('built module imports independently of checkout, dependencies, and page globals', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'moge-embedding-'));
  try {
    const modulePath = join(directory, 'moge-inference.mjs');
    await copyFile(new URL('../dist-lib/moge-inference.js', import.meta.url), modulePath);
    const embedded = await import(pathToFileURL(modulePath).href);
    assert.equal(typeof embedded.initGPU, 'function');
    assert.equal(typeof embedded.buildMogeDeviceRequest, 'function');
    assert.equal(typeof embedded.MoGeInference.prototype.run, 'function');
    const device = {
      queue: { onSubmittedWorkDone: async () => {} },
      destroy() { assert.fail('must preserve application device'); },
    };
    const inference = new embedded.MoGeInference({ device }, { weightsUrl: customUrl });
    const failure = new Error('independent app network failure');
    await withFetch(async url => {
      assert.equal(url, customUrl);
      throw failure;
    }, async () => {
      try { await assert.rejects(inference.init(), error => error === failure); }
      finally { await inference.dispose(); }
    });
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('diagnostic stub mode is explicit, labeled, and makes no weight request', async () => {
  await withFetch(async () => assert.fail('diagnostic mode must not download weights'), async () => {
    const inference = model({ diagnosticStub: true });
    try {
      await inference.init();
      assert.equal(inference.useRealWeights, false);
      assert.equal(inference.weightsSource, 'diagnostic-stub');
      assert.deepEqual(inference.weights, { diagnostic: true });
    } finally { await inference.dispose(); }
  });
});
