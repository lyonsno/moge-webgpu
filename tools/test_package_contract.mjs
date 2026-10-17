import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import { build } from 'vite';

const scratch = await fs.mkdtemp(path.join(os.tmpdir(), 'moge-package-'));
const root = path.resolve(import.meta.dirname, '..');
const previousFetch = globalThis.fetch;
try {
  const reference = path.join(scratch, 'reference');
  await build({ root, configFile: path.join(root, 'vite.lib.config.js'), build: { outDir: reference }, logLevel: 'error' });
  if (process.argv.includes('--baseline')) {
    assert.ok(await fs.stat(path.join(reference, 'package.json')).catch(() => null),
      'standalone MoGe build does not yet provide an installable model package');
  }
  const { packMoGeModule } = await import('./pack-module.mjs');
  const output = path.join(scratch, 'artifacts');
  await fs.mkdir(output);
  await fs.writeFile(path.join(output, 'keep.txt'), 'caller-owned output');
  const packed = await packMoGeModule({ out: output, version: '0.2.3' });
  assert.equal(packed.name, '@kaminos/moge');
  assert.equal(packed.version, '0.2.3');
  assert.equal(await fs.readFile(path.join(output, 'keep.txt'), 'utf8'), 'caller-owned output');
  const entries = execFileSync('tar', ['-tzf', packed.tarball], { encoding: 'utf8' }).trim().split('\n').sort();
  assert.deepEqual(entries, ['LICENSE', 'README.md', 'THIRD_PARTY_NOTICES.txt', 'moge-inference.js', 'package.json'].map(file => `package/${file}`).sort());
  const app = path.join(scratch, 'app');
  await fs.mkdir(app);
  await fs.writeFile(path.join(app, 'package.json'), JSON.stringify({ private: true, type: 'module' }));
  execFileSync('npm', ['install', '--ignore-scripts', '--no-audit', '--no-fund', '--package-lock=false', packed.tarball],
    { cwd: app, stdio: 'pipe' });
  const require = createRequire(path.join(app, 'package.json'));
  const entry = require.resolve('@kaminos/moge');
  const manifest = require('@kaminos/moge/package.json');
  assert.equal(manifest.private, true);
  assert.equal(manifest.license, 'MIT');
  assert.equal(manifest.version, '0.2.3');
  assert.equal(manifest.dependencies, undefined);
  assert.deepEqual(await fs.readFile(entry), await fs.readFile(path.join(reference, 'moge-inference.js')),
    'packaging must not change the exercised MoGe runtime');
  assert.deepEqual(await fs.readFile(path.join(path.dirname(entry), 'LICENSE')), await fs.readFile(path.join(root, 'LICENSE')));
  assert.deepEqual(await fs.readFile(path.join(path.dirname(entry), 'THIRD_PARTY_NOTICES.txt')),
    await fs.readFile(path.join(root, 'THIRD_PARTY_NOTICES.txt')));
  const api = await import(pathToFileURL(entry).href);
  assert.equal(typeof api.initGPU, 'function');
  assert.equal(typeof api.MoGeInference.prototype.run, 'function');
  const device = { queue: { onSubmittedWorkDone: async () => {} }, destroy() { assert.fail('application owns device'); } };
  const missing = new Error('installed application offline');
  const requested = [];
  globalThis.fetch = async url => { requested.push(url); throw missing; };
  const model = new api.MoGeInference({ device }, { weightsUrl: 'https://example.invalid/weights.bin' });
  try {
    await assert.rejects(model.init(), error => error === missing);
    assert.deepEqual(requested, ['https://example.invalid/weights.bin']);
  } finally { await model.dispose(); }
  console.log('MoGe installed package, explicit loading, borrowed device and unchanged runtime: PASS');
} finally {
  globalThis.fetch = previousFetch;
  await fs.rm(scratch, { recursive: true, force: true });
}
