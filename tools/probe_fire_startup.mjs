#!/usr/bin/env node
// Reversible startup diagnosis, not a cadence benchmark or production repair.
// The optional override replays Beaming's df23b7a3 capacity fix in the browser.
import assert from 'node:assert/strict';
import puppeteer from 'puppeteer-core';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import path from 'node:path';

const [rootArg, outArg, portArg, mode = 'unchanged'] = process.argv.slice(2);
assert.ok(rootArg && outArg && portArg, 'usage: root output port [grant-adapter-capacity]');
assert.ok(['unchanged', 'grant-adapter-capacity'].includes(mode));
const root = path.resolve(rootArg), out = path.resolve(outArg);
mkdirSync(out, { recursive: true });
const report = { status: 'running', phase: 'source-check', root, mode,
  authority: 'native-startup-only; browser override is not a source repair',
  console: [], pageErrors: [] };
const save = () => writeFileSync(path.join(out, 'startup.json'), JSON.stringify(report, null, 2));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
let browser, page;
save();
try {
  const url = new URL(readFileSync(path.join(root, 'artifacts/basin-mounts/elfinblue-fuckeryyy.moge-url.txt'), 'utf8').trim());
  assert.ok(['localhost', '127.0.0.1'].includes(url.hostname));
  url.port = portArg;
  const fragment = new URLSearchParams(url.hash.slice(1));
  fragment.set('moge_frame_admission', 'fresh-flame');
  url.hash = fragment.toString();
  report.requestedUrl = url.href;
  report.source = { commit: execFileSync('git', ['-C', root, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(), files: {} };
  for (const name of ['volume-core.js', 'moge-live-flame-inject.mjs', 'lib/moge-frame-admission.mjs', 'lib/moge-inference.js']) {
    const response = await fetch(new URL(name, url));
    assert.equal(response.status, 200, name);
    const local = hash(readFileSync(path.join(root, name))), served = hash(Buffer.from(await response.arrayBuffer()));
    report.source.files[name] = { local, served };
    assert.equal(served, local, `wrong served source: ${name}`);
  }
  report.phase = 'browser-launch'; save();
  browser = await puppeteer.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: false,
    args: ['--no-first-run', '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--window-size=1440,900'],
    defaultViewport: { width: 1440, height: 900 },
  });
  report.browser = await browser.version();
  page = await browser.newPage();
  page.setDefaultTimeout(0);
  page.setDefaultNavigationTimeout(0);
  page.on('console', message => report.console.push({ type: message.type(), text: message.text() }));
  page.on('pageerror', error => report.pageErrors.push(String(error)));
  await page.evaluateOnNewDocument(override => {
    window.__startupDevices = [];
    const original = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function(descriptor = {}) {
      const effective = structuredClone(descriptor);
      const row = { requested: descriptor, effective, adapter: {
        vendor: this.info?.vendor, architecture: this.info?.architecture,
        maxStorageBuffersPerShaderStage: this.limits.maxStorageBuffersPerShaderStage,
      }, errors: [] };
      window.__startupDevices.push(row);
      if (override && effective.requiredLimits?.maxStorageBuffersPerShaderStage === 9) {
        effective.requiredLimits.maxStorageBuffersPerShaderStage = this.limits.maxStorageBuffersPerShaderStage;
        row.overrideApplied = true;
      }
      const device = await original.call(this, effective);
      row.grantedStorageBindings = device.limits.maxStorageBuffersPerShaderStage;
      device.addEventListener('uncapturederror', event => row.errors.push(event.error?.message || String(event.error)));
      return device;
    };
  }, mode === 'grant-adapter-capacity');
  report.phase = 'native-startup'; save();
  const response = await page.goto(url.href, { waitUntil: 'domcontentloaded' });
  report.effectiveUrl = response.url();
  assert.equal(response.status(), 200);
  assert.equal(new URL(report.effectiveUrl).origin, url.origin);
  await page.bringToFront();
  report.pageVisibility = await page.evaluate(() => document.visibilityState);
  assert.equal(report.pageVisibility, 'visible', 'native startup page must be foreground-visible');
  await page.waitForFunction(() => {
    const state = window.__kaminosVolumePrototype?.debugState();
    return state?.error || (state?.active && state.frameCount > 8 && state.simStepCount > 8);
  });
  report.state = await page.evaluate(() => window.__kaminosVolumePrototype.debugState());
  report.devices = await page.evaluate(() => window.__startupDevices);
  report.compositionRoute = await page.evaluate(() => window.__compositionRoute ?? null);
  await page.screenshot({ path: path.join(out, 'full.png') });
  assert.equal(report.state.error, null, report.state.error || 'native startup error');
  assert.equal(report.state.active, true);
  assert.match(report.state.backend, /^WebGPU:/);
  assert.ok(report.state.frameCount > 8 && report.state.simStepCount > 8);
  if (mode === 'grant-adapter-capacity') assert.equal(report.devices.filter(row => row.overrideApplied).length, 1);
  for (const [name, identity] of Object.entries(report.source.files)) {
    assert.equal(hash(readFileSync(path.join(root, name))), identity.local, `source changed during observation: ${name}`);
  }
  report.status = 'complete';
} catch (error) {
  report.status = 'failed';
  report.failure = { phase: report.phase, message: String(error) };
  process.exitCode = 1;
} finally {
  if (page) {
    try {
      report.devices ??= await page.evaluate(() => window.__startupDevices ?? null);
      report.state ??= await page.evaluate(() => window.__kaminosVolumePrototype?.debugState() ?? null);
      await page.screenshot({ path: path.join(out, 'full.png') });
    } catch (error) { report.captureError = String(error); }
  }
  report.finishedAt = new Date().toISOString();
  save();
  if (browser) await browser.close();
  console.log(JSON.stringify({ status: report.status, phase: report.phase, failure: report.failure,
    state: report.state && { active: report.state.active, backend: report.state.backend,
      frameCount: report.state.frameCount, simStepCount: report.state.simStepCount, error: report.state.error } }));
}
