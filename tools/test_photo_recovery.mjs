import assert from 'node:assert/strict';
import puppeteer from 'puppeteer-core';
import path from 'node:path';
import { mkdir } from 'node:fs/promises';

const executablePath = process.env.CHROME_PATH;
assert.ok(executablePath && !executablePath.includes('/Applications/Google Chrome.app/'));
const browser = await puppeteer.launch({ executablePath, headless: true, args: ['--enable-unsafe-webgpu'] });
try {
  const page = await browser.newPage();
  await page.goto(process.env.DEMO_URL || 'http://127.0.0.1:5198/', { waitUntil: 'networkidle0' });
  await page.waitForFunction(() => !document.getElementById('run-button').disabled);
  await page.evaluate(async () => {
    const { MoGeInference } = await import('/src/lib/inference.js');
    MoGeInference.prototype.init = async function () { this.useRealWeights = false; };
    MoGeInference.prototype.run = async function () { throw new Error('TEST: stub execution reached'); };
    MoGeInference.prototype.dispose = async function () {};
  });
  await page.click('#run-button');
  await page.waitForFunction(() => window.__mogePhoto?.phase !== 'running');
  assert.equal(await page.$eval('#status', el => el.textContent), 'Stopped during weights', 'Stub fallback must be rejected before model execution');
  assert.match(await page.$eval('#error', el => el.textContent), /real model weights/i);
  assert.equal(await page.$eval('#run-button', el => el.disabled), false, 'Loading failure must allow retry');
  await page.evaluate(async () => {
    const { MoGeInference } = await import('/src/lib/inference.js');
    MoGeInference.prototype.init = async function () { this.useRealWeights = true; };
    MoGeInference.prototype.run = async function () {
      if (window.__testRejectNext) throw new Error('TEST: simulated inference failure');
      return { width: 2, height: 2, depth: new Float32Array([1, 1.01, 1.02, 1.03]),
        normals: new Float32Array([0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1]) };
    };
  });
  await page.click('#run-button');
  await page.waitForFunction(() => window.__mogePhoto?.phase === 'complete');
  assert.equal(await page.$eval('#explore-controls', el => el.disabled), false);
  assert.match(await page.$eval('#status', el => el.textContent), /^Inference: [\d.]+ s$/);
  assert.match(await page.$eval('#elapsed', el => el.textContent), /^Total: [\d.]+ s \(including preparation\)$/);
  assert.equal(await page.$eval('aside', el => el.firstElementChild.id), 'explore-section');
  assert.equal(await page.$eval('#run-button', el => el.classList.contains('primary')), false);
  assert.equal(await page.$eval('#run-button span', el => el.textContent), 'Run again');
  assert.match(await page.$eval('#result-detail', el => el.textContent), /inference/);
  await page.$eval('#zoom', el => { el.value = '1.2'; el.dispatchEvent(new Event('input')); });
  assert.equal(await page.$eval('#zoom-value', el => el.value), '1.20x');
  await page.click('#reset');
  assert.equal(await page.$eval('#zoom-value', el => el.value), '1.00x');
  if (process.env.PHOTO_UI_OUT) {
    await mkdir(process.env.PHOTO_UI_OUT, { recursive: true });
    for (const [name, width, height] of [['desktop', 1440, 960], ['narrow', 390, 844]]) {
      await page.setViewport({ width, height });
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
      const bounds = await page.evaluate(() => {
        const a = document.getElementById('explore-section').getBoundingClientRect();
        const b = document.getElementById('photograph-section').getBoundingClientRect();
        return { bottom: a.bottom, nextTop: b.top };
      });
      assert.ok(bounds.bottom <= bounds.nextTop, 'Explore and photograph controls cannot overlap');
      await page.screenshot({ path: path.join(process.env.PHOTO_UI_OUT, `${name}-synthetic-result.png`), fullPage: true });
    }
  }
  await page.evaluate(() => { window.__testPrevious = window.__mogeResult; window.__testRejectNext = true; });
  await page.click('#run-button');
  await page.waitForFunction(() => window.__mogePhoto?.phase === 'inference');
  assert.equal(await page.evaluate(() => window.__mogeResult === window.__testPrevious), true, 'Failed replacement retains result');
  assert.equal(await page.$eval('#explore-controls', el => el.disabled), false, 'Retained result stays explorable');
  assert.equal(await page.$eval('#scene-state', el => el.textContent), 'Inferred depth');
  assert.equal(await page.$eval('#run-button', el => el.disabled), false);
  await page.evaluate(() => { window.__testRejectNext = false; });
  await page.click('#run-button');
  await page.waitForFunction(() => window.__mogePhoto?.phase === 'complete');
  const originalPixel = await page.evaluate(async () => {
    const { PhotoViewer } = await import('/src/lib/photo-viewer.js');
    const holder = document.createElement('div'); holder.style.cssText = 'position:fixed;inset:0;width:160px;height:160px';
    const canvas = document.createElement('canvas'); holder.append(canvas); document.body.append(holder);
    const source = document.createElement('canvas'); source.width = source.height = 32;
    const ctx = source.getContext('2d'); ctx.fillStyle = '#00ff00'; ctx.fillRect(0, 0, 32, 32);
    const viewer = new PhotoViewer(canvas);
    try {
      viewer.setImage(source, { width: 4, height: 2, depth: new Float32Array([1, 1, 4, 4, 1, 1, 4, 4]),
        normals: new Float32Array(Array.from({ length: 8 }, () => [0, 0, -1]).flat()) });
      viewer.view = 'original'; viewer.render();
      const gl = canvas.getContext('webgl2'), pixel = new Uint8Array(4);
      gl.readPixels(Math.floor(canvas.width / 2), Math.floor(canvas.height / 2), 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
      return [...pixel];
    } finally { viewer.dispose(); holder.remove(); }
  });
  assert.deepEqual(originalPixel, [0, 255, 0, 255], 'Original must not inherit rejected depth triangles');
  let release;
  const held = new Promise(resolve => { release = resolve; });
  await page.setRequestInterception(true);
  page.on('request', request => {
    if (request.url().endsWith('/test_fixtures/input.png') && request.resourceType() === 'fetch') release(request);
    else request.continue();
  });
  await page.click('#sample-button');
  const request = await held;
  await (await page.$('#file-input')).uploadFile(path.resolve('public/test_fixtures/input.png'));
  await page.waitForFunction(() => document.getElementById('selected-name').textContent === 'input.png');
  await request.continue();
  await page.waitForNetworkIdle();
  assert.equal(await page.$eval('#selected-name', el => el.textContent), 'input.png', 'Older example fetch cannot replace newer selection');
  assert.equal(await page.$eval('#run-button', el => el.classList.contains('primary')), true, 'New image restores the main action');
  assert.equal(await page.$eval('#elapsed', el => el.textContent), '', 'Selection must clear the previous run total');
  console.log('Photo recovery: simulated fallback refused, retry, retained result after simulated failure, repeat passed (synthetic controller test only)');
} finally { await browser.close(); }
