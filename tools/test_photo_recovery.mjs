import assert from 'node:assert/strict';
import puppeteer from 'puppeteer-core';
import path from 'node:path';

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
  console.log('Photo recovery: simulated fallback refused, retry, retained result after simulated failure, repeat passed (synthetic controller test only)');
} finally { await browser.close(); }
