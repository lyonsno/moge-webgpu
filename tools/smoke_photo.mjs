import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import puppeteer from 'puppeteer-core';

const out = path.resolve(process.env.PHOTO_SMOKE_OUT || '/tmp/moge-photo-smoke');
await fs.mkdir(out, { recursive: true });
const executablePath = process.env.CHROME_PATH;
const report = { ok: false, phase: 'launch', executablePath, url: process.env.DEMO_URL || 'http://127.0.0.1:5198/',
  sourceCommit: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(), errors: [], infer: process.argv.includes('--infer') };
let browser, page;
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const settle = () => new Promise(resolve => setTimeout(resolve, 600));
try {
  assert.ok(executablePath && !executablePath.includes('/Applications/Google Chrome.app/'), 'Independent testing browser required');
  browser = await puppeteer.launch({ executablePath, headless: true, args: ['--enable-unsafe-webgpu'],
    defaultViewport: { width: 1440, height: 960 } });
  page = await browser.newPage();
  page.on('pageerror', e => report.errors.push(e.message));
  page.on('console', msg => { if (msg.type() === 'error') report.errors.push(msg.text()); });
  report.phase = 'entry';
  await page.goto(report.url, { waitUntil: 'networkidle0' });
  await page.waitForFunction(() => !document.getElementById('run-button').disabled);
  await settle();
  await page.screenshot({ path: path.join(out, 'desktop-original.png') });
  const pixels = await page.$eval('#photo-canvas', canvas => {
    const gl = canvas.getContext('webgl2');
    const p = new Uint8Array(4 * gl.drawingBufferWidth * gl.drawingBufferHeight);
    gl.readPixels(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight, gl.RGBA, gl.UNSIGNED_BYTE, p);
    let min = 255, max = 0;
    for (let i = 0; i < p.length; i += 4) { min = Math.min(min, p[i]); max = Math.max(max, p[i]); }
    return { min, max, width: canvas.width, height: canvas.height };
  });
  report.entryPixels = pixels;
  assert.ok(pixels.max - pixels.min > 50, 'Photograph canvas must have visible image detail');
  if (report.infer) {
    report.phase = 'inference';
    await page.click('#run-button');
    await page.waitForFunction(() => window.__mogePhoto?.phase !== 'running', { timeout: 0 });
    report.firstRun = await page.evaluate(() => ({ ...window.__mogePhoto,
      validDepth: window.__mogeResult && [...window.__mogeResult.depth].filter(d => Number.isFinite(d) && d > 0).length,
      scheduler: window.__mogeResult?.schedulerVerificationReceipt,
      backend: window.__mogeDebug }));
    assert.equal(report.firstRun.phase, 'complete', 'Actual model and surface must complete');
    assert.ok(report.firstRun.validDepth > 0);
    await settle();
    const canvas = await page.$('#photo-canvas');
    await page.screenshot({ path: path.join(out, 'desktop-depth.png') });
    const home = hash(await canvas.screenshot());
    await canvas.focus(); await page.keyboard.press('ArrowRight'); await page.keyboard.press('ArrowRight');
    await settle();
    const moved = hash(await canvas.screenshot());
    assert.notEqual(home, moved, 'Camera movement must change rendered pixels');
    await page.screenshot({ path: path.join(out, 'desktop-moved.png') });
    await page.click('[data-tool="light"]'); await canvas.focus();
    await page.keyboard.press('ArrowLeft'); await settle();
    const lit = hash(await canvas.screenshot());
    assert.notEqual(moved, lit, 'Relighting must change rendered pixels');
    await page.screenshot({ path: path.join(out, 'desktop-relit.png') });
    report.phase = 'retained-scene';
    const retained = await page.$eval('#displayed-name', el => el.textContent);
    await page.click('#run-button');
    assert.equal(await page.$eval('#explore-controls', el => el.disabled), false, 'Previous scene remains interactive');
    assert.equal(await page.$eval('#displayed-name', el => el.textContent), retained);
    await canvas.focus(); await page.keyboard.press('ArrowRight');
    await page.waitForFunction(() => window.__mogePhoto?.phase !== 'running', { timeout: 0 });
    report.repeatRun = await page.evaluate(() => window.__mogePhoto);
    assert.equal(report.repeatRun.phase, 'complete');
    report.interactionHashes = { home, moved, lit };
  }
  report.phase = 'mobile';
  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  await settle();
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, 'Mobile has no horizontal overflow');
  await page.screenshot({ path: path.join(out, 'mobile.png'), fullPage: true });
  assert.deepEqual(report.errors, [], 'No browser/shader errors');
  report.phase = 'complete'; report.ok = true;
} catch (error) {
  report.error = error.stack;
  if (page) await page.screenshot({ path: path.join(out, 'failure.png'), fullPage: true }).catch(() => {});
  process.exitCode = 1;
} finally {
  if (browser) await browser.close();
  await fs.writeFile(path.join(out, 'report.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
}
