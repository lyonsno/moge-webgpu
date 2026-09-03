#!/usr/bin/env node
/**
 * Visual smoke witness for the Kaminos moge-live-flame composition page.
 *
 * Loads the page, waits for fire + weights, verifies the fire canvas is
 * actually animating (two frame captures differ), runs cooperative MoGe
 * inference via the page button, and asserts:
 *   - frames kept rendering during inference (page not frozen)
 *   - scheduler receipt verified
 *   - depth panel appeared with non-blank content
 * Captures screenshots before/during/after for human smoke.
 *
 * Usage: node tools/smoke_live_flame_page.mjs --url http://localhost:8090/moge-live-flame.html --out <dir>
 */
import puppeteer from 'puppeteer-core';
import { mkdirSync, writeFileSync } from 'fs';
import path from 'path';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const args = process.argv.slice(2);
const arg = (name, dflt) => args.includes(name) ? args[args.indexOf(name) + 1] : dflt;
const url = arg('--url', 'http://localhost:8090/moge-live-flame.html');
const outDir = path.resolve(arg('--out', '/tmp/moge-live-flame-smoke'));
mkdirSync(outDir, { recursive: true });

const browser = await puppeteer.launch({
  executablePath: CHROME_PATH,
  headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--disable-gpu-sandbox', '--no-sandbox', '--window-size=1440,900'],
  defaultViewport: { width: 1440, height: 900 },
});
const page = await browser.newPage();
const pageErrors = [];
page.on('pageerror', e => pageErrors.push(e.message));
page.on('console', m => { if (/error|Error/.test(m.text())) console.log(`  [page] ${m.text()}`); });

const failures = [];
try {
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForFunction(() => window.__mogeLiveFlameReady === true, { timeout: 600000 });

  // Fire liveness: two viewport captures a second apart must differ.
  await new Promise(r => setTimeout(r, 2500));
  const shotA = await page.screenshot({ encoding: 'binary' });
  await new Promise(r => setTimeout(r, 1000));
  const shotB = await page.screenshot({ encoding: 'binary' });
  writeFileSync(path.join(outDir, 'flame-before.png'), shotB);
  const fireAnimating = Buffer.compare(shotA, shotB) !== 0;
  if (!fireAnimating) failures.push('fire canvas is not animating (identical frames 1s apart)');

  const fireStatus = await page.evaluate(() => document.getElementById('hud-fire').textContent);
  if (/error/i.test(fireStatus)) failures.push(`fire status: ${fireStatus}`);

  await page.click('#ignite');
  // Screenshot mid-inference for the human record.
  await new Promise(r => setTimeout(r, 1500));
  await page.screenshot({ path: path.join(outDir, 'flame-during-inference.png') });

  await page.waitForFunction(
    () => /done|error/.test(document.getElementById('hud-infer').textContent),
    { timeout: 300000 }
  );
  await new Promise(r => setTimeout(r, 500));
  await page.screenshot({ path: path.join(outDir, 'flame-after-depth.png') });

  const verdict = await page.evaluate(() => ({
    infer: document.getElementById('hud-infer').textContent,
    sched: document.getElementById('hud-sched').textContent,
    frames: Number(document.getElementById('hud-frames').textContent),
    p95: document.getElementById('hud-p95').textContent,
    weights: document.getElementById('hud-weights').textContent,
    depthVisible: document.getElementById('depth-panel').style.display === 'block',
    depthNonBlank: (() => {
      const c = document.getElementById('depth-canvas');
      const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
      let sum = 0;
      for (let i = 0; i < d.length; i += 4096) sum += d[i] + d[i + 1] + d[i + 2];
      return sum > 0;
    })(),
    routeResult: window.__mogeInference?.weightsSource || null,
  }));

  if (/error/.test(verdict.infer)) failures.push(`inference: ${verdict.infer}`);
  if (!/verified/.test(verdict.sched)) failures.push(`scheduler receipt not verified: ${verdict.sched}`);
  if (!(verdict.frames > 30)) failures.push(`too few frames during inference: ${verdict.frames}`);
  if (!verdict.depthVisible || !verdict.depthNonBlank) failures.push('depth panel missing or blank');
  if (/STUB/.test(verdict.weights)) failures.push(`weights: ${verdict.weights}`);
  if (pageErrors.length) failures.push(`page errors: ${pageErrors.slice(0, 3).join('; ')}`);

  writeFileSync(path.join(outDir, 'verdict.json'), JSON.stringify({ url, verdict, fireAnimating, failures }, null, 2));
  console.log(JSON.stringify({ verdict, fireAnimating, failures }, null, 2));
} catch (e) {
  failures.push(`ERROR: ${e.message}`);
  console.error(e.message);
  try { await page.screenshot({ path: path.join(outDir, 'flame-failure.png') }); } catch {}
  writeFileSync(path.join(outDir, 'verdict.json'), JSON.stringify({ url, failures, pageErrors }, null, 2));
} finally {
  await browser.close();
}
process.exitCode = failures.length ? 1 : 0;
console.log(failures.length ? 'FAIL' : 'PASS');
