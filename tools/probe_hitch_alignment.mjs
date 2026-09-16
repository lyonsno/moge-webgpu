#!/usr/bin/env node
// Headless relative-only witness. Raw observations are retained without caps.
import puppeteer from 'puppeteer-core';
import { mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { analyzeHitches } from './hitch_analysis.mjs';

const args = process.argv.slice(2);
const arg = (name, fallback) => args.includes(name) ? args[args.indexOf(name) + 1] : fallback;
const url = arg('--url', 'http://localhost:8093/moge-live-flame.html');
const outDir = path.resolve(arg('--out', '/tmp/moge-hitch-probe'));
const runId = arg('--run-id', randomUUID());
const hitchMs = Number(arg('--hitch-ms', '50'));
const settleMs = Number(arg('--settle-ms', '4000'));
mkdirSync(outDir, { recursive: true });
const report = { status: 'running', runId, url, startedAt: new Date().toISOString(),
  measurementAuthority: 'headless-relative-only', raw: null, phase: 'configuration' };
const save = () => writeFileSync(path.join(outDir, 'hitch-report.json'), JSON.stringify(report, null, 2));
save(); // Invalidate old success before fallible work, including launch.
let browser, page;
try {
  if (!(hitchMs > 0) || !Number.isFinite(hitchMs) || !(settleMs >= 0) || !Number.isFinite(settleMs)) throw new Error('Invalid hitch/settle milliseconds');
  const manifestPath = arg('--source-manifest', null);
  report.sourceIdentity = manifestPath ? JSON.parse(readFileSync(manifestPath, 'utf8'))
    : { status: 'unverified', reason: 'no owned-server source manifest supplied' };
  if (manifestPath) {
    report.phase = 'source-admission'; save();
    const response = await fetch(url);
    if (!response.ok || response.headers.get('x-moge-witness-run') !== runId
        || report.sourceIdentity.runId !== runId || report.sourceIdentity.url !== url) throw new Error('Requested route is not served by this witness server');
  }
  report.phase = 'browser-launch'; save();
  browser = await puppeteer.launch({
    executablePath: arg('--chrome', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'),
    headless: true,
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--disable-gpu-sandbox', '--no-sandbox'],
    defaultViewport: { width: 1440, height: 900 },
  });
  report.browser = await browser.version();
  page = await browser.newPage();
  await page.setCacheEnabled(false);
  report.phase = 'page-load'; save();
  const response = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
  report.effectiveUrl = response?.url();
  if (manifestPath && response?.headers()['x-moge-witness-run'] !== runId) throw new Error('Browser navigated to a different server');
  await page.waitForFunction(() => window.__mogeLiveFlameReady === true, { timeout: 600000 });
  await new Promise(resolve => setTimeout(resolve, settleMs));
  report.phase = 'observation'; save();
  report.raw = await page.evaluate(async ({ runId, sourceIdentity }) => {
    const raw = { runId, sourceIdentity, frameTimes: [], baselineStart: performance.now(),
      compositionRoute: window.__compositionRoute ?? null, effectiveUrl: location.href };
    window.__mogeHitchObservation = raw;
    let rafId;
    const tick = now => { raw.frameTimes.push(now); rafId = requestAnimationFrame(tick); };
    rafId = requestAnimationFrame(tick);
    try {
      await new Promise(resolve => setTimeout(resolve, 3000));
      const button = document.getElementById('ignite'), hud = document.getElementById('hud-infer');
      if (!button || button.disabled || !hud) throw new Error('Inference controls unavailable');
      if (window.__mogeDebug) window.__mogeDebug.webGpuRouteResult = null;
      hud.textContent = 'probe awaiting current inference';
      raw.inferStart = performance.now();
      button.click();
      await new Promise(resolve => {
        const check = () => /done|error/i.test(hud.textContent) ? resolve() : setTimeout(check, 200);
        check();
      });
      raw.inferEnd = performance.now();
      raw.terminalStatus = hud.textContent;
      raw.routeResult = window.__mogeDebug?.webGpuRouteResult ?? null;
      // The nested scheduler receipt owns its status and trace; no duplicate
      // fields that can drift into contradictory evidence.
      await new Promise(resolve => setTimeout(resolve, 1500));
    } catch (error) { raw.observationError = String(error); }
    finally { cancelAnimationFrame(rafId); }
    return raw;
  }, { runId, sourceIdentity: report.sourceIdentity });
  save();
  report.phase = 'analysis';
  Object.assign(report, analyzeHitches(report.raw, hitchMs));
  if (report.evidence.errors.length) throw new Error(report.evidence.errors.join('; '));
  report.status = 'complete';
  console.log(JSON.stringify(report.summary, null, 2));
} catch (error) {
  if (!report.raw && page && report.phase === 'observation') {
    try { report.raw = await page.evaluate(() => window.__mogeHitchObservation ?? null); }
    catch (recoveryError) { report.rawRecoveryError = String(recoveryError); }
  }
  report.status = 'failed';
  report.failure = { phase: report.phase, message: String(error) };
  process.exitCode = 1;
  console.error(report.failure);
} finally {
  report.finishedAt = new Date().toISOString(); save();
  if (browser) {
    try { await browser.close(); }
    catch (error) { report.cleanupError = String(error); save(); process.exitCode = 1; }
  }
}
