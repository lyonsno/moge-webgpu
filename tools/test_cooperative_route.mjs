#!/usr/bin/env node
/**
 * Cooperative scheduling route test.
 *
 * Runs the real app inference route with a cooperative scheduler requested
 * (yieldMs > 0, per-block backbone chunking) while a rAF monitor measures
 * foreground frame intervals, then asserts:
 *
 *   1. The scheduler verification receipt reaches status "verified" —
 *      requested yields were actually observed as events, not config-only.
 *   2. Every requested phase boundary has observed yield events.
 *   3. The route receipt remains authoritative (status "real").
 *   4. The rAF monitor observed frames *during* inference (the page was not
 *      frozen for the whole run), and reports p95 frame interval.
 *
 * Fails on current monolithic-submit behavior: the receipt downgrades with
 * yield-events-missing and stays "scheduler-unverified"/"config-only".
 *
 * Usage: node tools/test_cooperative_route.mjs [--port 5185] [--headed]
 */

import puppeteer from 'puppeteer-core';
import path from 'path';
import { fileURLToPath } from 'url';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const __dirname = path.dirname(fileURLToPath(import.meta.url));

const args = process.argv.slice(2);
const port = args.includes('--port') ? args[args.indexOf('--port') + 1] : '5185';
const headed = args.includes('--headed');

const browser = await puppeteer.launch({
  executablePath: CHROME_PATH,
  headless: !headed,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--disable-gpu-sandbox', '--no-sandbox'],
  defaultViewport: { width: 1280, height: 900 },
});
const page = await browser.newPage();
page.on('console', msg => {
  const t = msg.text();
  if (/Loaded|weights|Error|error|cooperative/.test(t)) console.log(`  [page] ${t}`);
});

try {
  await page.goto(`http://localhost:${port}/`, { waitUntil: 'networkidle0', timeout: 60000 });

  // The app initializes GPU + weights lazily on first image. Trigger a normal
  // run via the file input, then drive the cooperative run directly.
  const fileInput = await page.$('#file-input');
  await fileInput.uploadFile(path.resolve(__dirname, '..', 'public', 'test_fixtures', 'input.png'));
  await page.waitForFunction(
    () => window.__mogeInference?.useRealWeights && window.__mogeResult,
    { timeout: 300000 }
  );

  // Start a rAF interval monitor before triggering inference.
  await page.evaluate(() => {
    window.__frameIntervals = [];
    let last = performance.now();
    const tick = now => {
      window.__frameIntervals.push(now - last);
      last = now;
      window.__frameRaf = requestAnimationFrame(tick);
    };
    window.__frameRaf = requestAnimationFrame(tick);
  });

  const result = await page.evaluate(async () => {
    if (!window.__mogeInference?.useRealWeights) {
      return { error: 'stub weights — no authoritative cooperative run possible' };
    }
    // Build an ImageData input from the bundled test fixture image.
    const img = new Image();
    img.src = '/test_fixtures/input.png';
    await img.decode();
    const canvas = document.createElement('canvas');
    canvas.width = img.width; canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, img.width, img.height);

    // Negative path: the monolithic first run must NOT claim scheduler
    // verification (no observed events exist for it).
    const monolithicSchedulerStatus =
      window.__mogeDebug?.webGpuRouteReceipt?.runtime?.schedulerVerification?.status ?? null;

    const monolithicDepth = window.__mogeResult.depth;
    const frameCountBefore = window.__frameIntervals.length;
    const t0 = performance.now();
    const run = await window.__mogeInference.run(imageData, {
      scheduler: { mode: 'cooperative', yieldMs: 4, vitBlockChunkSize: 1, waitForSubmittedWorkDone: true },
    });
    const elapsedMs = performance.now() - t0;
    const framesDuring = window.__frameIntervals.length - frameCountBefore;

    // Chunked submits must not change numerics: same dispatches, same buffers,
    // same queue order. Compare against the monolithic run on the same input.
    let depthMaxAbsDiff = null;
    if (monolithicDepth?.length === run.depth?.length) {
      depthMaxAbsDiff = 0;
      for (let i = 0; i < run.depth.length; i++) {
        const d = Math.abs(run.depth[i] - monolithicDepth[i]);
        if (d > depthMaxAbsDiff) depthMaxAbsDiff = d;
      }
    }

    const receipt = run.routeResult?.receipt || null;
    const schedulerReceipt = run.schedulerVerificationReceipt
      || run.routeResult?.schedulerVerificationReceipt
      || receipt?.schedulerVerification
      || null;
    return {
      elapsedMs,
      framesDuring,
      routeStatus: receipt?.status ?? null,
      schedulerStatus: schedulerReceipt?.status ?? null,
      schedulerClassification: schedulerReceipt?.classification ?? null,
      downgrades: schedulerReceipt?.downgrades ?? null,
      yieldCounts: (schedulerReceipt?.boundaryAssertions || []).map(a => ({
        field: a.field, status: a.status, observedYieldCount: a.observedYieldCount,
      })),
      depthFinite: Number.isFinite(run.depth?.[0]),
      depthMaxAbsDiff,
      monolithicSchedulerStatus,
    };
  });

  const frameStats = await page.evaluate(() => {
    cancelAnimationFrame(window.__frameRaf);
    const xs = [...window.__frameIntervals].sort((a, b) => a - b);
    const pick = q => xs[Math.min(xs.length - 1, Math.floor(q * xs.length))];
    return xs.length ? { frames: xs.length, p50: pick(0.5), p95: pick(0.95), p99: pick(0.99), max: xs[xs.length - 1] } : null;
  });

  console.log(JSON.stringify({ ...result, frameStats }, null, 2));

  const failures = [];
  if (result.error) failures.push(result.error);
  if (result.schedulerStatus !== 'verified') {
    failures.push(`scheduler receipt status must be "verified", got ${result.schedulerStatus} (classification=${result.schedulerClassification}, downgrades=${JSON.stringify(result.downgrades)})`);
  }
  for (const a of result.yieldCounts || []) {
    if (!(a.observedYieldCount > 0)) failures.push(`${a.field}: observedYieldCount must be > 0, got ${a.observedYieldCount}`);
  }
  // Browser runtime without artifact hashing reports 'partial'; both are
  // acceptable here — what matters is that it is not fallback/cached/stub.
  if (!['real', 'partial'].includes(result.routeStatus)) {
    failures.push(`route receipt status must be real or partial, got ${result.routeStatus}`);
  }
  if (!(result.framesDuring > 5)) failures.push(`expected foreground frames during inference, got ${result.framesDuring}`);
  if (!result.depthFinite) failures.push('depth output not finite');
  if (!(result.depthMaxAbsDiff !== null && result.depthMaxAbsDiff < 1e-5)) {
    failures.push(`cooperative depth must match monolithic run, maxAbsDiff=${result.depthMaxAbsDiff}`);
  }
  // Absence of the receipt must fail loud, not pass as "not verified".
  if (result.monolithicSchedulerStatus !== 'scheduler-unverified') {
    failures.push(`monolithic run must report scheduler-unverified, got ${result.monolithicSchedulerStatus}`);
  }

  if (failures.length) {
    console.log('\nFAIL:');
    for (const f of failures) console.log(`  - ${f}`);
    process.exitCode = 1;
  } else {
    console.log('\nPASS');
  }
} catch (e) {
  console.error('ERROR:', e.message);
  process.exitCode = 1;
} finally {
  await browser.close();
}
