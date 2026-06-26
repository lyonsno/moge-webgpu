#!/usr/bin/env node
/**
 * Paired browser WebGPU benchmark for comparing two served MoGe checkouts.
 *
 * Usage:
 *   node tools/benchmark_ab.mjs \
 *     --baseline http://127.0.0.1:5188/ --baseline-label main \
 *     --candidate http://127.0.0.1:5190/ --candidate-label fused \
 *     --runs 5 --json
 */

import puppeteer from 'puppeteer-core';
import path from 'path';
import { fileURLToPath } from 'url';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const __dirname = path.dirname(fileURLToPath(import.meta.url));

function parseArgs() {
  const args = process.argv.slice(2);
  const get = (flag, def) => args.includes(flag) ? args[args.indexOf(flag) + 1] : def;
  const baseline = get('--baseline', null);
  const candidate = get('--candidate', null);
  if (!baseline || !candidate) {
    throw new Error('Usage requires --baseline <url> and --candidate <url>');
  }
  return {
    baseline,
    candidate,
    baselineLabel: get('--baseline-label', 'baseline'),
    candidateLabel: get('--candidate-label', 'candidate'),
    runs: parseInt(get('--runs', '5'), 10),
    jsonOnly: args.includes('--json'),
  };
}

function stats(samples) {
  const sorted = [...samples].sort((a, b) => a - b);
  const sum = sorted.reduce((a, b) => a + b, 0);
  return {
    min: sorted[0],
    max: sorted[sorted.length - 1],
    median: sorted[Math.floor(sorted.length / 2)],
    mean: sum / sorted.length,
    samples: sorted,
  };
}

async function runTarget(browser, target, runs) {
  const page = await browser.newPage();
  const consoleLogs = [];
  page.on('console', msg => consoleLogs.push(msg.text()));
  page.on('pageerror', err => console.error(`[${target.label}] PAGE ERROR:`, err.message));

  const testImagePath = path.resolve(__dirname, '..', 'public', 'test_fixtures', 'input.png');
  const result = {
    label: target.label,
    requestedUrl: target.url,
    effectiveUrl: null,
    firstInferenceMs: null,
    warmInferenceMs: [],
    warmPhaseTimings: [],
    gpuInfo: null,
    consoleTail: null,
  };

  try {
    await page.goto(target.url, { waitUntil: 'networkidle0', timeout: 30000 });
    result.effectiveUrl = page.url();
    if (!result.effectiveUrl.startsWith(target.url)) {
      throw new Error(`effective URL ${result.effectiveUrl} does not start with requested ${target.url}`);
    }

    const fileInput = await page.$('#file-input');
    if (!fileInput) throw new Error('file input #file-input not found');

    const firstStart = performance.now();
    await fileInput.uploadFile(testImagePath);
    await page.waitForFunction(
      () => document.getElementById('output')?.classList.contains('visible'),
      { timeout: 300000 }
    );
    const firstEnd = performance.now();

    const firstElapsed = await page.evaluate(() => {
      const dbg = window.__mogeDebug || {};
      return parseFloat(dbg.elapsed) || null;
    });
    result.firstInferenceMs = firstElapsed ? firstElapsed * 1000 : firstEnd - firstStart;

    for (let i = 0; i < runs; i++) {
      const warm = await page.evaluate(async () => {
        const inf = window.__mogeInference;
        if (!inf) throw new Error('MoGeInference not exposed on window');
        const inputCanvas = document.getElementById('input-canvas');
        if (!inputCanvas || inputCanvas.width === 0 || inputCanvas.height === 0) {
          throw new Error('input canvas missing or blank');
        }
        const ctx = inputCanvas.getContext('2d');
        const imageData = ctx.getImageData(0, 0, inputCanvas.width, inputCanvas.height);
        const t0 = performance.now();
        await inf.run(imageData);
        const dt = performance.now() - t0;
        return {
          ms: dt,
          phaseTimings: window.__mogeDebug?.phaseTimings || null,
          outputSize: window.__mogeDebug?.outputSize || null,
          depthRange: window.__mogeDebug?.depthRange || null,
          pointsDiag: window.__mogeDebug?.pointsDiag || null,
        };
      });

      if (!warm.outputSize || !warm.depthRange || !warm.pointsDiag) {
        throw new Error(`warm run ${i + 1} missing output diagnostics`);
      }
      if (warm.pointsDiag.includes('NaN=') && !warm.pointsDiag.includes('NaN=0')) {
        throw new Error(`warm run ${i + 1} reported nonzero NaNs: ${warm.pointsDiag}`);
      }

      result.warmInferenceMs.push(warm.ms);
      result.warmPhaseTimings.push(warm.phaseTimings);
      result.gpuInfo = {
        outputSize: warm.outputSize,
        depthRange: warm.depthRange,
        pointsDiag: warm.pointsDiag,
      };
    }

    result.warmStats = stats(result.warmInferenceMs);
    result.consoleTail = consoleLogs.slice(-20);
    return result;
  } finally {
    await page.close();
  }
}

async function main() {
  const opts = parseArgs();
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: true,
    args: [
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan',
      '--disable-gpu-sandbox',
      '--no-sandbox',
      '--disable-gpu-shader-disk-cache',
      '--window-size=1280,900',
    ],
    defaultViewport: { width: 1280, height: 900 },
  });

  try {
    const baseline = await runTarget(browser, { label: opts.baselineLabel, url: opts.baseline }, opts.runs);
    const candidate = await runTarget(browser, { label: opts.candidateLabel, url: opts.candidate }, opts.runs);
    const deltaMs = candidate.warmStats.median - baseline.warmStats.median;
    const deltaPct = (deltaMs / baseline.warmStats.median) * 100;
    const result = {
      runs: opts.runs,
      baseline,
      candidate,
      delta: {
        medianMs: deltaMs,
        medianPct: deltaPct,
        faster: deltaMs < 0 ? opts.candidateLabel : opts.baselineLabel,
      },
    };

    const json = JSON.stringify(result, null, 2);
    if (opts.jsonOnly) {
      console.log(json);
    } else {
      console.error(`${opts.baselineLabel}: median ${(baseline.warmStats.median / 1000).toFixed(3)}s`);
      console.error(`${opts.candidateLabel}: median ${(candidate.warmStats.median / 1000).toFixed(3)}s`);
      console.error(`Delta: ${deltaMs.toFixed(1)}ms (${deltaPct.toFixed(1)}%)`);
      console.log(json);
    }
  } finally {
    await browser.close();
  }
}

main().catch(err => {
  console.error(err.message);
  process.exit(1);
});
