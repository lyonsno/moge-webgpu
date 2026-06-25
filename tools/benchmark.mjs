#!/usr/bin/env node
/**
 * Benchmark harness for MoGe-WebGPU.
 *
 * Measures: cold load, warm init, first inference, warm inference (N runs), peak memory.
 * Outputs JSON to stdout and a human-readable summary.
 *
 * Usage:
 *   node tools/benchmark.mjs [--port 5180] [--runs 5] [--headed] [--json]
 */

import puppeteer from 'puppeteer-core';
import path from 'path';
import { fileURLToPath } from 'url';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const __dirname = path.dirname(fileURLToPath(import.meta.url));

function parseArgs() {
  const args = process.argv.slice(2);
  const get = (flag, def) => args.includes(flag) ? args[args.indexOf(flag) + 1] : def;
  return {
    port: get('--port', '5180'),
    runs: parseInt(get('--runs', '5'), 10),
    headed: args.includes('--headed'),
    jsonOnly: args.includes('--json'),
  };
}

function stats(arr) {
  const sorted = [...arr].sort((a, b) => a - b);
  const sum = sorted.reduce((a, b) => a + b, 0);
  return {
    min: sorted[0],
    max: sorted[sorted.length - 1],
    median: sorted[Math.floor(sorted.length / 2)],
    mean: sum / sorted.length,
    samples: sorted,
  };
}

async function main() {
  const opts = parseArgs();
  const url = `http://localhost:${opts.port}/`;

  if (!opts.jsonOnly) console.error(`MoGe-WebGPU benchmark — ${url}, ${opts.runs} warm runs\n`);

  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: !opts.headed,
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

  const results = {
    runtime: 'WebGPU (native compute shaders)',
    precision: 'fp16 weights, fp32 compute',
    model: 'moge-2-vitl-normal',
    tokenGrid: '37x37',
    inputSize: '518x518',
    runs: opts.runs,
    coldLoadMs: null,
    warmInitMs: null,
    firstInferenceMs: null,
    warmInferenceMs: [],
    memoryMB: null,
  };

  const page = await browser.newPage();
  const consoleLogs = [];
  page.on('console', msg => consoleLogs.push(msg.text()));
  page.on('pageerror', err => console.error('PAGE ERROR:', err.message));

  try {
    // --- Cold load ---
    const coldStart = Date.now();
    await page.goto(url, { waitUntil: 'networkidle0', timeout: 30000 });
    results.coldLoadMs = Date.now() - coldStart;
    if (!opts.jsonOnly) console.error(`Cold page load: ${results.coldLoadMs}ms`);

    const testImagePath = path.resolve(__dirname, '..', 'public', 'test_fixtures', 'input.png');

    // --- First inference (includes weight loading + init + first run) ---
    // Inject a benchmark hook into the page
    await page.evaluate(() => {
      window.__benchTimings = {};
    });

    // Upload image to trigger inference
    const fileInput = await page.$('#file-input');
    const firstInfStart = Date.now();
    await fileInput.uploadFile(testImagePath);

    // Wait for inference complete
    await page.waitForFunction(
      () => document.getElementById('output')?.classList.contains('visible'),
      { timeout: 300000 }
    );
    const firstInfEnd = Date.now();

    // Extract timing from page (the page logs elapsed time)
    const firstElapsed = await page.evaluate(() => {
      const dbg = window.__mogeDebug || {};
      return parseFloat(dbg.elapsed) || null;
    });

    results.firstInferenceMs = firstElapsed ? firstElapsed * 1000 : (firstInfEnd - firstInfStart);
    if (!opts.jsonOnly) console.error(`First inference (incl. weight load): ${(results.firstInferenceMs / 1000).toFixed(3)}s`);

    // --- Warm inference runs ---
    // Re-run inference multiple times using the already-initialized pipeline
    for (let i = 0; i < opts.runs; i++) {
      const warmTime = await page.evaluate(async () => {
        const inf = window.__mogeInference;
        if (!inf) throw new Error('MoGeInference not exposed on window');

        // Get the input canvas data
        const inputCanvas = document.getElementById('input-canvas');
        const ctx = inputCanvas.getContext('2d');
        const imageData = ctx.getImageData(0, 0, inputCanvas.width, inputCanvas.height);

        const t0 = performance.now();
        await inf.run(imageData);
        return performance.now() - t0;
      });

      results.warmInferenceMs.push(warmTime);
      if (!opts.jsonOnly) console.error(`  Warm run ${i + 1}: ${(warmTime / 1000).toFixed(3)}s`);
    }

    // --- Memory ---
    const memInfo = await page.evaluate(() => {
      if (performance.memory) {
        return {
          usedJSHeapMB: (performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(1),
          totalJSHeapMB: (performance.memory.totalJSHeapSize / 1024 / 1024).toFixed(1),
        };
      }
      return null;
    });
    results.memoryMB = memInfo;

    // GPU memory (not directly accessible from JS, but we can get buffer sizes from debug info)
    const gpuInfo = await page.evaluate(() => {
      const dbg = window.__mogeDebug || {};
      return {
        outputSize: dbg.outputSize,
        depthRange: dbg.depthRange,
      };
    });
    results.gpuInfo = gpuInfo;

  } catch (err) {
    console.error('BENCHMARK ERROR:', err.message);
    await page.screenshot({ path: '/tmp/moge-benchmark-error.png' });
    results.error = err.message;
  } finally {
    await browser.close();
  }

  // Compute stats
  if (results.warmInferenceMs.length > 0) {
    results.warmStats = stats(results.warmInferenceMs);
  }

  // Output
  const json = JSON.stringify(results, null, 2);
  if (opts.jsonOnly) {
    console.log(json);
  } else {
    console.error('\n--- Results ---');
    console.error(`Cold load:        ${results.coldLoadMs}ms`);
    console.error(`First inference:  ${(results.firstInferenceMs / 1000).toFixed(3)}s (includes weight load + init)`);
    if (results.warmStats) {
      console.error(`Warm inference:   median=${(results.warmStats.median / 1000).toFixed(3)}s, min=${(results.warmStats.min / 1000).toFixed(3)}s, max=${(results.warmStats.max / 1000).toFixed(3)}s`);
    }
    if (results.memoryMB) {
      console.error(`JS Heap:          ${results.memoryMB.usedJSHeapMB} / ${results.memoryMB.totalJSHeapMB} MB`);
    }
    console.log(json);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
