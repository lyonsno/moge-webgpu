#!/usr/bin/env node
/**
 * Smoke test the tomosud/Image_to_Mesh_web competitor.
 *
 * Loads the page, uploads our test image, captures inference timing and evidence.
 *
 * Usage:
 *   node tools/smoke_competitor.mjs [--headed] [--model vitb] [--json]
 */

import puppeteer from 'puppeteer-core';
import path from 'path';
import { fileURLToPath } from 'url';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const COMPETITOR_URL = 'https://tomosud.github.io/Image_to_Mesh_web/';

function parseArgs() {
  const args = process.argv.slice(2);
  const get = (flag, def) => args.includes(flag) ? args[args.indexOf(flag) + 1] : def;
  return {
    headed: args.includes('--headed'),
    model: get('--model', 'vitb'),
    jsonOnly: args.includes('--json'),
    runs: parseInt(get('--runs', '3'), 10),
  };
}

async function main() {
  const opts = parseArgs();

  if (!opts.jsonOnly) console.error(`Competitor smoke — ${COMPETITOR_URL}, model=${opts.model}\n`);

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
    url: COMPETITOR_URL,
    model: opts.model,
    runtime: null,
    executionProvider: null,
    coldLoadMs: null,
    firstInferenceMs: null,
    warmInferenceMs: [],
    resolution: null,
    error: null,
    screenshots: [],
  };

  const page = await browser.newPage();
  const consoleLogs = [];
  page.on('console', msg => consoleLogs.push(msg.text()));
  page.on('pageerror', err => {
    console.error('PAGE ERROR:', err.message);
    consoleLogs.push('ERROR: ' + err.message);
  });

  try {
    // --- Cold load ---
    if (!opts.jsonOnly) console.error('Loading competitor page...');
    const coldStart = Date.now();
    await page.goto(COMPETITOR_URL, { waitUntil: 'networkidle0', timeout: 60000 });
    results.coldLoadMs = Date.now() - coldStart;
    if (!opts.jsonOnly) console.error(`Cold load: ${results.coldLoadMs}ms`);

    // Select the model
    if (opts.model !== 'vitb') {
      await page.select('#modelSelect', opts.model);
      if (!opts.jsonOnly) console.error(`Selected model: ${opts.model}`);
    }

    // Upload our test image
    const testImagePath = path.resolve(__dirname, '..', 'public', 'test_fixtures', 'input.png');
    if (!opts.jsonOnly) console.error('Uploading test image...');

    const fileInput = await page.$('#fileInput');
    const firstInfStart = Date.now();
    await fileInput.uploadFile(testImagePath);

    // Wait for mesh info panel to show resolution (inference complete + mesh built)
    await page.waitForFunction(
      () => {
        const res = document.getElementById('meshResolution');
        return res && res.textContent && res.textContent !== '--';
      },
      { timeout: 600000 }  // 10 min — ViT-L model is 1.32GB
    );

    // Extra wait for rendering to settle
    await new Promise(r => setTimeout(r, 3000));
    results.firstInferenceMs = Date.now() - firstInfStart;

    // Get execution provider and resolution
    const pageInfo = await page.evaluate(() => {
      return {
        provider: document.getElementById('executionProvider')?.textContent || '--',
        resolution: document.getElementById('meshResolution')?.textContent || '--',
      };
    });
    results.executionProvider = pageInfo.provider;
    results.resolution = pageInfo.resolution;
    results.runtime = `ONNXRuntime-Web (${pageInfo.provider})`;

    if (!opts.jsonOnly) {
      console.error(`Execution provider: ${pageInfo.provider}`);
      console.error(`Resolution: ${pageInfo.resolution}`);
      console.error(`First inference (incl. model download): ${(results.firstInferenceMs / 1000).toFixed(1)}s`);
    }

    // Screenshot evidence
    await page.screenshot({ path: '/tmp/competitor-smoke-full.png', fullPage: false });
    results.screenshots.push('/tmp/competitor-smoke-full.png');
    if (!opts.jsonOnly) console.error('Saved /tmp/competitor-smoke-full.png');

    // Warm runs: reload and re-upload (model cached, so this measures warm inference)
    for (let i = 0; i < opts.runs; i++) {
      const warmStart = Date.now();
      await page.reload({ waitUntil: 'networkidle0', timeout: 30000 });

      // Select model again
      if (opts.model !== 'vitb') {
        await page.select('#modelSelect', opts.model);
      }

      // Re-upload image
      const warmInput = await page.$('#fileInput');
      await warmInput.uploadFile(testImagePath);

      // Wait for mesh resolution
      await page.waitForFunction(
        () => {
          const res = document.getElementById('meshResolution');
          return res && res.textContent && res.textContent !== '--';
        },
        { timeout: 600000 }
      );
      await new Promise(r => setTimeout(r, 1000));
      const warmTime = Date.now() - warmStart;

      results.warmInferenceMs.push(warmTime);
      if (!opts.jsonOnly) console.error(`  Warm run ${i + 1}: ${(warmTime / 1000).toFixed(3)}s`);
    }

  } catch (err) {
    console.error('SMOKE ERROR:', err.message);
    await page.screenshot({ path: '/tmp/competitor-smoke-error.png' });
    results.error = err.message;
    results.screenshots.push('/tmp/competitor-smoke-error.png');
  } finally {
    results.consoleLogs = consoleLogs.slice(-50);
    await browser.close();
  }

  const json = JSON.stringify(results, null, 2);
  if (opts.jsonOnly) {
    console.log(json);
  } else {
    console.error('\n--- Competitor Results ---');
    console.error(`Runtime: ${results.runtime}`);
    console.error(`Provider: ${results.executionProvider}`);
    console.error(`First inference: ${(results.firstInferenceMs / 1000).toFixed(1)}s (includes model download)`);
    if (results.warmInferenceMs.length > 0) {
      const sorted = [...results.warmInferenceMs].sort((a, b) => a - b);
      console.error(`Warm inference: median=${(sorted[Math.floor(sorted.length / 2)] / 1000).toFixed(3)}s`);
    }
    console.log(json);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
