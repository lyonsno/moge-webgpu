#!/usr/bin/env node
/**
 * Automated backbone comparison harness.
 * Launches Chrome with WebGPU, loads the app, runs backbone comparison,
 * and prints structured results to stdout.
 *
 * Usage:
 *   node tools/compare_backbone.mjs [--port 5181] [--headed]
 *
 * Requires: puppeteer-core, Chrome installed, vite dev server running.
 */

import puppeteer from 'puppeteer-core';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

async function main() {
  const args = process.argv.slice(2);
  const port = args.includes('--port') ? args[args.indexOf('--port') + 1] : '5181';
  const headed = args.includes('--headed');
  const url = `http://localhost:${port}/test.html`;

  console.log(`Backbone comparison — ${url}\n`);

  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: !headed,
    args: [
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan',
      '--disable-gpu-sandbox',
      '--no-sandbox',
      '--disable-gpu-shader-disk-cache',
      '--gpu-no-context-lost',
    ],
  });

  const page = await browser.newPage();

  // Stream console output
  const logs = [];
  page.on('console', msg => {
    const text = msg.text();
    logs.push(text);
    if (text.includes('===') || text.includes('---') || text.includes('maxErr') ||
        text.includes('GPU only') || text.includes('patch_embed') || text.includes('b0_') ||
        text.includes('block_') || text.includes('Loaded') || text.includes('ERROR') ||
        text.includes('Weights') || text.includes('Running') || text.includes('Done')) {
      console.log(text);
    }
  });

  page.on('pageerror', err => console.error('PAGE ERROR:', err.message));

  try {
    console.log('Loading test page...');
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });

    // Wait for comparison to complete (signaled by title change)
    console.log('Waiting for weights + comparison (may take a couple minutes)...\n');
    await page.waitForFunction(
      () => document.title === 'DONE' || document.title === 'ERROR',
      { timeout: 300000 }
    );

    const error = await page.evaluate(() => window.__compareError);
    if (error) {
      console.error('Comparison failed:', error);
      process.exit(1);
    }

    const results = await page.evaluate(() => window.__compareResults);

    // Print summary table
    console.log('\n\n════════════════════════════════════════════');
    console.log('  SUMMARY');
    console.log('════════════════════════════════════════════');
    console.log(`${'Stage'.padEnd(20)} ${'maxErr'.padStart(10)} ${'rmsErr'.padStart(10)} ${'relStd'.padStart(8)} ${'GPU std'.padStart(10)} ${'REF std'.padStart(10)}`);
    console.log('─'.repeat(70));
    for (const [key, r] of Object.entries(results)) {
      if (!r.maxErr && r.maxErr !== 0) {
        console.log(`${key.padEnd(20)} (no ref)`);
        continue;
      }
      const flag = r.maxErr > 0.1 ? ' ⚠️' : r.maxErr > 0.01 ? ' ⚡' : ' ✓';
      console.log(
        `${key.padEnd(20)} ${r.maxErr.toFixed(4).padStart(10)} ${r.rmsErr.toFixed(4).padStart(10)} ${r.relStd.toFixed(4).padStart(8)} ${r.gpu.std.toFixed(4).padStart(10)} ${r.ref.std.toFixed(4).padStart(10)}${flag}`
      );
    }
    console.log('─'.repeat(70));
    console.log('  ✓ = maxErr < 0.01   ⚡ = maxErr < 0.1   ⚠️  = maxErr >= 0.1\n');

    // Write results JSON
    const fs = await import('fs');
    const outPath = 'tools/compare_results.json';
    fs.writeFileSync(outPath, JSON.stringify(results, null, 2));
    console.log(`Results written to ${outPath}`);

  } catch (err) {
    console.error('ERROR:', err.message);
    // Dump any collected logs
    if (logs.length > 0) {
      console.log('\n--- Console logs collected before error ---');
      logs.forEach(l => console.log('  ', l));
    }
    process.exit(1);
  } finally {
    await browser.close();
  }
}

main();
