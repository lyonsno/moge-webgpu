#!/usr/bin/env node
/**
 * Browser WebGPU harness that verifies inference records coarse phase timings.
 *
 * Usage:
 *   node tools/test_phase_timings.mjs --port 5190
 */

import puppeteer from 'puppeteer-core';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const REQUIRED_PHASES = [
  'preprocessMs',
  'backboneEncodeMs',
  'neckAndHeadsEncodeMs',
  'gpuReadbackMs',
  'scaleHeadMs',
  'postprocessMs',
  'totalMs',
];

function parseArgs() {
  const args = process.argv.slice(2);
  const get = (flag, def) => args.includes(flag) ? args[args.indexOf(flag) + 1] : def;
  return { port: get('--port', '5190') };
}

async function main() {
  const { port } = parseArgs();
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

  const page = await browser.newPage();
  page.on('pageerror', err => console.error('PAGE ERROR:', err.message));

  try {
    await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'networkidle0', timeout: 30000 });
    const input = await page.$('#file-input');
    await input.uploadFile('public/test_fixtures/input.png');
    await page.waitForFunction(
      () => document.getElementById('output')?.classList.contains('visible'),
      { timeout: 300000 }
    );

    const timings = await page.evaluate(() => window.__mogeDebug?.phaseTimings || null);
    if (!timings) {
      throw new Error('phaseTimings missing from window.__mogeDebug');
    }

    const missing = REQUIRED_PHASES.filter(name => !Number.isFinite(timings[name]));
    if (missing.length > 0) {
      throw new Error(`phaseTimings missing finite phases: ${missing.join(', ')}`);
    }

    if (timings.totalMs <= 0) {
      throw new Error(`phaseTimings totalMs must be positive, got ${timings.totalMs}`);
    }

    console.log(JSON.stringify(timings, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch(err => {
  console.error(err.message);
  process.exit(1);
});
