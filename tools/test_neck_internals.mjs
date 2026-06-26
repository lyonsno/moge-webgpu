#!/usr/bin/env node
/**
 * Browser WebGPU harness that verifies staged neck-level internal profiling.
 *
 * Usage:
 *   node tools/test_neck_internals.mjs --port 5190
 */

import puppeteer from 'puppeteer-core';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const REQUIRED_STAGE_NAMES = [
  'inputAdd',
  'resBlock0',
  'resBlock1',
  'output',
  'resampler',
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
    const result = await page.evaluate(async () => {
      const inputResp = await fetch('/test_fixtures/input.png');
      const blob = await inputResp.blob();
      const bitmap = await createImageBitmap(blob);
      const canvas = document.createElement('canvas');
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(bitmap, 0, 0);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

      const { initGPU } = await import('/src/lib/gpu.js');
      const { MoGeInference } = await import('/src/lib/inference.js');
      const gpu = await initGPU();
      const inf = new MoGeInference(gpu);
      await inf.init();
      await inf.run(imageData, { profileNeckInternals: { level: 1 } });
      return {
        neckInternalTimings: window.__mogeDebug?.neckInternalTimings || null,
        outputSize: window.__mogeDebug?.outputSize || null,
        depthRange: window.__mogeDebug?.depthRange || null,
        pointsDiag: window.__mogeDebug?.pointsDiag || null,
      };
    });

    const timings = result.neckInternalTimings;
    if (!timings) {
      throw new Error('neckInternalTimings missing from window.__mogeDebug');
    }
    if (timings.route !== 'neck-internal-staged-submits') {
      throw new Error(`neckInternalTimings route must be neck-internal-staged-submits, got ${timings.route}`);
    }
    if (timings.level !== 1) {
      throw new Error(`neckInternalTimings level must be 1, got ${timings.level}`);
    }
    if (!Array.isArray(timings.stages)) {
      throw new Error('neckInternalTimings.stages missing');
    }

    const byName = new Map(timings.stages.map(stage => [stage.name, stage]));
    for (const name of REQUIRED_STAGE_NAMES) {
      const stage = byName.get(name);
      if (!stage) throw new Error(`missing stage ${name}`);
      if (!Number.isFinite(stage.submitWaitMs)) throw new Error(`stage ${name} missing submitWaitMs`);
      if (!Array.isArray(stage.shape) || stage.shape.length !== 3) throw new Error(`stage ${name} missing shape`);
    }

    if (!Number.isFinite(timings.totalLevelInternalMs) || timings.totalLevelInternalMs <= 0) {
      throw new Error(`totalLevelInternalMs must be positive, got ${timings.totalLevelInternalMs}`);
    }
    if (!result.outputSize || !result.depthRange || !result.pointsDiag?.includes('NaN=0')) {
      throw new Error(`missing or invalid output diagnostics: ${JSON.stringify(result)}`);
    }

    console.log(JSON.stringify(result, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch(err => {
  console.error(err.message);
  process.exit(1);
});
