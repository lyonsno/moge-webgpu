#!/usr/bin/env node
/**
 * Browser WebGPU harness that verifies timestamp-query backed GPU phase timings.
 *
 * Usage:
 *   node tools/test_gpu_timestamps.mjs --port 5190
 */

import puppeteer from 'puppeteer-core';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const REQUIRED_PHASES = [
  'backboneGpuMs',
  'neckInputGpuMs',
  'decoderGpuMs',
  'mainPassGpuMs',
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
      await inf.run(imageData, { profileGpuTimestamps: true });
      return {
        adapterFeatures: Array.from(gpu.adapter.features || []),
        deviceFeatures: Array.from(gpu.device.features || []),
        gpuPhaseTimings: window.__mogeDebug?.gpuPhaseTimings || null,
        outputSize: window.__mogeDebug?.outputSize || null,
        depthRange: window.__mogeDebug?.depthRange || null,
        pointsDiag: window.__mogeDebug?.pointsDiag || null,
      };
    });

    if (!result.deviceFeatures.includes('timestamp-query')) {
      throw new Error(`device missing timestamp-query feature: ${result.deviceFeatures.join(', ')}`);
    }

    const timings = result.gpuPhaseTimings;
    if (!timings) {
      throw new Error('gpuPhaseTimings missing from window.__mogeDebug');
    }
    if (timings.route !== 'timestamp-query') {
      throw new Error(`gpuPhaseTimings route must be timestamp-query, got ${timings.route}`);
    }

    const missing = REQUIRED_PHASES.filter(name => !Number.isFinite(timings[name]));
    if (missing.length > 0) {
      throw new Error(`gpuPhaseTimings missing finite phases: ${missing.join(', ')}`);
    }
    if (timings.mainPassGpuMs <= 0) {
      throw new Error(`mainPassGpuMs must be positive, got ${timings.mainPassGpuMs}`);
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
