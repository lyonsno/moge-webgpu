#!/usr/bin/env node
/**
 * Visual smoke test: load main page, drop test image, capture depth/normal output.
 *
 * Usage:
 *   node tools/visual_smoke.mjs [--port 5180] [--headed]
 *
 * Outputs: /tmp/moge-smoke-depth.png, /tmp/moge-smoke-normals.png, /tmp/moge-smoke-pointcloud.png
 */

import puppeteer from 'puppeteer-core';
import path from 'path';
import { fileURLToPath } from 'url';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function main() {
  const args = process.argv.slice(2);
  const port = args.includes('--port') ? args[args.indexOf('--port') + 1] : '5180';
  const headed = args.includes('--headed');
  const url = `http://localhost:${port}/`;

  console.log(`Visual smoke test — ${url}\n`);

  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: !headed,
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

  // Collect console output
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('Scale head') || text.includes('Backbone') ||
        text.includes('Points raw') || text.includes('depth') ||
        text.includes('ERROR') || text.includes('Loaded') ||
        text.includes('Weights') || text.includes('metric')) {
      console.log(`  [page] ${text}`);
    }
  });
  page.on('pageerror', err => console.error('PAGE ERROR:', err.message));

  try {
    console.log('Loading main page...');
    await page.goto(url, { waitUntil: 'networkidle0', timeout: 30000 });

    // Programmatically trigger the file input with the test image
    console.log('Triggering inference with test image...');

    // Upload the test image via the file input
    const testImagePath = path.resolve(__dirname, '..', 'public', 'test_fixtures', 'input.png');
    const fileInput = await page.$('#file-input');
    await fileInput.uploadFile(testImagePath);

    // Wait for inference to complete by watching for the output panel to become visible
    console.log('Waiting for inference to complete...');
    await page.waitForFunction(
      () => document.getElementById('output')?.classList.contains('visible'),
      { timeout: 300000 }
    );

    console.log('Inference done. Capturing screenshots...');

    // Wait a moment for rendering to settle
    await new Promise(r => setTimeout(r, 2000));

    // Capture depth canvas
    const depthEl = await page.$('#depth-canvas');
    if (depthEl) {
      await depthEl.screenshot({ path: '/tmp/moge-smoke-depth.png' });
      console.log('Saved /tmp/moge-smoke-depth.png');
    }

    // Capture normals canvas
    const normalEl = await page.$('#normal-canvas');
    if (normalEl) {
      await normalEl.screenshot({ path: '/tmp/moge-smoke-normals.png' });
      console.log('Saved /tmp/moge-smoke-normals.png');
    }

    // Full page screenshot
    await page.screenshot({ path: '/tmp/moge-smoke-full.png', fullPage: true });
    console.log('Saved /tmp/moge-smoke-full.png');

    // Get metrics
    const metrics = await page.evaluate(() => ({
      debug: window.__mogeDebug,
    }));
    console.log('\nMetrics:', JSON.stringify(metrics, null, 2));

  } catch (err) {
    console.error('ERROR:', err.message);
    await page.screenshot({ path: '/tmp/moge-smoke-error.png' });
    console.log('Error screenshot saved to /tmp/moge-smoke-error.png');
    process.exit(1);
  } finally {
    await browser.close();
  }
}

main();
