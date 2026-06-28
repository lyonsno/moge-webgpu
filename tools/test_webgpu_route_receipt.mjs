#!/usr/bin/env node
/**
 * Browser WebGPU harness that verifies MoGE emits a Kaminos-consumable
 * webgpu-local route receipt from live inference.
 *
 * Usage:
 *   node tools/test_webgpu_route_receipt.mjs --port 5195
 */

import puppeteer from 'puppeteer-core';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

function parseArgs() {
  const args = process.argv.slice(2);
  const get = (flag, def) => args.includes(flag) ? args[args.indexOf(flag) + 1] : def;
  return { port: get('--port', '5195') };
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
      await inf.run(imageData, {
        profileStagedGpu: true,
        routeReceipt: {
          sourceArtifact: {
            artifactId: 'image:test-fixture-input',
            sha256: 'sha256:test-fixture-input',
            shape: [imageData.height, imageData.width, 4],
          },
          outputs: {
            depth: { artifactId: 'depth:test-fixture-input', sha256: 'sha256:depth-output' },
            normal: { artifactId: 'normal:test-fixture-input', sha256: 'sha256:normal-output' },
            pointMap: { artifactId: 'pointmap:test-fixture-input', sha256: 'sha256:pointmap-output' },
          },
          model: {
            revision: 'local-moge-2-vitl-normal',
            weightsHash: 'sha256:weights-fixture',
            dtype: 'fp16',
          },
          kernel: {
            profile: 'conv-transpose2d-stride2',
            commit: '003763d',
          },
        },
      });

      return {
        receipt: window.__mogeDebug?.webGpuRouteReceipt || null,
        outputSize: window.__mogeDebug?.outputSize || null,
        depthRange: window.__mogeDebug?.depthRange || null,
        pointsDiag: window.__mogeDebug?.pointsDiag || null,
      };
    });

    const receipt = result.receipt;
    if (!receipt) throw new Error('webGpuRouteReceipt missing from window.__mogeDebug');
    if (receipt.schema !== 'kaminos.webgpu-route-receipt.v0') throw new Error(`bad receipt schema ${receipt.schema}`);
    if (receipt.requestedRouteId !== 'moge.depth-normal.webgpu-local.v0') throw new Error(`bad requestedRouteId ${receipt.requestedRouteId}`);
    if (receipt.effectiveRouteId !== 'moge.depth-normal.webgpu-local.v0') throw new Error(`bad effectiveRouteId ${receipt.effectiveRouteId}`);
    if (receipt.status !== 'real') throw new Error(`receipt status must be real, got ${receipt.status}`);
    if (receipt.backend?.kind !== 'webgpu-local') throw new Error(`bad backend kind ${receipt.backend?.kind}`);
    if (receipt.backend?.runtime !== 'browser') throw new Error(`bad backend runtime ${receipt.backend?.runtime}`);
    if (!receipt.backend?.adapterName) throw new Error('receipt missing backend.adapterName');
    if (!Array.isArray(receipt.backend?.features) || receipt.backend.features.length === 0) throw new Error('receipt missing backend.features');
    if (receipt.model?.id !== 'Ruicheng/moge-2-vitl-normal') throw new Error(`bad model id ${receipt.model?.id}`);
    if (receipt.model?.weightsHash !== 'sha256:weights-fixture') throw new Error('receipt did not preserve model weightsHash');
    if (receipt.kernel?.profile !== 'conv-transpose2d-stride2') throw new Error(`bad kernel profile ${receipt.kernel?.profile}`);
    if (receipt.timings?.source !== 'queue-submit-wait') throw new Error(`bad timing source ${receipt.timings?.source}`);
    if (!Number.isFinite(receipt.timings?.totalMs) || receipt.timings.totalMs <= 0) throw new Error('receipt missing positive totalMs');
    const roles = new Set((receipt.outputs || []).map(output => output.role));
    for (const role of ['depth', 'normal', 'pointmap']) {
      if (!roles.has(role)) throw new Error(`receipt missing output role ${role}`);
    }
    if (!result.outputSize || !result.depthRange || !result.pointsDiag?.includes('NaN=0')) {
      throw new Error(`missing output diagnostics: ${JSON.stringify(result)}`);
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
