#!/usr/bin/env node
/**
 * Browser WebGPU correctness harness for the fused ReLU -> Conv2d kernel.
 *
 * Compares:
 *   reference: dispatchActivation(op=ReLU) -> dispatchConv2d
 *   fused:     dispatchReluConv2d
 *
 * Usage:
 *   node tools/test_fused_relu_conv2d.mjs --port 5189
 */

import puppeteer from 'puppeteer-core';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

function parseArgs() {
  const args = process.argv.slice(2);
  const get = (flag, def) => args.includes(flag) ? args[args.indexOf(flag) + 1] : def;
  return {
    port: get('--port', '5189'),
  };
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
      '--window-size=900,700',
    ],
    defaultViewport: { width: 900, height: 700 },
  });

  const page = await browser.newPage();
  page.on('pageerror', err => console.error('PAGE ERROR:', err.message));

  try {
    await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'networkidle0', timeout: 30000 });
    const result = await page.evaluate(async () => {
      const gpu = await import('/src/lib/gpu.js');
      const ops = await import('/src/lib/shader_ops.js');

      if (typeof ops.dispatchReluConv2d !== 'function') {
        throw new Error('dispatchReluConv2d export missing');
      }

      const { device } = await gpu.initGPU();
      const inC = 2;
      const inH = 4;
      const inW = 5;
      const outC = 3;
      const kH = 3;
      const kW = 3;
      const padH = 1;
      const padW = 1;
      const strideH = 1;
      const strideW = 1;
      const outH = 4;
      const outW = 5;

      const input = new Float32Array(inC * inH * inW);
      for (let i = 0; i < input.length; i++) {
        input[i] = ((i % 11) - 5) / 3;
      }

      const weight = new Float32Array(outC * inC * kH * kW);
      for (let i = 0; i < weight.length; i++) {
        weight[i] = ((i % 7) - 3) / 13;
      }

      const bias = new Float32Array([0.25, -0.5, 0.125]);
      const inputBuf = gpu.createStorageBuffer(device, input);
      const weightBuf = gpu.createStorageBuffer(device, weight);
      const biasBuf = gpu.createStorageBuffer(device, bias);
      const params = { inC, inH, inW, outC, kH, kW, padH, padW, strideH, strideW };

      const encoder = device.createCommandEncoder();
      const reluBuf = ops.dispatchActivation(device, encoder, inputBuf, null, input.length, 0);
      const reference = ops.dispatchConv2d(device, encoder, reluBuf, weightBuf, biasBuf, params);
      const fused = ops.dispatchReluConv2d(device, encoder, inputBuf, weightBuf, biasBuf, params);
      device.queue.submit([encoder.finish()]);

      const [referenceData, fusedData] = await Promise.all([
        gpu.readBuffer(device, reference.buffer, outC * outH * outW * 4),
        gpu.readBuffer(device, fused.buffer, outC * outH * outW * 4),
      ]);

      let maxAbsErr = 0;
      let badIndex = -1;
      for (let i = 0; i < referenceData.length; i++) {
        const err = Math.abs(referenceData[i] - fusedData[i]);
        if (!Number.isFinite(fusedData[i]) || err > maxAbsErr) {
          maxAbsErr = err;
          badIndex = i;
        }
      }

      inputBuf.destroy();
      weightBuf.destroy();
      biasBuf.destroy();
      reluBuf.destroy();
      reference.buffer.destroy();
      fused.buffer.destroy();

      return {
        shape: [outC, outH, outW],
        elements: referenceData.length,
        maxAbsErr,
        badIndex,
        tolerance: 1e-5,
      };
    });

    if (result.maxAbsErr > result.tolerance) {
      throw new Error(`fused ReLU conv mismatch: maxAbsErr=${result.maxAbsErr} badIndex=${result.badIndex}`);
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
