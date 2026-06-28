#!/usr/bin/env node
/**
 * Browser WebGPU correctness harness for the optimized stride-2 deconv path.
 *
 * Usage:
 *   node tools/test_conv_transpose2d_stride2.mjs --port 5192
 */

import puppeteer from 'puppeteer-core';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

function parseArgs() {
  const args = process.argv.slice(2);
  const get = (flag, def) => args.includes(flag) ? args[args.indexOf(flag) + 1] : def;
  return { port: get('--port', '5192') };
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
      const { device } = await gpu.initGPU();

      const inC = 3;
      const inH = 3;
      const inW = 4;
      const outC = 4;
      const stride = 2;
      const outH = inH * stride;
      const outW = inW * stride;

      const input = new Float32Array(inC * inH * inW);
      for (let i = 0; i < input.length; i++) {
        input[i] = ((i % 17) - 8) / 7;
      }

      const weight = new Float32Array(inC * outC * 2 * 2);
      for (let i = 0; i < weight.length; i++) {
        weight[i] = ((i % 13) - 6) / 19;
      }

      const bias = new Float32Array([0.25, -0.5, 0.125, -0.0625]);

      const expected = new Float32Array(outC * outH * outW);
      for (let oc = 0; oc < outC; oc++) {
        for (let oy = 0; oy < outH; oy++) {
          const iy = Math.floor(oy / stride);
          const ky = oy % stride;
          for (let ox = 0; ox < outW; ox++) {
            const ix = Math.floor(ox / stride);
            const kx = ox % stride;
            let sum = bias[oc];
            for (let ic = 0; ic < inC; ic++) {
              const inputIdx = ic * inH * inW + iy * inW + ix;
              const weightIdx = ic * outC * 4 + oc * 4 + ky * 2 + kx;
              sum += input[inputIdx] * weight[weightIdx];
            }
            expected[oc * outH * outW + oy * outW + ox] = sum;
          }
        }
      }

      const inputBuf = gpu.createStorageBuffer(device, input);
      const weightBuf = gpu.createStorageBuffer(device, weight);
      const biasBuf = gpu.createStorageBuffer(device, bias);
      const encoder = device.createCommandEncoder();
      const deconv = ops.dispatchConvTranspose2d(device, encoder, inputBuf, weightBuf, biasBuf,
        { inC, inH, inW, outC, stride });
      device.queue.submit([encoder.finish()]);

      if (deconv.kernel !== 'conv_transpose2d_stride2') {
        throw new Error(`expected optimized stride2 kernel metadata, got ${deconv.kernel}`);
      }

      const actual = await gpu.readBuffer(device, deconv.buffer, outC * outH * outW * 4);
      let maxAbsErr = 0;
      let badIndex = -1;
      for (let i = 0; i < expected.length; i++) {
        const err = Math.abs(actual[i] - expected[i]);
        if (!Number.isFinite(actual[i]) || err > maxAbsErr) {
          maxAbsErr = err;
          badIndex = i;
        }
      }

      inputBuf.destroy();
      weightBuf.destroy();
      biasBuf.destroy();
      deconv.buffer.destroy();

      return {
        kernel: deconv.kernel,
        shape: [outC, outH, outW],
        elements: expected.length,
        maxAbsErr,
        badIndex,
        tolerance: 1e-5,
      };
    });

    if (result.maxAbsErr > result.tolerance) {
      throw new Error(`stride2 deconv mismatch: maxAbsErr=${result.maxAbsErr} badIndex=${result.badIndex}`);
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
