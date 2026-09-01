#!/usr/bin/env node
/**
 * Isolate the global depth-scale offset: run the app once (GPU CLS → scale via
 * the app's own JS MLP, logged by inference.js), then run the SAME JS MLP on
 * the PyTorch-dumped CLS token (layer_dumps/cls_token_final.bin).
 *
 * If gpuScale/refClsScale ≈ the observed ~1.147 depth ratio, the offset is
 * encoder CLS drift; if ≈1.0, the bug is elsewhere.
 *
 * Usage: node tools/probe_scale_head.mjs [--port 5183]
 */
import puppeteer from 'puppeteer-core';
import path from 'path';
import { fileURLToPath } from 'url';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const __dirname = path.dirname(fileURLToPath(import.meta.url));

const args = process.argv.slice(2);
const port = args.includes('--port') ? args[args.indexOf('--port') + 1] : '5183';

const browser = await puppeteer.launch({
  executablePath: CHROME_PATH,
  headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--disable-gpu-sandbox', '--no-sandbox'],
});
const page = await browser.newPage();

let gpuRaw = null;
page.on('console', msg => {
  const m = msg.text().match(/Scale head: raw=([-\d.]+)/);
  if (m) gpuRaw = parseFloat(m[1]);
});

await page.goto(`http://localhost:${port}/`, { waitUntil: 'networkidle0', timeout: 60000 });
const fileInput = await page.$('#file-input');
await fileInput.uploadFile(path.resolve(__dirname, '..', 'public', 'test_fixtures', 'input.png'));
await page.waitForFunction(() => window.__mogeResult, { timeout: 300000 });

const refRaw = await page.evaluate(async () => {
  const resp = await fetch('/layer_dumps/cls_token_final.bin');
  if (!resp.ok) return { error: `fetch ${resp.status}` };
  let x = new Float32Array(await resp.arrayBuffer());
  const layers = window.__mogeInference.weights.scaleHead.layers;
  for (let li = 0; li < layers.length; li++) {
    const { weight, bias, inDim, outDim } = layers[li];
    const out = new Float32Array(outDim);
    for (let o = 0; o < outDim; o++) {
      let sum = bias[o];
      for (let i = 0; i < inDim; i++) sum += weight[o * inDim + i] * x[i];
      out[o] = (li < layers.length - 1) ? Math.max(0, sum) : sum;
    }
    x = out;
  }
  return { raw: x[0], clsLen: (await (await fetch('/layer_dumps/cls_token_final.bin')).arrayBuffer()).byteLength / 4 };
});

console.log(JSON.stringify({
  gpuRaw,
  refClsRaw: refRaw.raw ?? refRaw,
  gpuScale: gpuRaw !== null ? Math.exp(gpuRaw) : null,
  refClsScale: refRaw.raw !== undefined ? Math.exp(refRaw.raw) : null,
  ratio: gpuRaw !== null && refRaw.raw !== undefined ? Math.exp(gpuRaw - refRaw.raw) : null,
}, null, 2));

await browser.close();
