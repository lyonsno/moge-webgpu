/**
 * MoGe-WebGPU — Main entry point
 *
 * MoGe-2 (DINOv2 ViT-Large + ConvStack decoder) inference in WebGPU compute.
 * Produces depth, surface normals, and interactive 3D pointcloud from a single image.
 */

import { initGPU } from './lib/gpu.js';
import { MoGeInference } from './lib/inference.js';
import { PointcloudRenderer } from './lib/pointcloud.js';

let gpu = null;
let inference = null;
let pointcloudRenderer = null;

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const statusEl = document.getElementById('status');
const errorEl = document.getElementById('error');
const outputEl = document.getElementById('output');

function setStatus(msg) {
  statusEl.textContent = msg;
  errorEl.style.display = 'none';
}

function setError(msg) {
  errorEl.textContent = msg;
  errorEl.style.display = 'block';
  statusEl.textContent = '';
}

// --- Drop zone ---
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleImage(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleImage(fileInput.files[0]);
});

async function handleImage(file) {
  try {
    setStatus('Initializing WebGPU...');
    if (!gpu) {
      gpu = await initGPU();
    }

    setStatus('Loading image...');
    const bitmap = await createImageBitmap(file);

    // Resize to model input size (224x224 for now)
    const inputSize = 224;
    const inputCanvas = document.getElementById('input-canvas');
    inputCanvas.width = inputSize;
    inputCanvas.height = inputSize;
    const ctx = inputCanvas.getContext('2d');
    ctx.drawImage(bitmap, 0, 0, inputSize, inputSize);
    const imageData = ctx.getImageData(0, 0, inputSize, inputSize);

    if (!inference) {
      inference = new MoGeInference(gpu);
      setStatus('Loading MoGe-2 weights (622 MB, first load only)...');
      await inference.init((received, total) => {
        const mb = (received / 1024 / 1024).toFixed(0);
        const totalMb = (total / 1024 / 1024).toFixed(0);
        setStatus(`Loading weights: ${mb} / ${totalMb} MB`);
      });
    }

    setStatus('Running MoGe-2 inference...');

    const t0 = performance.now();
    const result = await inference.run(imageData);
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);

    setStatus(`Inference complete in ${elapsed}s`);

    // Display depth map
    displayDepthMap(result.depth, inputSize, inputSize);

    // Display normal map
    displayNormalMap(result.normals, inputSize, inputSize);

    // Display pointcloud
    if (!pointcloudRenderer) {
      const pcCanvas = document.getElementById('pointcloud-canvas');
      pointcloudRenderer = new PointcloudRenderer(gpu, pcCanvas);
      await pointcloudRenderer.init();
    }
    pointcloudRenderer.setPoints(result.points, result.colors);

    outputEl.classList.add('visible');
  } catch (e) {
    setError(`Error: ${e.message}`);
    console.error(e);
  }
}

function displayDepthMap(depth, w, h) {
  const canvas = document.getElementById('depth-canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, h);

  // Find min/max for normalization (skip inf/nan)
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < depth.length; i++) {
    const v = depth[i];
    if (isFinite(v)) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  const range = max - min || 1;

  // Turbo-ish colormap
  for (let i = 0; i < depth.length; i++) {
    const v = depth[i];
    const t = isFinite(v) ? (v - min) / range : 0;
    const idx = i * 4;
    // Simple warm colormap: near=warm, far=cool
    img.data[idx + 0] = Math.floor((1 - t) * 255);
    img.data[idx + 1] = Math.floor(Math.sin(t * Math.PI) * 200);
    img.data[idx + 2] = Math.floor(t * 255);
    img.data[idx + 3] = isFinite(v) ? 255 : 0;
  }
  ctx.putImageData(img, 0, 0);
}

function displayNormalMap(normals, w, h) {
  const canvas = document.getElementById('normal-canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, h);

  for (let i = 0; i < w * h; i++) {
    const nx = normals[i * 3 + 0];
    const ny = normals[i * 3 + 1];
    const nz = normals[i * 3 + 2];
    const idx = i * 4;
    // Standard normal map coloring: (n+1)/2 * 255
    img.data[idx + 0] = Math.floor((nx * 0.5 + 0.5) * 255);
    img.data[idx + 1] = Math.floor((ny * 0.5 + 0.5) * 255);
    img.data[idx + 2] = Math.floor((nz * 0.5 + 0.5) * 255);
    img.data[idx + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
