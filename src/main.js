import './style.css';
import { createIcons, ArrowUpRight, CodeXml, ImagePlus, Scan, Move3d, Sun, RotateCcw } from 'lucide';
import { initGPU } from './lib/gpu.js';
import { MoGeInference } from './lib/inference.js';
import { PhotoViewer } from './lib/photo-viewer.js';

createIcons({ icons: { ArrowUpRight, CodeXml, ImagePlus, Scan, Move3d, Sun, RotateCcw } });
const $ = id => document.getElementById(id);
let viewer, gpu, inference, selected, busy = false, selection = 0, timer;
const sampleUrl = `${import.meta.env.BASE_URL}test_fixtures/input.png`;
const debug = new URLSearchParams(location.search).has('debug');
$('debug-output').hidden = !debug;

function status(message) { $('status').textContent = message; }
function error(err) { $('error').textContent = err.message; $('error').hidden = false; console.error(err); }
function setBusy(value) {
  busy = value;
  for (const id of ['upload-button', 'sample-button', 'run-button', 'file-input']) $(id).disabled = value;
  $('run-button').querySelector('span').textContent = value ? 'Processing' : 'Reveal depth';
  $('progress').hidden = !value;
  $('progress').removeAttribute('value');
}

async function selectImage(blob, name) {
  if (busy) return;
  const ticket = ++selection;
  const bitmap = await createImageBitmap(blob);
  if (ticket !== selection || busy) { bitmap.close(); return; }
  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width; canvas.height = bitmap.height;
  canvas.getContext('2d').drawImage(bitmap, 0, 0); bitmap.close();
  selected = { canvas, name };
  $('selected-image').src = canvas.toDataURL('image/jpeg');
  $('selected-name').textContent = name;
  $('run-button').disabled = false;
  $('error').hidden = true;
  if (!viewer.hasDepth) { viewer.setImage(canvas); $('displayed-name').textContent = name; }
  status(inference ? 'Model loaded' : 'First run downloads model weights (~660 MB)');
}

async function useExample() {
  if (busy) return;
  const response = await fetch(sampleUrl);
  if (!response.ok) throw new Error(`Example photograph could not load (${response.status})`);
  await selectImage(await response.blob(), 'Celebration');
}

async function run() {
  if (busy || !selected) return;
  const input = selected;
  setBusy(true); $('error').hidden = true;
  const start = performance.now();
  window.__mogePhoto = { phase: 'running', name: input.name };
  timer = setInterval(() => { $('elapsed').textContent = `${((performance.now() - start) / 1000).toFixed(1)} s elapsed`; }, 100);
  let phase = 'device';
  try {
    status('Connecting to WebGPU');
    if (!gpu) gpu = await initGPU();
    if (!inference) {
      phase = 'weights'; status('Loading model weights');
      const candidate = new MoGeInference(gpu);
      try {
        await candidate.init((received, total) => {
          status(`Loading weights: ${(received / 1048576).toFixed(0)} MB${total > 0 ? ` / ${(total / 1048576).toFixed(0)} MB` : ''}`);
          if (total > 0) { $('progress').max = total; $('progress').value = received; }
        });
        inference = candidate;
      } catch (err) { await candidate.dispose(); throw err; }
    }
    phase = 'inference'; status('Estimating depth and surface normals'); $('progress').removeAttribute('value');
    const imageData = input.canvas.getContext('2d').getImageData(0, 0, input.canvas.width, input.canvas.height);
    const t0 = performance.now();
    const result = await inference.run(imageData, { scheduler: {
      mode: 'cooperative', splitVitBlocks: true, splitDecoderResBlocks: true, pacing: 'bounded-prefix',
    } });
    const seconds = (performance.now() - t0) / 1000;
    phase = 'surface'; status('Opening depth surface');
    await new Promise(resolve => requestAnimationFrame(() => setTimeout(resolve, 0)));
    viewer.setImage(input.canvas, result);
    viewer.view = 'photo'; $('view').value = 'photo';
    $('zoom').value = 1; $('zoom-value').value = '1.00';
    $('explore-controls').disabled = false;
    $('scene-state').textContent = 'Inferred depth';
    $('displayed-name').textContent = input.name;
    $('result-detail').textContent = `${result.width} × ${result.height} / ${seconds.toFixed(2)} s`;
    status('Depth complete');
    window.__mogeResult = result;
    window.__mogePhoto = { phase: 'complete', seconds, name: input.name, width: result.width, height: result.height };
    if (debug) $('debug-output').textContent = JSON.stringify({ ...window.__mogePhoto, debug: window.__mogeDebug }, null, 2);
  } catch (err) {
    error(err); status(`Stopped during ${phase}`);
    window.__mogePhoto = { phase, error: err.message };
    if (debug) $('debug-output').textContent = JSON.stringify(window.__mogePhoto, null, 2);
  } finally { clearInterval(timer); setBusy(false); }
}

$('upload-button').onclick = () => $('file-input').click();
$('file-input').onchange = () => {
  const file = $('file-input').files[0]; $('file-input').value = '';
  if (file) selectImage(file, file.name).catch(error);
};
$('sample-button').onclick = () => useExample().catch(error);
$('run-button').onclick = run;
const drop = $('drop-zone');
drop.addEventListener('dragover', e => { e.preventDefault(); if (!busy) drop.classList.add('dragover'); });
drop.addEventListener('dragleave', () => drop.classList.remove('dragover'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('dragover');
  if (busy) return;
  const file = e.dataTransfer.files[0];
  if (file) selectImage(file, file.name).catch(error);
});
for (const button of document.querySelectorAll('[data-tool]')) button.onclick = () => {
  viewer.mode = button.dataset.tool;
  for (const other of document.querySelectorAll('[data-tool]')) other.setAttribute('aria-pressed', String(other === button));
};
$('zoom').oninput = () => { viewer.zoom = Number($('zoom').value); $('zoom-value').value = viewer.zoom.toFixed(2); };
$('view').onchange = () => { viewer.view = $('view').value; };
$('reset').onclick = () => { viewer.reset(); $('zoom').value = 1; $('zoom-value').value = '1.00'; };
try {
  viewer = new PhotoViewer($('photo-canvas'));
  await useExample();
  if (!navigator.gpu) { status('WebGPU unavailable in this browser'); $('run-button').disabled = true; }
} catch (err) { error(err); status('Photograph viewer could not start'); }
