/**
 * pointcloud.js — Interactive 3D pointcloud renderer using WebGPU render pass.
 *
 * Renders colored points as instanced billboard quads with:
 *   - Orbit camera (drag to rotate, scroll to zoom)
 *   - Depth testing
 *   - Soft circular point rendering
 */

import pointcloudWGSL from '../shaders/pointcloud.wgsl?raw';
import { createStorageBuffer } from './gpu.js';

export class PointcloudRenderer {
  constructor(gpu, canvas) {
    this.device = gpu.device;
    this.canvas = canvas;
    this.context = null;
    this.pipeline = null;
    this.depthTexture = null;
    this.uniformBuffer = null;

    // Camera state
    this.orbitTheta = Math.PI * 0.25;
    this.orbitPhi = Math.PI * 0.3;
    this.orbitRadius = 5.0;
    this.target = [0, 0, 2];
    this.pointSize = 3.0;

    // Points
    this.pointsBuffer = null;
    this.colorsBuffer = null;
    this.numPoints = 0;

    this._setupInputHandlers();
  }

  async init() {
    const device = this.device;
    const canvas = this.canvas;

    this.context = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device,
      format,
      alphaMode: 'premultiplied',
    });

    const module = device.createShaderModule({ code: pointcloudWGSL });

    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module,
        entryPoint: 'vs_main',
      },
      fragment: {
        module,
        entryPoint: 'fs_main',
        targets: [{
          format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: {
        topology: 'triangle-list',
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });

    this.depthTexture = device.createTexture({
      size: [canvas.width, canvas.height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // Uniform buffer: mat4x4 (64) + vec3 (12) + f32 (4) + u32 (4) = 84 → pad to 96
    this.uniformBuffer = device.createBuffer({
      size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  setPoints(points, colors) {
    if (this.pointsBuffer) this.pointsBuffer.destroy();
    if (this.colorsBuffer) this.colorsBuffer.destroy();

    this.numPoints = points.length / 3;

    // Find center of mass for camera target
    let cx = 0, cy = 0, cz = 0, validCount = 0;
    for (let i = 0; i < this.numPoints; i++) {
      const z = points[i * 3 + 2];
      if (isFinite(z)) {
        cx += points[i * 3 + 0];
        cy += points[i * 3 + 1];
        cz += z;
        validCount++;
      }
    }
    if (validCount > 0) {
      this.target = [cx / validCount, cy / validCount, cz / validCount];
    }

    // Compute bounding radius for auto-zoom
    let maxDist = 0;
    for (let i = 0; i < this.numPoints; i++) {
      const dx = points[i * 3 + 0] - this.target[0];
      const dy = points[i * 3 + 1] - this.target[1];
      const dz = points[i * 3 + 2] - this.target[2];
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (isFinite(d) && d > maxDist) maxDist = d;
    }
    this.orbitRadius = maxDist * 1.5 || 5.0;

    this.pointsBuffer = createStorageBuffer(this.device, points);
    this.colorsBuffer = createStorageBuffer(this.device, colors);

    this._render();
  }

  _setupInputHandlers() {
    let dragging = false;
    let lastX = 0, lastY = 0;

    this.canvas.addEventListener('pointerdown', (e) => {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      this.canvas.setPointerCapture(e.pointerId);
    });

    this.canvas.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;

      this.orbitTheta -= dx * 0.005;
      this.orbitPhi = Math.max(0.1, Math.min(Math.PI - 0.1, this.orbitPhi - dy * 0.005));

      this._render();
    });

    this.canvas.addEventListener('pointerup', () => { dragging = false; });
    this.canvas.addEventListener('pointercancel', () => { dragging = false; });

    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.orbitRadius *= 1 + e.deltaY * 0.001;
      this.orbitRadius = Math.max(0.1, this.orbitRadius);
      this._render();
    }, { passive: false });
  }

  _getViewProjection() {
    const theta = this.orbitTheta;
    const phi = this.orbitPhi;
    const r = this.orbitRadius;

    const camX = this.target[0] + r * Math.sin(phi) * Math.cos(theta);
    const camY = this.target[1] + r * Math.cos(phi);
    const camZ = this.target[2] + r * Math.sin(phi) * Math.sin(theta);

    const view = lookAt([camX, camY, camZ], this.target, [0, -1, 0]);
    const aspect = this.canvas.width / this.canvas.height;
    const proj = perspective(Math.PI / 4, aspect, 0.01, 1000);

    const vp = multiply4x4(proj, view);
    return { matrix: vp, cameraPos: [camX, camY, camZ] };
  }

  _render() {
    if (!this.pointsBuffer || this.numPoints === 0) return;

    const device = this.device;
    const { matrix, cameraPos } = this._getViewProjection();

    // Write uniforms
    const uniformData = new ArrayBuffer(96);
    const f32 = new Float32Array(uniformData);
    const u32 = new Uint32Array(uniformData);
    f32.set(matrix, 0);        // viewProjection mat4x4 at offset 0
    f32[16] = cameraPos[0];    // cameraPos at offset 64
    f32[17] = cameraPos[1];
    f32[18] = cameraPos[2];
    f32[19] = this.pointSize;  // pointSize at offset 76
    u32[20] = this.numPoints;  // numPoints at offset 80

    device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

    const bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.pointsBuffer } },
        { binding: 2, resource: { buffer: this.colorsBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.05, g: 0.05, b: 0.08, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6, this.numPoints); // 6 vertices per quad, N instances
    pass.end();

    device.queue.submit([encoder.finish()]);
  }
}

// --- Matrix math (minimal, no dependencies) ---

function lookAt(eye, center, up) {
  const zAxis = normalize(sub(eye, center));
  const xAxis = normalize(cross(up, zAxis));
  const yAxis = cross(zAxis, xAxis);

  return [
    xAxis[0], yAxis[0], zAxis[0], 0,
    xAxis[1], yAxis[1], zAxis[1], 0,
    xAxis[2], yAxis[2], zAxis[2], 0,
    -dot(xAxis, eye), -dot(yAxis, eye), -dot(zAxis, eye), 1,
  ];
}

function perspective(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2);
  const rangeInv = 1 / (near - far);
  return [
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, far * rangeInv, -1,
    0, 0, near * far * rangeInv, 0,
  ];
}

function multiply4x4(a, b) {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      out[j * 4 + i] = a[0 * 4 + i] * b[j * 4 + 0]
                      + a[1 * 4 + i] * b[j * 4 + 1]
                      + a[2 * 4 + i] * b[j * 4 + 2]
                      + a[3 * 4 + i] * b[j * 4 + 3];
    }
  }
  return out;
}

function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function normalize(v) { const l = Math.sqrt(dot(v, v)); return l > 0 ? [v[0]/l, v[1]/l, v[2]/l] : v; }
