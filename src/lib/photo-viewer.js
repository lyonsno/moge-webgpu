import * as THREE from 'three';
import { buildPhotoSurface } from './photo-surface.js';

const vertexShader = `
  attribute vec3 flatPosition;
  attribute float depthShade;
  uniform float reveal;
  varying vec2 imageUv;
  varying vec3 surfaceNormal;
  varying float depthValue;
  void main() {
    imageUv = uv; surfaceNormal = normal; depthValue = depthShade;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(mix(flatPosition, position, reveal), 1.0);
  }
`;
const fragmentShader = `
  uniform sampler2D photograph;
  uniform vec3 lightDirection;
  uniform float relight;
  uniform int viewMode;
  varying vec2 imageUv;
  varying vec3 surfaceNormal;
  varying float depthValue;
  void main() {
    vec3 n = normalize(surfaceNormal);
    vec3 color = texture2D(photograph, imageUv).rgb;
    if (viewMode == 1) color = mix(vec3(0.93, 0.64, 0.13), vec3(0.035, 0.22, 0.30), depthValue);
    else if (viewMode == 2) color = vec3(n.x, -n.y, -n.z) * 0.5 + 0.5;
    else {
      float light = 0.35 + 0.9 * max(0.0, dot(n, normalize(lightDirection)));
      color *= mix(1.0, light, relight);
    }
    gl_FragColor = vec4(color, 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }
`;

export class PhotoViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setClearColor(0x101313);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(2 * Math.atan(0.5) * 180 / Math.PI, 1, 0.001, 100);
    this.camera.position.z = 1;
    this.target = new THREE.Vector2();
    this.current = new THREE.Vector2();
    this.orbit = new THREE.Vector2();
    this.mode = 'move'; this.view = 'photo'; this.reveal = 0; this.hasDepth = false; this.zoom = 1;
    this.light = new THREE.Vector3(-0.4, 0.4, 1);
    this.reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.onPointerMove = e => {
      const r = canvas.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width * 2 - 1, y = (e.clientY - r.top) / r.height * 2 - 1;
      if (this.mode === 'light') this.light.set(x * 2, -y * 2, 1);
      else if (this.drag) this.orbit.set(
        THREE.MathUtils.clamp(this.drag.orbitX + (e.clientX - this.drag.x) / r.width, -0.65, 0.65),
        THREE.MathUtils.clamp(this.drag.orbitY + (e.clientY - this.drag.y) / r.height, -0.45, 0.45));
      else if (!this.reducedMotion && e.pointerType !== 'touch') this.target.set(x * 0.035, y * 0.025);
    };
    this.onPointerDown = e => {
      this.drag = { x: e.clientX, y: e.clientY, orbitX: this.orbit.x, orbitY: this.orbit.y };
      canvas.setPointerCapture(e.pointerId); this.onPointerMove(e);
    };
    this.onPointerUp = () => { this.drag = null; };
    this.onPointerLeave = () => this.target.set(0, 0);
    this.onKey = e => {
      const delta = { ArrowLeft: [-0.04, 0], ArrowRight: [0.04, 0], ArrowUp: [0, -0.04], ArrowDown: [0, 0.04] }[e.key];
      if (delta) {
        e.preventDefault();
        if (this.mode === 'light') { this.light.x += delta[0] * 4; this.light.y -= delta[1] * 4; }
        else { this.orbit.x = THREE.MathUtils.clamp(this.orbit.x + delta[0], -0.65, 0.65); this.orbit.y = THREE.MathUtils.clamp(this.orbit.y + delta[1], -0.45, 0.45); }
      }
      if (e.key === 'Home') this.reset();
    };
    this.listeners = { pointermove: this.onPointerMove, pointerdown: this.onPointerDown, pointerup: this.onPointerUp,
      pointercancel: this.onPointerUp, lostpointercapture: this.onPointerUp, pointerleave: this.onPointerLeave, keydown: this.onKey };
    for (const [event, handler] of Object.entries(this.listeners)) canvas.addEventListener(event, handler);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas.parentElement);
    this.renderer.setAnimationLoop(() => this.render());
  }

  setImage(source, result = null) {
    const aspect = source.width / source.height;
    const data = result ? buildPhotoSurface(result, aspect) : {
      position: new Float32Array([-aspect / 2, 0.5, 0, aspect / 2, 0.5, 0, -aspect / 2, -0.5, 0, aspect / 2, -0.5, 0]),
      normal: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
      uv: new Float32Array([0, 1, 1, 1, 0, 0, 1, 0]),
      shade: new Float32Array(4), indices: new Uint32Array([0, 2, 1, 1, 2, 3]), far: 10,
    };
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(data.position, 3));
    geometry.setAttribute('flatPosition', new THREE.BufferAttribute(data.flat || data.position, 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(data.normal, 3));
    geometry.setAttribute('uv', new THREE.BufferAttribute(data.uv, 2));
    geometry.setAttribute('depthShade', new THREE.BufferAttribute(data.shade, 1));
    geometry.setIndex(new THREE.BufferAttribute(data.indices, 1));
    const texture = new THREE.CanvasTexture(source);
    texture.colorSpace = THREE.SRGBColorSpace;
    const material = new THREE.ShaderMaterial({ vertexShader, fragmentShader, side: THREE.DoubleSide,
      uniforms: { photograph: { value: texture }, reveal: { value: 0 }, relight: { value: 0 },
        lightDirection: { value: this.light }, viewMode: { value: 0 } } });
    const next = new THREE.Mesh(geometry, material);
    const original = new THREE.Mesh(new THREE.PlaneGeometry(aspect, 1), new THREE.MeshBasicMaterial({ map: texture }));
    original.visible = false;
    next.frustumCulled = false;
    const previous = this.mesh;
    const previousOriginal = this.original;
    this.original = original; this.scene.add(original);
    this.scene.add(next); this.mesh = next; this.aspect = aspect; this.camera.far = data.far;
    this.hasDepth = !!result; this.reveal = this.reducedMotion && result ? 1 : 0;
    this.reset(); this.resize();
    this.openedAt = performance.now();
    if (previous) {
      this.scene.remove(previous); previous.geometry.dispose();
      previous.material.uniforms.photograph.value.dispose(); previous.material.dispose();
    }
    if (previousOriginal) { this.scene.remove(previousOriginal); previousOriginal.geometry.dispose(); previousOriginal.material.dispose(); }
  }

  reset() {
    this.target.set(0, 0); this.current.set(0, 0); this.orbit.set(0, 0); this.zoom = 1; this.light.set(-0.4, 0.4, 1);
  }

  resize() {
    const { width, height } = this.canvas.parentElement.getBoundingClientRect();
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.zoom = Math.min(1, this.camera.aspect / (this.aspect || 1)) * 0.9;
    this.camera.updateProjectionMatrix();
  }

  render() {
    if (!this.mesh || document.hidden) return;
    const original = this.view === 'original';
    this.mesh.visible = !original; this.original.visible = original;
    this.reveal += ((this.hasDepth && !original ? 1 : 0) - this.reveal) * 0.09;
    this.current.lerp(this.target, 0.08);
    const opening = this.hasDepth && !this.reducedMotion ? Math.sin(Math.min(1, (performance.now() - this.openedAt) / 1200) * Math.PI) * 0.025 : 0;
    const x = this.hasDepth && !original ? this.current.x + this.orbit.x + opening : 0;
    const y = this.hasDepth && !original ? this.current.y + this.orbit.y : 0;
    this.camera.position.set(x, -y, original ? 1 : this.zoom); this.camera.lookAt(0, 0, 0);
    this.mesh.material.uniforms.reveal.value = this.reveal;
    this.mesh.material.uniforms.relight.value = this.hasDepth && !original && this.mode === 'light' ? 0.85 : 0;
    this.mesh.material.uniforms.viewMode.value = this.view === 'depth' ? 1 : this.view === 'normals' ? 2 : 0;
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.renderer.setAnimationLoop(null); this.resizeObserver.disconnect();
    for (const [event, handler] of Object.entries(this.listeners)) this.canvas.removeEventListener(event, handler);
    if (this.mesh) { this.mesh.geometry.dispose(); this.mesh.material.uniforms.photograph.value.dispose(); this.mesh.material.dispose(); }
    if (this.original) { this.original.geometry.dispose(); this.original.material.dispose(); }
    this.renderer.dispose();
  }
}
