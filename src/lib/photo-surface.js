// A depth-image surface, with an illustrative pinhole camera (not calibrated intrinsics).
// Its projection is the original photograph at the home camera position.
export function buildPhotoSurface({ depth, normals, width, height }, aspect = width / height) {
  const count = width * height;
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 2 || height < 2 ||
      depth?.length !== count || normals?.length !== count * 3 || !(aspect > 0) || !Number.isFinite(aspect)) {
    throw new Error('Invalid depth surface dimensions');
  }
  const finite = Array.from(depth).filter(d => Number.isFinite(d) && d > 0).sort((a, b) => a - b);
  if (!finite.length) throw new Error('The model returned no valid depth');
  const median = finite[Math.floor(finite.length / 2)];
  const low = finite[Math.floor(finite.length * 0.02)];
  const high = finite[Math.floor(finite.length * 0.98)];
  const position = new Float32Array(count * 3);
  const flat = new Float32Array(count * 3);
  const normal = new Float32Array(count * 3);
  const uv = new Float32Array(count * 2);
  const shade = new Float32Array(count);
  const valid = new Uint8Array(count);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const u = x / (width - 1), v = y / (height - 1);
      const good = Number.isFinite(depth[i]) && depth[i] > 0;
      const d = good ? depth[i] / median : 1;
      valid[i] = good ? 1 : 0;
      flat.set([(u - 0.5) * aspect, 0.5 - v, 0], i * 3);
      position.set([(u - 0.5) * aspect * d, (0.5 - v) * d, 1 - d], i * 3);
      uv.set([u, 1 - v], i * 2);
      // MoGe camera coordinates are x-right, y-down, z-forward.
      const nx = normals[i * 3], ny = -normals[i * 3 + 1], nz = -normals[i * 3 + 2];
      const length = Math.hypot(nx, ny, nz);
      normal.set(Number.isFinite(length) && length > 0 ? [nx / length, ny / length, nz / length] : [0, 0, 1], i * 3);
      shade[i] = good ? Math.max(0, Math.min(1, (depth[i] - low) / (high - low || 1))) : 0;
    }
  }
  const indices = [];
  const triangle = (a, b, c) => {
    if (!valid[a] || !valid[b] || !valid[c]) return;
    const lo = Math.min(depth[a], depth[b], depth[c]);
    const hi = Math.max(depth[a], depth[b], depth[c]);
    // Display-only discontinuity rejection; all model values remain untouched.
    if ((hi - lo) / lo > 0.08) return;
    indices.push(a, b, c);
  };
  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width - 1; x++) {
      const a = y * width + x, b = a + 1, c = a + width, d = c + 1;
      triangle(a, c, b);
      triangle(b, c, d);
    }
  }
  if (!indices.length) throw new Error('The model returned no connected depth surface');
  return { position, flat, normal, uv, shade, indices: new Uint32Array(indices), median, far: Math.max(10, finite.at(-1) / median * 2) };
}
