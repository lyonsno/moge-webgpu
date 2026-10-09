import assert from 'node:assert/strict';
import { buildPhotoSurface } from '../src/lib/photo-surface.js';

const fixture = (depth, width = 2, height = 2) => ({
  width, height, depth: Float32Array.from(depth),
  normals: Float32Array.from(depth.flatMap(() => [0, 0, -1])),
});
const input = fixture([1, 1.02, 1.04, 1.06]);
const before = input.depth.slice();
const mesh = buildPhotoSurface(input, 1.5);
assert.equal(mesh.indices.length, 6);
assert.deepEqual(input.depth, before, 'display must not modify model depth');
for (let i = 0; i < 4; i++) {
  const zDistance = 1 - mesh.position[i * 3 + 2];
  assert.ok(Math.abs(mesh.position[i * 3] / zDistance - mesh.flat[i * 3]) < 1e-6, 'home projection x is unchanged');
  assert.ok(Math.abs(mesh.position[i * 3 + 1] / zDistance - mesh.flat[i * 3 + 1]) < 1e-6, 'home projection y is unchanged');
  assert.equal(mesh.normal[i * 3 + 2], 1, 'camera-facing normal becomes viewer-facing');
}
const split = buildPhotoSurface(fixture([1, 1, 4, 4, 1, 1, 4, 4], 4, 2));
assert.equal(split.indices.length, 12, 'two surfaces remain disconnected at a depth edge');
assert.throws(() => buildPhotoSurface(fixture([NaN, 0, Infinity, -1])), /no valid depth/);
assert.throws(() => buildPhotoSurface(fixture([1, 1, 1])), /dimensions/);
assert.throws(() => buildPhotoSurface(fixture([1, 2, 3, 4])), /no connected/);
const masked = buildPhotoSurface(fixture([NaN, 1, 1, 1]));
assert.deepEqual([...masked.indices], [1, 2, 3]);
console.log('Photo surface: projection, normals, discontinuities, invalid output and immutability passed');
