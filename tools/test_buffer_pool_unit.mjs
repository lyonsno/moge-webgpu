#!/usr/bin/env node
/**
 * Pure-Node unit tests for the per-device GPU buffer pool (src/lib/gpu.js).
 *
 * Contract:
 *   1. acquire() hands out a buffer of exactly the requested size+usage; a
 *      released buffer of the same size+usage is reused on the next acquire
 *      (steady state: zero new allocations per run).
 *   2. Different sizes or usages never alias.
 *   3. An in-use buffer is never handed out twice before release.
 *   4. releaseAll() returns every in-use buffer to the free list without
 *      destroying it; destroyAll() destroys and empties the pool.
 */
import assert from 'node:assert/strict';
import { createBufferPool } from '../src/lib/gpu.js';

let created = 0;
const mockDevice = {
  createBuffer({ size, usage }) {
    created++;
    return { size, usage, destroyed: false, destroy() { this.destroyed = true; } };
  },
};

const pool = createBufferPool(mockDevice);

// 1. Reuse after release.
const a = pool.acquire(1024, 7);
assert.equal(a.size, 1024);
assert.equal(created, 1);
pool.releaseAll();
const a2 = pool.acquire(1024, 7);
assert.equal(a2, a, 'same size+usage must reuse the released buffer');
assert.equal(created, 1, 'no new allocation in steady state');

// 2. Different size / usage never alias.
const b = pool.acquire(2048, 7);
const c = pool.acquire(1024, 3);
assert.notEqual(b, a2); assert.notEqual(c, a2); assert.notEqual(b, c);
assert.equal(created, 3);

// 3. In-use buffers are never double-issued.
const d = pool.acquire(1024, 7);
assert.notEqual(d, a2, 'in-use buffer must not be handed out again');
assert.equal(created, 4);

// 4. releaseAll keeps buffers alive; destroyAll destroys and empties.
pool.releaseAll();
assert.equal(a.destroyed, false);
const stats = pool.stats();
assert.equal(stats.inUse, 0);
assert.equal(stats.free, 4);
pool.destroyAll();
assert.equal(a.destroyed, true);
assert.equal(pool.stats().free, 0);
const e = pool.acquire(1024, 7);
assert.equal(created, 5, 'after destroyAll a fresh buffer is allocated');
assert.equal(e.destroyed, false);

console.log('buffer pool contracts: PASS');
