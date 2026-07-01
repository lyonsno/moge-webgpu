#!/usr/bin/env node
/**
 * Pure checks for CPU fallback encoder feature selection.
 *
 * The live browser receipt smoke must not reinterpret fixture encoder features
 * generated for one token grid as another grid. A mismatched fixture should
 * fall back to an explicitly shaped stub instead of exploding at Float32Array.set.
 */

import assert from 'node:assert/strict';

import {
  selectCpuFallbackEncoderFeatures,
} from '../src/lib/encoder_features.js';

const compatibleFixture = {
  features: new Float32Array(1024 * 37 * 37).fill(0.25),
  tokenH: 37,
  tokenW: 37,
};
const accepted = selectCpuFallbackEncoderFeatures({
  fixture: compatibleFixture,
  encoderDim: 1024,
  tokenH: 37,
  tokenW: 37,
  random: () => 0.5,
});
assert.equal(accepted.source, 'fixture');
assert.equal(accepted.features, compatibleFixture.features);
assert.equal(accepted.rejectedFixture, null);

const mismatchedFixture = {
  features: new Float32Array(1024 * 49 * 49).fill(0.75),
  tokenH: 49,
  tokenW: 49,
};
const rejected = selectCpuFallbackEncoderFeatures({
  fixture: mismatchedFixture,
  encoderDim: 1024,
  tokenH: 37,
  tokenW: 37,
  random: () => 0.5,
});
assert.equal(rejected.source, 'stub-shape-mismatch');
assert.equal(rejected.features.length, 1024 * 37 * 37);
assert.notEqual(rejected.features, mismatchedFixture.features);
assert.deepEqual(rejected.rejectedFixture, {
  actualLength: 1024 * 49 * 49,
  expectedLength: 1024 * 37 * 37,
  fixtureTokenH: 49,
  fixtureTokenW: 49,
  expectedTokenH: 37,
  expectedTokenW: 37,
});

const noFixture = selectCpuFallbackEncoderFeatures({
  fixture: null,
  encoderDim: 4,
  tokenH: 2,
  tokenW: 3,
  random: () => 0.5,
});
assert.equal(noFixture.source, 'stub-no-fixture');
assert.equal(noFixture.features.length, 24);
assert.equal(noFixture.rejectedFixture, null);

console.log('encoder feature shape contracts passed');
