function makeStubEncoderFeatures(length, random) {
  const features = new Float32Array(length);
  for (let i = 0; i < features.length; i++) {
    features[i] = (random() - 0.5) * 0.5;
  }
  return features;
}

function fixtureShapeInfo(fixture, expectedLength, tokenH, tokenW) {
  return {
    actualLength: fixture?.features?.length ?? null,
    expectedLength,
    fixtureTokenH: Number.isInteger(fixture?.tokenH) ? fixture.tokenH : null,
    fixtureTokenW: Number.isInteger(fixture?.tokenW) ? fixture.tokenW : null,
    expectedTokenH: tokenH,
    expectedTokenW: tokenW,
  };
}

function fixtureMatchesRuntimeGrid(fixture, expectedLength, tokenH, tokenW) {
  if (!(fixture?.features instanceof Float32Array)) return false;
  if (fixture.features.length !== expectedLength) return false;
  if (Number.isInteger(fixture.tokenH) && fixture.tokenH !== tokenH) return false;
  if (Number.isInteger(fixture.tokenW) && fixture.tokenW !== tokenW) return false;
  return true;
}

export function selectCpuFallbackEncoderFeatures({
  fixture,
  encoderDim,
  tokenH,
  tokenW,
  random = Math.random,
} = {}) {
  const expectedLength = encoderDim * tokenH * tokenW;

  if (fixtureMatchesRuntimeGrid(fixture, expectedLength, tokenH, tokenW)) {
    return {
      features: fixture.features,
      clsToken: fixture.clsToken || null,
      source: 'fixture',
      rejectedFixture: null,
    };
  }

  const hasFixture = fixture?.features instanceof Float32Array;
  return {
    features: makeStubEncoderFeatures(expectedLength, random),
    clsToken: null,
    source: hasFixture ? 'stub-shape-mismatch' : 'stub-no-fixture',
    rejectedFixture: hasFixture ? fixtureShapeInfo(fixture, expectedLength, tokenH, tokenW) : null,
  };
}
