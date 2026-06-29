export const MOGE_DEPTH_NORMAL_ROUTE_ID = 'moge.depth-normal.webgpu-local.v0';
export const MOGE_ROUTE_REQUEST_SCHEMA = 'kaminos.webgpu-route-request.v0';
export const MOGE_ROUTE_RESULT_SCHEMA = 'kaminos.webgpu-route-result.v0';

function clone(value) {
  return value == null ? value : JSON.parse(JSON.stringify(value));
}

function isNonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

function requireString(errors, value, path) {
  if (!isNonEmptyString(value)) errors.push(`${path} must be a non-empty string`);
}

function sourceArtifactFrom(routeReceipt) {
  const source = routeReceipt?.sourceArtifact || {};
  return {
    role: 'source-image',
    artifactId: source.artifactId || 'runtime:browser-imagedata',
    sha256: source.sha256 || null,
    hashStatus: source.sha256 ? 'provided' : 'not-hashed-browser-runtime',
    shape: Array.isArray(source.shape) ? [...source.shape] : null,
  };
}

function outputArtifactFrom(role, artifact, shape) {
  return {
    role,
    artifactId: artifact?.artifactId || `runtime:${role}`,
    sha256: artifact?.sha256 || null,
    hashStatus: artifact?.sha256 ? 'provided' : 'not-hashed-browser-runtime',
    shape,
  };
}

function outputArtifactsFrom(routeReceipt, outH, outW) {
  const outputs = routeReceipt?.outputs || {};
  return [
    outputArtifactFrom('depth', outputs.depth, [outH, outW]),
    outputArtifactFrom('normal', outputs.normal, [3, outH, outW]),
    outputArtifactFrom('pointmap', outputs.pointMap, [3, outH, outW]),
  ];
}

export function createMogeRouteInvocationRequest({ routeReceipt, outH, outW, requestId } = {}) {
  const now = new Date().toISOString();
  return {
    schema: MOGE_ROUTE_REQUEST_SCHEMA,
    requestId: requestId || routeReceipt?.requestId || `moge-depth-normal:${now}`,
    routeId: MOGE_DEPTH_NORMAL_ROUTE_ID,
    backendKind: 'webgpu-local',
    inputs: [sourceArtifactFrom(routeReceipt)],
    outputs: outputArtifactsFrom(routeReceipt, outH, outW).map(output => ({
      role: output.role,
      artifactId: output.artifactId,
      sha256: output.sha256,
      hashStatus: output.hashStatus,
      shape: output.shape,
    })),
    routeConfig: {
      timingSource: routeReceipt?.timingSource || 'queue-submit-wait',
      profileStagedGpu: routeReceipt?.profileStagedGpu ?? null,
    },
    model: clone(routeReceipt?.model || {}),
    kernel: clone(routeReceipt?.kernel || {}),
    createdAt: now,
  };
}

export function validateMogeRouteInvocationRequest(request) {
  const errors = [];
  if (!request || typeof request !== 'object') return { ok: false, errors: ['request must be an object'] };
  if (request.schema !== MOGE_ROUTE_REQUEST_SCHEMA) errors.push(`schema must be ${MOGE_ROUTE_REQUEST_SCHEMA}`);
  requireString(errors, request.requestId, 'requestId');
  if (request.routeId !== MOGE_DEPTH_NORMAL_ROUTE_ID) errors.push(`routeId must be ${MOGE_DEPTH_NORMAL_ROUTE_ID}`);
  if (request.backendKind !== 'webgpu-local') errors.push('backendKind must be webgpu-local');
  validateArtifacts(errors, request.inputs, 'inputs', ['source-image'], { requireHash: true });
  validateArtifacts(errors, request.outputs, 'outputs', ['depth', 'normal', 'pointmap'], { requireHash: false });
  return { ok: errors.length === 0, errors };
}

function validateArtifacts(errors, artifacts, path, allowedRoles, { requireHash }) {
  const allowed = new Set(allowedRoles);
  const seen = new Set();
  if (!Array.isArray(artifacts) || artifacts.length === 0) {
    errors.push(`${path} must be a non-empty array`);
    return;
  }
  artifacts.forEach((artifact, index) => {
    const artifactPath = `${path}[${index}]`;
    requireString(errors, artifact?.role, `${artifactPath}.role`);
    if (artifact?.role) {
      if (!allowed.has(artifact.role)) errors.push(`${artifactPath}.role is not defined by route`);
      if (seen.has(artifact.role)) errors.push(`${artifactPath}.role duplicates ${artifact.role}`);
      seen.add(artifact.role);
    }
    requireString(errors, artifact?.artifactId, `${artifactPath}.artifactId`);
    if (requireHash) requireString(errors, artifact?.sha256, `${artifactPath}.sha256`);
    if (artifact?.shape != null && (!Array.isArray(artifact.shape) || !artifact.shape.every(Number.isInteger))) {
      errors.push(`${artifactPath}.shape must be an integer array when present`);
    }
  });
  for (const role of allowedRoles) {
    if (!seen.has(role)) errors.push(`${path} missing role ${role}`);
  }
}

function validateRouteReceipt(errors, receipt) {
  if (!receipt || typeof receipt !== 'object') {
    errors.push('receipt must be an object');
    return;
  }
  if (receipt.schema !== 'kaminos.webgpu-route-receipt.v0') errors.push('receipt.schema must be kaminos.webgpu-route-receipt.v0');
  if (receipt.requestedRouteId !== MOGE_DEPTH_NORMAL_ROUTE_ID) errors.push(`receipt.requestedRouteId must be ${MOGE_DEPTH_NORMAL_ROUTE_ID}`);
  if (receipt.effectiveRouteId !== MOGE_DEPTH_NORMAL_ROUTE_ID) errors.push(`receipt.effectiveRouteId must be ${MOGE_DEPTH_NORMAL_ROUTE_ID}`);
  if (receipt.status !== 'real') errors.push(`receipt.status must be real for authoritative result, got ${receipt.status}`);
  if (receipt.fallbackReason) errors.push(`receipt.fallbackReason must be empty for authoritative result, got ${receipt.fallbackReason}`);
  if (receipt.backend?.kind !== 'webgpu-local') errors.push('receipt.backend.kind must be webgpu-local');
  if (receipt.backend?.runtime !== 'browser') errors.push('receipt.backend.runtime must be browser');
  requireString(errors, receipt.backend?.adapterName, 'receipt.backend.adapterName');
  if (!Array.isArray(receipt.backend?.features) || receipt.backend.features.length === 0) {
    errors.push('receipt.backend.features must be a non-empty array');
  }
  requireString(errors, receipt.model?.id, 'receipt.model.id');
  requireString(errors, receipt.model?.revision, 'receipt.model.revision');
  requireString(errors, receipt.model?.weightsHash, 'receipt.model.weightsHash');
  requireString(errors, receipt.model?.dtype, 'receipt.model.dtype');
  requireString(errors, receipt.kernel?.profile, 'receipt.kernel.profile');
  validateArtifacts(errors, receipt.inputs, 'receipt.inputs', ['source-image'], { requireHash: true });
  validateArtifacts(errors, receipt.outputs, 'receipt.outputs', ['depth', 'normal', 'pointmap'], { requireHash: true });
  for (const output of receipt.outputs || []) {
    if (output.status !== 'real') errors.push(`receipt.outputs.${output.role}.status must be real, got ${output.status}`);
  }
  requireString(errors, receipt.timings?.source, 'receipt.timings.source');
  if (!Number.isFinite(receipt.timings?.totalMs) || receipt.timings.totalMs <= 0) {
    errors.push('receipt.timings.totalMs must be a positive finite number');
  }
  if (!Array.isArray(receipt.timings?.stages) || receipt.timings.stages.length === 0) {
    errors.push('receipt.timings.stages must be a non-empty array');
  }
}

export function validateMogeRouteWorkerResult(result) {
  const errors = [];
  if (!result || typeof result !== 'object') return { ok: false, errors: ['result must be an object'] };
  if (result.schema !== MOGE_ROUTE_RESULT_SCHEMA) errors.push(`schema must be ${MOGE_ROUTE_RESULT_SCHEMA}`);
  requireString(errors, result.requestId, 'requestId');
  if (result.routeId !== MOGE_DEPTH_NORMAL_ROUTE_ID) errors.push(`routeId must be ${MOGE_DEPTH_NORMAL_ROUTE_ID}`);
  const requestResult = validateMogeRouteInvocationRequest(result.request);
  if (!requestResult.ok) errors.push(...requestResult.errors.map(error => `request.${error}`));
  validateRouteReceipt(errors, result.receipt);
  validateArtifacts(errors, result.outputs, 'outputs', ['depth', 'normal', 'pointmap'], { requireHash: true });
  return { ok: errors.length === 0, errors };
}

export function createMogeRouteWorkerResult({ request, receipt } = {}) {
  const result = {
    schema: MOGE_ROUTE_RESULT_SCHEMA,
    requestId: request?.requestId || null,
    routeId: MOGE_DEPTH_NORMAL_ROUTE_ID,
    status: receipt?.status || 'unknown',
    request: clone(request),
    receipt: clone(receipt),
    backend: clone(receipt?.backend || null),
    outputs: clone(receipt?.outputs || []),
    timings: clone(receipt?.timings || null),
    createdAt: new Date().toISOString(),
  };
  const validation = validateMogeRouteWorkerResult(result);
  return {
    ...result,
    validation,
    authoritative: validation.ok && result.status === 'real',
  };
}
