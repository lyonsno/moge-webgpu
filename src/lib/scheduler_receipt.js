/**
 * scheduler_receipt.js — cooperative scheduler config, event capture, and the
 * scheduler verification receipt.
 *
 * Pure JS, no browser/WebGPU/vite dependencies, so the receipt authority logic
 * is unit-testable in plain Node (tools/test_scheduler_receipt_unit.mjs).
 *
 * Authority boundary (adjudicated in the e4f794f scheduler-proof review and
 * preserved here): observed queue/readback stage boundaries are NOT proof of
 * cooperative yielding, and a trace synthesized from stage timings is NOT an
 * observation. `status: "verified"` / `classification: "observed-boundary"`
 * require a genuinely observed event trace (provenance "observed"), and when
 * the scheduler requests yields, observed yield events.
 */

export const SCHEDULER_VERIFICATION_RECEIPT_SCHEMA = 'kaminos.webgpu-scheduler-verification-receipt.v0';
export const SCHEDULER_EVENT_TRACE_SCHEMA = 'kaminos.webgpu-scheduler-event-trace.v0';
export const MOGE_DEPTH_NORMAL_ROUTE_ID = 'moge.depth-normal.webgpu-local.v0';

const OBSERVED_PROVENANCE = 'observed';
const SYNTHESIZED_PROVENANCE = 'synthesized-from-stage-timings';

function cloneJson(value) {
  return value == null ? value : JSON.parse(JSON.stringify(value));
}

// --- Cooperative scheduling (model-owned, aligned to the kit's declared
// moge scheduler profile: per-phase chunked submits with browser yields) ---

export function resolveCooperativeScheduler(requested) {
  if (!requested || requested.mode !== 'cooperative') return null;
  return {
    mode: 'cooperative',
    yieldMs: Math.max(0, Number.isFinite(Number(requested.yieldMs)) ? Number(requested.yieldMs) : 4),
    vitBlockChunkSize: Math.max(1, Math.floor(Number(requested.vitBlockChunkSize) || 1)),
    // Fine granularity: split inside vit blocks (attention/MLP segments) and
    // inside decoder ConvStack levels (per res block), keeping each GPU
    // submission near a frame budget for shared-device hosts.
    splitVitBlocks: requested.splitVitBlocks === true,
    splitDecoderResBlocks: requested.splitDecoderResBlocks === true,
    waitForSubmittedWorkDone: requested.waitForSubmittedWorkDone !== false,
    events: [],
  };
}

export function coopEvent(coop, phase, kind, extra = {}) {
  coop.events.push({
    tMs: performance.now(),
    phase,
    boundary: `moge-stage:${phase}`,
    kind,
    source: 'moge-webgpu-runtime',
    provenance: OBSERVED_PROVENANCE,
    ...extra,
  });
}

export async function coopYield(coop, phase) {
  if (!coop) return;
  coopEvent(coop, phase, 'yield-start');
  await new Promise(resolve => setTimeout(resolve, coop.yieldMs));
  coopEvent(coop, phase, 'yield-end', { yieldMs: coop.yieldMs });
}

/**
 * Scheduler descriptor for the route request. `backboneTotalItems` bounds the
 * effective backbone chunk size: a requested chunk larger than the block count
 * collapses to a single submit, and the effective descriptor must say so
 * rather than echo a granularity the run did not deliver.
 */
export function cooperativeSchedulerDescriptor(coop, { backboneTotalItems } = {}) {
  const requestedChunks = {
    backbone: coop.vitBlockChunkSize,
    'decoder-heads': 1,
    'output-readback': 1,
  };
  const effectiveChunks = {
    ...requestedChunks,
    backbone: Number.isFinite(backboneTotalItems)
      ? Math.min(coop.vitBlockChunkSize, backboneTotalItems)
      : coop.vitBlockChunkSize,
  };
  const base = {
    mode: 'cooperative',
    yieldMs: coop.yieldMs,
    waitForSubmittedWorkDone: coop.waitForSubmittedWorkDone,
  };
  return {
    requestedScheduler: { ...base, phaseChunkSize: requestedChunks },
    effectiveScheduler: { ...base, phaseChunkSize: effectiveChunks, unsupportedFields: [] },
  };
}

export function createMogeSchedulerEventTrace(stagedStages, observedEvents) {
  if (Array.isArray(observedEvents) && observedEvents.length > 0) {
    return {
      schema: SCHEDULER_EVENT_TRACE_SCHEMA,
      clock: 'performance.now',
      timingAuthority: 'queue-submit-wait',
      eventProvenance: OBSERVED_PROVENANCE,
      events: observedEvents.map(event => ({ provenance: OBSERVED_PROVENANCE, ...event })),
    };
  }
  let cursorMs = 0;
  const events = [];
  for (const [index, stage] of (stagedStages || []).entries()) {
    if (!stage?.name || !Number.isFinite(stage.ms)) continue;
    const boundary = `moge-stage:${stage.name}`;
    const waitKind = stage.name === 'output-readback' ? 'readback-wait' : 'queue-work-done';
    events.push({
      tMs: cursorMs,
      phase: stage.name,
      boundary,
      kind: `${waitKind}-start`,
      index,
      source: 'moge-webgpu-runtime',
      provenance: SYNTHESIZED_PROVENANCE,
    });
    cursorMs += Math.max(0, stage.ms);
    events.push({
      tMs: cursorMs,
      phase: stage.name,
      boundary,
      kind: `${waitKind}-end`,
      index,
      waitMs: stage.ms,
      source: 'moge-webgpu-runtime',
      provenance: SYNTHESIZED_PROVENANCE,
    });
  }
  return {
    schema: SCHEDULER_EVENT_TRACE_SCHEMA,
    clock: 'performance.now',
    timingAuthority: stagedStages ? 'queue-submit-wait' : 'not-observed',
    eventProvenance: events.length ? SYNTHESIZED_PROVENANCE : 'none',
    events,
  };
}

export function createMogeBoundaryAssertions(scheduler, events) {
  const requested = scheduler?.requestedScheduler?.phaseChunkSize || {};
  const effective = scheduler?.effectiveScheduler?.phaseChunkSize || {};
  const unsupportedFields = scheduler?.effectiveScheduler?.unsupportedFields || [];
  return Object.entries(requested).map(([phase, requestedValue]) => {
    const field = `phaseChunkSize.${phase}`;
    const boundary = `moge-stage:${phase}`;
    const boundaryEvents = events.filter(event => event.boundary === boundary);
    const observedBoundaryEvents = boundaryEvents.filter(event => event.provenance === OBSERVED_PROVENANCE);
    const unsupported = unsupportedFields.includes(field) || unsupportedFields.includes('phaseChunkSize');
    // Observation counts come only from genuinely observed events; a trace
    // synthesized from stage timings must not masquerade as observation.
    const observedQueueWaitCount = observedBoundaryEvents.filter(event =>
      event.kind === 'queue-work-done-end' || event.kind === 'readback-wait-end'
    ).length;
    const observedYieldCount = observedBoundaryEvents.filter(event => event.kind === 'yield-end').length;
    const observedStart = observedBoundaryEvents.some(event => String(event.kind || '').endsWith('-start'));
    const observedEnd = observedBoundaryEvents.some(event => String(event.kind || '').endsWith('-end'));
    const observedCount = Math.max(observedQueueWaitCount, observedStart && observedEnd ? 1 : 0);
    // A trace synthesized from stage timings carries real queue-submit-wait
    // measurements but is not event observation: it earns "timing-only",
    // never "verified".
    const synthesizedStart = boundaryEvents.some(event =>
      event.provenance !== OBSERVED_PROVENANCE && String(event.kind || '').endsWith('-start'));
    const synthesizedEnd = boundaryEvents.some(event =>
      event.provenance !== OBSERVED_PROVENANCE && String(event.kind || '').endsWith('-end'));
    const status = unsupported
      ? 'unsupported'
      : (observedCount > 0 ? 'verified' : (synthesizedStart && synthesizedEnd ? 'timing-only' : 'unverified'));
    return {
      field,
      requested: requestedValue,
      effective: Number.isFinite(effective[phase]) ? effective[phase] : null,
      status,
      observedBoundary: boundary,
      observedCount,
      expectedMinimumCount: 1,
      observedQueueWaitCount,
      observedYieldCount,
      unsupportedReason: unsupported ? 'effective scheduler declared this field unsupported' : null,
    };
  });
}

export function schedulerRequestsYield(scheduler = {}) {
  const requested = scheduler.requestedScheduler || {};
  const effective = scheduler.effectiveScheduler || {};
  return Number(requested.yieldMs || 0) > 0 || Number(effective.yieldMs || 0) > 0;
}

export function createMogeSchedulerVerificationReceipt({ routeRequest, scheduler, backpressure, stagedStages, observedEvents }) {
  const eventTrace = createMogeSchedulerEventTrace(stagedStages, observedEvents);
  const boundaryAssertions = createMogeBoundaryAssertions(scheduler, eventTrace.events);
  const traceObserved = eventTrace.eventProvenance === OBSERVED_PROVENANCE;
  const yieldObserved = boundaryAssertions.some(assertion => assertion.observedYieldCount > 0);
  const downgrades = [];
  const falseAuthorityChecks = {
    eventTraceMissing: false,
    verifiedWithoutObservedBoundary: false,
    timingProxyOnly: false,
    queueWaitEventsMissing: false,
    boundaryAssertionEventMismatch: false,
    requestedBoundaryAssertionMissing: false,
    requestedFieldDroppedWithoutUnsupported: false,
  };

  if (!eventTrace.events.length) {
    downgrades.push('event-trace-missing');
    falseAuthorityChecks.eventTraceMissing = true;
  }
  if (!traceObserved && eventTrace.events.length) {
    // Stage-timing-derived trace: a timing proxy, never observation authority.
    downgrades.push('event-trace-synthesized');
    falseAuthorityChecks.timingProxyOnly = true;
  }
  if (schedulerRequestsYield(scheduler) && !yieldObserved) downgrades.push('yield-events-missing');

  const requestedPhases = Object.keys(scheduler?.requestedScheduler?.phaseChunkSize || {});
  const verifiedPhases = new Set(
    boundaryAssertions
      .filter(assertion => assertion.status === 'verified')
      .map(assertion => assertion.field.replace(/^phaseChunkSize\./, ''))
  );
  // A timing-only assertion is present evidence (real stage waits), so it does
  // not count as a missing boundary assertion — it just cannot verify.
  const presentPhases = new Set(
    boundaryAssertions
      .filter(assertion => assertion.status === 'verified' || assertion.status === 'timing-only')
      .map(assertion => assertion.field.replace(/^phaseChunkSize\./, ''))
  );
  for (const phase of requestedPhases) {
    if (!presentPhases.has(phase)) {
      downgrades.push('requested-boundary-assertion-missing');
      falseAuthorityChecks.requestedBoundaryAssertionMissing = true;
      break;
    }
  }

  const unsupported = boundaryAssertions.some(assertion => assertion.status === 'unsupported');
  const verified = traceObserved
    && eventTrace.events.length > 0
    && requestedPhases.length > 0
    && requestedPhases.every(phase => verifiedPhases.has(phase))
    && (!schedulerRequestsYield(scheduler) || yieldObserved)
    && !unsupported;
  const status = unsupported ? 'unsupported' : (verified ? 'verified' : 'scheduler-unverified');

  return {
    schema: SCHEDULER_VERIFICATION_RECEIPT_SCHEMA,
    status,
    classification: status === 'verified'
      ? 'observed-boundary'
      : (status === 'unsupported' ? 'unsupported' : 'config-only'),
    observationClass: boundaryAssertions.some(assertion => assertion.status === 'verified')
      ? 'observed-stage-boundary'
      : (boundaryAssertions.some(assertion => assertion.status === 'timing-only') ? 'stage-timing-proxy' : 'none'),
    route: {
      requestedRouteId: routeRequest?.routeId || MOGE_DEPTH_NORMAL_ROUTE_ID,
      effectiveRouteId: MOGE_DEPTH_NORMAL_ROUTE_ID,
      backendClass: 'browser-webgpu',
      requestId: routeRequest?.requestId || null,
    },
    scheduler: cloneJson(scheduler),
    backpressure: cloneJson(backpressure),
    eventTrace,
    boundaryAssertions,
    frameTail: {
      evidenceSource: eventTrace.timingAuthority,
      disclaimer: 'not-gpu-exclusive-or-present-latency',
      rafFps: null,
      frameP95Ms: null,
      queueDoneP95Ms: null,
    },
    downgrades: [...new Set(downgrades)],
    falseAuthorityChecks,
  };
}
