// Pure replay over hitch-report.raw. Temporal overlap is not a causal claim.
import { isDeepStrictEqual } from 'node:util';
import { validateWebGpuBackendIdentity, validateSchedulerVerificationReceipt } from '@kaminos/webgpu-inference-kit';
import { MOGE_DEPTH_NORMAL_ROUTE_ID, MOGE_ROUTE_RESULT_SCHEMA, MOGE_ROUTE_RECEIPT_SCHEMA, MOGE_ROUTE_REQUEST_SCHEMA } from '../src/lib/route_boundary.js';
export function analyzeHitches(data, hitchMs = 50) {
  const errors = [];
  const result = data.routeResult, receipt = result?.receipt, request = result?.request;
  const verification = receipt?.runtime?.schedulerVerification;
  const trace = verification?.eventTrace, events = trace?.events ?? [], frameTimes = data.frameTimes ?? [];
  if (!/^done\b/i.test(data.terminalStatus ?? '') || data.observationError) errors.push('Inference did not complete');
  errors.push(...validateSchedulerVerificationReceipt(verification ?? {}).errors.map(e=>`scheduler.${e}`));
  if (verification?.status !== 'verified' || verification?.classification !== 'observed-boundary') errors.push('Current scheduler receipt is not verified observed timing');
  // Old recordings copied these fields; new recordings have one source only.
  if ((data.schedStatus !== undefined && data.schedStatus !== verification?.status)
      || (data.eventTrace !== undefined && !isDeepStrictEqual(data.eventTrace, trace))) errors.push('Detached scheduler capture contradicts nested receipt');
  if (request?.schema !== MOGE_ROUTE_REQUEST_SCHEMA || request?.backendKind !== 'webgpu-local'
      || [result?.routeId, request?.routeId, verification?.route?.requestedRouteId,
        verification?.route?.effectiveRouteId].some(id=>id !== MOGE_DEPTH_NORMAL_ROUTE_ID)
      || verification?.route?.backendClass !== 'browser-webgpu') errors.push('Conflicting request/result/scheduler route identity');
  if (typeof result?.requestId !== 'string' || !result.requestId.trim()
      || result.requestId !== request?.requestId || result.requestId !== verification?.route?.requestId
      || (receipt?.requestId != null && receipt.requestId !== result.requestId)) errors.push('Conflicting or missing inference request identity');
  if (result?.status !== receipt?.status) errors.push('Conflicting result/receipt status');
  // Timing does not require content-addressed model/input/output artifacts.
  // It does require observed execution on the requested, non-fallback route.
  if (result?.schema !== MOGE_ROUTE_RESULT_SCHEMA || receipt?.schema !== MOGE_ROUTE_RECEIPT_SCHEMA
      || receipt?.requestedRouteId !== MOGE_DEPTH_NORMAL_ROUTE_ID
      || receipt?.effectiveRouteId !== MOGE_DEPTH_NORMAL_ROUTE_ID) errors.push('Wrong or missing MoGe route identity');
  if (!['real', 'partial'].includes(receipt?.status) || receipt?.fallbackReason
      || receipt?.runtimeEvidence?.weights !== 'real'
      || receipt?.runtimeEvidence?.encoderFeatures !== 'backbone-gpu') errors.push('Route does not attest actual non-fallback GPU inference');
  errors.push(...validateWebGpuBackendIdentity(receipt?.backend).errors.map(e => `backend.${e}`));
  if (trace?.schema !== 'kaminos.webgpu-scheduler-event-trace.v0' || trace?.clock !== 'performance.now'
      || trace?.timingAuthority !== 'queue-submit-wait' || trace?.eventProvenance !== 'observed') errors.push('Scheduler trace is not observed queue timing');
  const times = [data.baselineStart, data.inferStart, data.inferEnd];
  if (!times.every(Number.isFinite) || !(times[0] < times[1] && times[1] < times[2])) errors.push('Invalid observation windows');
  if (frameTimes.length < 2 || frameTimes.some((t, i) => !Number.isFinite(t) || (i && t <= frameTimes[i - 1]))) errors.push('Missing or nonmonotonic frames');
  if (!frameTimes.some(t => t >= times[0] && t < times[1]) || !frameTimes.some(t => t >= times[1] && t <= times[2])) errors.push('Frame windows are not both observed');
  const spans = [], waits = [], open = new Map(), submitted = new Map(), retired = new Set();
  let submits = 0, previous = -Infinity;
  for (const rawEvent of events) {
    const e = { ...rawEvent, chunk: rawEvent.chunk ?? (Number.isInteger(rawEvent.firstBlock)
      && Number.isInteger(rawEvent.lastBlock) ? `blocks-${rawEvent.firstBlock}-${rawEvent.lastBlock}` : undefined) };
    if (e.provenance !== 'observed') errors.push('Scheduler event is not observed');
    if (!Number.isFinite(e.tMs) || e.tMs < previous || e.tMs < data.inferStart || e.tMs > data.inferEnd) errors.push('Invalid or stale scheduler event clock');
    previous = e.tMs;
    // Explicit CPU row-yield markers in inference.js, not GPU submissions.
    const cpuMarker = /^(preprocess:rows-|postprocess:points:rows-)/.test(e.chunk ?? '');
    if (e.kind === 'queue-work-done-start' && !cpuMarker) {
      submits++;
      if (!e.chunk || submitted.has(e.chunk) || retired.has(e.chunk)) errors.push('Missing or duplicate submit identity');
      submitted.set(e.chunk, e);
    }
    if (e.kind === 'chunk-retired') {
      const start = submitted.get(e.chunk);
      if (!start) errors.push('Unmatched retirement');
      else {
        spans.push({ kind: 'gpu-occupancy', semantics: 'submit-to-fence-including-queue-delay',
          phase: start.phase, chunk: e.chunk, t0: start.tMs, t1: e.tMs,
          waitMs: e.tMs - start.tMs, firstBlock: start.firstBlock, lastBlock: start.lastBlock });
        submitted.delete(e.chunk); retired.add(e.chunk);
      }
    }
    const base = String(e.kind).replace(/-(start|end)$/, ''), key = `${e.boundary}|${base}`;
    if (String(e.kind).endsWith('-start')) open.set(key, e);
    else if (String(e.kind).endsWith('-end')) {
      const start = open.get(key);
      if (!start) errors.push('Unmatched span end');
      else {
        open.delete(key);
        // These two production stages unconditionally await the queue fence
        // (inference.js). A bounded-prefix chunk's wait-end does NOT retire it.
        if (base === 'queue-work-done' && start.phase === 'decoder-heads'
            && ['neck-input', 'decoder-tail'].includes(start.chunk) && submitted.has(start.chunk)) {
          spans.push({ kind: 'gpu-occupancy', semantics: 'submit-to-fence-including-queue-delay',
            phase: start.phase, chunk: start.chunk, t0: start.tMs, t1: e.tMs,
            waitMs: e.tMs - start.tMs, completion: 'direct-queue-fence' });
          submitted.delete(start.chunk); retired.add(start.chunk);
        }
        waits.push({ kind: cpuMarker ? 'cpu-yield-marker' : base, phase: start.phase,
          chunk: start.chunk, t0: start.tMs, t1: e.tMs, waitMs: e.waitMs ?? e.yieldMs ?? e.tMs - start.tMs });
      }
    }
  }
  const occupancyComplete = submits > 0 && submitted.size === 0 && retired.size === submits;
  if (!occupancyComplete) errors.push('Submit-to-fence evidence incomplete');
  if (open.size) errors.push('Unclosed scheduler spans');
  spans.push(...waits.filter(s => s.kind !== 'queue-work-done' || !retired.has(s.chunk)));
  const gaps = frameTimes.slice(1).map((t, i) => ({ at: frameTimes[i], gapMs: t - frameTimes[i] })).filter(g => g.gapMs >= hitchMs);
  const attributed = gaps.map(g => ({ ...g,
    duringInference: g.at < data.inferEnd && g.at + g.gapMs > data.inferStart,
    overlapping: spans.filter(s => s.t0 < g.at + g.gapMs && s.t1 > g.at),
  }));
  const baselineHitches = gaps.filter(g => g.at >= data.baselineStart && g.at + g.gapMs <= data.inferStart).length;
  const inferenceHitches = attributed.filter(g => g.duringInference).length;
  return { spans, attributed, claim: 'observed-GPU-timing-only',
    outputValidation: result?.validation ?? null,
    evidence: { occupancyStatus: occupancyComplete ? 'complete' : 'incomplete', submits,
      retirements: retired.size, unmatchedSubmits: [...submitted.keys()], errors },
    summary: { schedStatus: verification?.status, inferenceMs: data.inferEnd - data.inferStart,
      baselineMs: data.inferStart - data.baselineStart, totalFrames: frameTimes.length,
      hitchThresholdMs: hitchMs, baselineHitches, inferenceHitches,
      baselineHitchesPerSecond: baselineHitches * 1000 / (data.inferStart - data.baselineStart),
      inferenceHitchesPerSecond: inferenceHitches * 1000 / (data.inferEnd - data.inferStart) } };
}
