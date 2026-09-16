// Pure replay over hitch-report.raw. Temporal overlap is not a causal claim.
export function analyzeHitches(data, hitchMs = 50) {
  const errors = [];
  const events = data.eventTrace?.events ?? [], frameTimes = data.frameTimes ?? [];
  if (!/^done\b/i.test(data.terminalStatus ?? '') || data.observationError) errors.push('Inference did not complete');
  if (data.schedStatus !== 'verified' || !data.routeResult) errors.push('Current scheduler receipt is not verified');
  if (data.routeResult?.receipt?.status !== 'real') errors.push('Route did not produce real inference output');
  const times = [data.baselineStart, data.inferStart, data.inferEnd];
  if (!times.every(Number.isFinite) || !(times[0] < times[1] && times[1] < times[2])) errors.push('Invalid observation windows');
  if (frameTimes.length < 2 || frameTimes.some((t, i) => !Number.isFinite(t) || (i && t <= frameTimes[i - 1]))) errors.push('Missing or nonmonotonic frames');
  if (!frameTimes.some(t => t >= times[0] && t < times[1]) || !frameTimes.some(t => t >= times[1] && t <= times[2])) errors.push('Frame windows are not both observed');
  const spans = [], waits = [], open = new Map(), submitted = new Map(), retired = new Set();
  let submits = 0, previous = -Infinity;
  for (const rawEvent of events) {
    const e = { ...rawEvent, chunk: rawEvent.chunk ?? (Number.isInteger(rawEvent.firstBlock)
      && Number.isInteger(rawEvent.lastBlock) ? `blocks-${rawEvent.firstBlock}-${rawEvent.lastBlock}` : undefined) };
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
  return { spans, attributed,
    evidence: { occupancyStatus: occupancyComplete ? 'complete' : 'incomplete', submits,
      retirements: retired.size, unmatchedSubmits: [...submitted.keys()], errors },
    summary: { schedStatus: data.schedStatus, inferenceMs: data.inferEnd - data.inferStart,
      baselineMs: data.inferStart - data.baselineStart, totalFrames: frameTimes.length,
      hitchThresholdMs: hitchMs, baselineHitches, inferenceHitches,
      baselineHitchesPerSecond: baselineHitches * 1000 / (data.inferStart - data.baselineStart),
      inferenceHitchesPerSecond: inferenceHitches * 1000 / (data.inferEnd - data.inferStart) } };
}
