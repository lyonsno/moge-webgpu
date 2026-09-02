#!/usr/bin/env node
/**
 * Pure-Node unit tests for the scheduler verification receipt authority logic
 * (src/lib/scheduler_receipt.js). No GPU, no browser, no vite.
 *
 * Pins the authority boundary from the e4f794f scheduler-proof review:
 *   1. A trace synthesized from stage timings must NEVER reach
 *      status "verified" / classification "observed-boundary" — even when the
 *      declared scheduler has yieldMs 0 (the bypass found in independent
 *      review of the first cooperative-scheduling slice).
 *   2. A genuinely observed trace with observed yields per phase verifies.
 *   3. Observed queue waits without requested-but-unobserved yields downgrade
 *      with yield-events-missing.
 *   4. No events at all → scheduler-unverified with event-trace-missing.
 */

import {
  cooperativeSchedulerDescriptor,
  createMogeSchedulerVerificationReceipt,
  resolveCooperativeScheduler,
} from '../src/lib/scheduler_receipt.js';

const failures = [];
function check(name, cond, detail) {
  if (!cond) failures.push(`${name}: ${detail}`);
}

const STAGES = [
  { name: 'backbone', ms: 100 },
  { name: 'neck-input', ms: 10 },
  { name: 'decoder-heads', ms: 50 },
  { name: 'output-readback', ms: 20 },
];

function schedulerWithYield(yieldMs) {
  const coop = resolveCooperativeScheduler({ mode: 'cooperative', yieldMs, vitBlockChunkSize: 1 });
  return cooperativeSchedulerDescriptor(coop, { backboneTotalItems: 24 });
}

function observedEventsFor(phases, { withYields }) {
  const events = [];
  let t = 0;
  for (const phase of phases) {
    const kind = phase === 'output-readback' ? 'readback-wait' : 'queue-work-done';
    events.push({ tMs: t++, phase, boundary: `moge-stage:${phase}`, kind: `${kind}-start`, source: 'moge-webgpu-runtime', provenance: 'observed' });
    events.push({ tMs: t++, phase, boundary: `moge-stage:${phase}`, kind: `${kind}-end`, waitMs: 1, source: 'moge-webgpu-runtime', provenance: 'observed' });
    if (withYields) {
      events.push({ tMs: t++, phase, boundary: `moge-stage:${phase}`, kind: 'yield-start', source: 'moge-webgpu-runtime', provenance: 'observed' });
      events.push({ tMs: t++, phase, boundary: `moge-stage:${phase}`, kind: 'yield-end', yieldMs: 4, source: 'moge-webgpu-runtime', provenance: 'observed' });
    }
  }
  return events;
}

const PHASES = ['backbone', 'decoder-heads', 'output-readback'];

// 1. Synthesized trace + yieldMs 0 must not verify (the Finding 1 bypass).
{
  const receipt = createMogeSchedulerVerificationReceipt({
    scheduler: schedulerWithYield(0),
    stagedStages: STAGES,
    observedEvents: null,
  });
  check('synthesized-yield0-status', receipt.status !== 'verified', `got ${receipt.status}`);
  check('synthesized-yield0-classification', receipt.classification !== 'observed-boundary', `got ${receipt.classification}`);
  check('synthesized-yield0-downgrade', receipt.downgrades.includes('event-trace-synthesized'), `downgrades=${JSON.stringify(receipt.downgrades)}`);
  check('synthesized-yield0-proxy-flag', receipt.falseAuthorityChecks.timingProxyOnly === true, 'timingProxyOnly not set');
}

// 1b. Synthesized trace + yieldMs 4 must not verify either.
{
  const receipt = createMogeSchedulerVerificationReceipt({
    scheduler: schedulerWithYield(4),
    stagedStages: STAGES,
    observedEvents: null,
  });
  check('synthesized-yield4-status', receipt.status !== 'verified', `got ${receipt.status}`);
  check('synthesized-yield4-yield-downgrade', receipt.downgrades.includes('yield-events-missing'), `downgrades=${JSON.stringify(receipt.downgrades)}`);
}

// 2. Observed trace with yields verifies.
{
  const receipt = createMogeSchedulerVerificationReceipt({
    scheduler: schedulerWithYield(4),
    stagedStages: STAGES,
    observedEvents: observedEventsFor(PHASES, { withYields: true }),
  });
  check('observed-status', receipt.status === 'verified', `got ${receipt.status} downgrades=${JSON.stringify(receipt.downgrades)}`);
  check('observed-classification', receipt.classification === 'observed-boundary', `got ${receipt.classification}`);
  check('observed-no-downgrades', receipt.downgrades.length === 0, `downgrades=${JSON.stringify(receipt.downgrades)}`);
  for (const a of receipt.boundaryAssertions) {
    check(`observed-yieldcount-${a.field}`, a.observedYieldCount > 0, `got ${a.observedYieldCount}`);
  }
}

// 3. Observed queue waits, yields requested but not performed → downgraded.
{
  const receipt = createMogeSchedulerVerificationReceipt({
    scheduler: schedulerWithYield(4),
    stagedStages: STAGES,
    observedEvents: observedEventsFor(PHASES, { withYields: false }),
  });
  check('observed-noyield-status', receipt.status !== 'verified', `got ${receipt.status}`);
  check('observed-noyield-downgrade', receipt.downgrades.includes('yield-events-missing'), `downgrades=${JSON.stringify(receipt.downgrades)}`);
}

// 4. No evidence at all.
{
  const receipt = createMogeSchedulerVerificationReceipt({
    scheduler: schedulerWithYield(4),
    stagedStages: null,
    observedEvents: null,
  });
  check('empty-status', receipt.status === 'scheduler-unverified', `got ${receipt.status}`);
  check('empty-downgrade', receipt.downgrades.includes('event-trace-missing'), `downgrades=${JSON.stringify(receipt.downgrades)}`);
}

// 5. Effective backbone chunk size is clamped to the block count (truthful
//    descriptor for oversized requests — no silent cap on the request itself).
{
  const coop = resolveCooperativeScheduler({ mode: 'cooperative', vitBlockChunkSize: 1e9 });
  const descriptor = cooperativeSchedulerDescriptor(coop, { backboneTotalItems: 24 });
  check('oversized-request-preserved', descriptor.requestedScheduler.phaseChunkSize.backbone === 1e9,
    `got ${descriptor.requestedScheduler.phaseChunkSize.backbone}`);
  check('oversized-effective-clamped', descriptor.effectiveScheduler.phaseChunkSize.backbone === 24,
    `got ${descriptor.effectiveScheduler.phaseChunkSize.backbone}`);
}

if (failures.length) {
  console.log('FAIL:');
  for (const f of failures) console.log(`  - ${f}`);
  process.exit(1);
}
console.log(`PASS (${5} cases)`);
