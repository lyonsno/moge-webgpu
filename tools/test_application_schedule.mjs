import assert from 'node:assert/strict';
import * as scheduler from '../src/lib/scheduler_receipt.js';

const events = [];
const schedule = {
  invocation: { id: 'host-run' }, signal: new AbortController().signal,
  control: { async runDuty(work) { events.push('control'); return work(); } },
  runtime: {
    async prepareCommandDutyAtBoundary(request, invocation) {
      assert.equal(invocation, schedule.invocation);
      events.push(`prepare:${request.phase}`); return { id: 'duty' };
    },
    settleCommandDuty(duty, result) { assert.equal(duty.id, 'duty'); events.push(result.status); },
  },
  async withForeground(label, work) { events.push(label); return work(); },
};
const coop = scheduler.resolveCooperativeScheduler({ mode: 'cooperative', yieldMs: 0, schedule });
await scheduler.coopYield(coop, 'backbone');
assert.ok(events.includes('moge:backbone'), 'MoGe yields must service the borrowed application foreground');
assert.ok(events.includes('control'), 'CPU yield windows must participate in pause/cancellation');
assert.equal(typeof scheduler.runCooperativeCommand, 'function', 'MoGe submissions need host duty admission');
events.length = 0;
assert.equal(await scheduler.runCooperativeCommand(coop, 'backbone', async () => { events.push('submit'); return 42; }), 42);
assert.deepEqual(events, ['control', 'prepare:backbone', 'submit', 'encoded']);
events.length = 0;
await assert.rejects(scheduler.runCooperativeCommand(coop, 'decoder-heads', () => { throw Error('submit failed'); }), /submit failed/);
assert.deepEqual(events, ['control', 'prepare:decoder-heads', 'failed']);

const abort = new AbortController();
const stopped = scheduler.resolveCooperativeScheduler({ mode: 'cooperative', schedule: { ...schedule, signal: abort.signal,
  runtime: { ...schedule.runtime, async prepareCommandDutyAtBoundary() { abort.abort('stopped during admission'); return { id: 'duty' }; } } } });
events.length = 0;
await assert.rejects(scheduler.runCooperativeCommand(stopped, 'backbone', () => assert.fail('cancelled duty submitted')), { name: 'AbortError' });
assert.ok(events.includes('failed'));
await assert.rejects(scheduler.coopYield(stopped, 'backbone'), { name: 'AbortError' });
assert.throws(() => scheduler.resolveCooperativeScheduler({ mode: 'cooperative', schedule: {} }), /schedule/);

events.length = 0;
const plain = scheduler.resolveCooperativeScheduler({ mode: 'cooperative', yieldMs: 0 });
await scheduler.coopYield(plain, 'backbone');
assert.equal(await scheduler.runCooperativeCommand(plain, 'backbone', () => 7), 7);
assert.deepEqual(events, []);
assert.equal(plain.events.filter(e => e.kind === 'yield-end').length, 1);
assert.equal(JSON.stringify(scheduler.cooperativeSchedulerDescriptor(coop)).includes('host-run'), false);
console.log('MoGe host duty admission, foreground yield, cancellation and direct scheduling contracts pass');
