import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';

// Executes the real probe with deterministic page observations. This isolates
// evidence handling; it is not a browser/GPU conformance witness.
const root = mkdtempSync(path.join(tmpdir(), 'moge-probe-contract-'));
const fake = path.join(root, 'puppeteer.mjs');
writeFileSync(fake, `export default { launch: async () => {
 if (process.env.PROBE_FAIL_LAUNCH) throw new Error('Chrome unavailable');
 return { version:async()=> 'fixture-browser', close:async()=>{}, newPage:async()=>({
 setCacheEnabled:async()=>{}, goto:async()=>({url:()=> 'http://fixture/', headers:()=>({})}),
 waitForFunction:async()=>{}, evaluate:async()=> JSON.parse(process.env.PROBE_DATA),
 }) }; } };`);
const loader = path.join(root, 'loader.mjs');
writeFileSync(loader, `export async function resolve(s,c,n) { if(s==='puppeteer-core') return {url:${JSON.stringify(pathToFileURL(fake).href)},shortCircuit:true}; return n(s,c); }`);
const events = [
  { kind:'queue-work-done-start', tMs:10, phase:'backbone', chunk:'a', boundary:1 },
  { kind:'queue-work-done-end', tMs:10, phase:'backbone', chunk:'a', boundary:1 },
  { kind:'chunk-retired', tMs:80, phase:'backbone', chunk:'a', boundary:2 },
];
const input = { frameTimes:[0,10,80,90,100], baselineStart:0,inferStart:10,inferEnd:90,
  eventTrace:{events}, schedStatus:'verified', terminalStatus:'done',
  runId:'fixture-run', sourceIdentity:{status:'verified'}, routeResult:{receipt:{status:'real'}} };
let failures = 0;
function run(name, data, env = {}) {
  const out = path.join(root, name);
  const result = spawnSync(process.execPath, ['--loader',loader,'tools/probe_hitch_alignment.mjs',
    '--url','http://fixture/','--out',out,'--settle-ms','0','--run-id','fixture-run'],
    {env:{...process.env,PROBE_DATA:JSON.stringify(data),...env},encoding:'utf8'});
  let report;
  try { report = JSON.parse(readFileSync(path.join(out,'hitch-report.json'),'utf8')); } catch {}
  return { result, report };
}
function check(name, fn) { try { fn(); console.log('PASS',name); } catch(e) { failures++;console.error('FAIL',name,e.message); } }
check('raw timing inputs remain replayable and retirement wins over a zero wait', () => {
  const {result,report}=run('valid',input);
  assert.equal(result.status,0,result.stderr);
  assert.deepEqual(report.raw,input);
  assert.equal(report.evidence.occupancyStatus,'complete');
  assert.equal(report.spans.filter(s=>s.kind==='gpu-occupancy').length,1);
  assert.equal(report.spans.filter(s=>s.kind==='queue-work-done').length,0);
});
check('missing retirement cannot claim occupancy evidence', () => {
  const {result,report}=run('missing',{...input,eventTrace:{events:events.slice(0,2)}});
  assert.notEqual(result.status,0);
  assert.equal(report.evidence.occupancyStatus,'incomplete');
  assert.deepEqual(report.raw.eventTrace.events,events.slice(0,2));
});
check('error terminal cannot be admitted as completed inference', () => {
  const {result,report}=run('error',{...input,terminalStatus:'error: failed'});
  assert.notEqual(result.status,0);
  assert.equal(report.status,'failed');
});
check('launch failure replaces previous success with a phase report', () => {
  const {result,report}=run('valid',input,{PROBE_FAIL_LAUNCH:'1'});
  assert.notEqual(result.status,0);
  assert.equal(report.failure.phase,'browser-launch');
  assert.equal(report.raw,null);
});
check('fallback output cannot become measured real inference', () => {
  const {result,report}=run('fallback',{...input,routeResult:{receipt:{status:'fallback'}}});
  assert.notEqual(result.status,0);
  assert.equal(report.status,'failed');
});
check('unsplit block range has its runtime retirement identity', () => {
  const blockEvents=events.map(e=>({...e,chunk:'blocks-0-1'}));
  delete blockEvents[0].chunk;
  Object.assign(blockEvents[0],{firstBlock:0,lastBlock:1});
  const {result,report}=run('block-range',{...input,eventTrace:{events:blockEvents}});
  assert.equal(result.status,0,result.stderr);
  assert.equal(report.evidence.retirements,1);
});
console.log('Evidence artifacts:',root);
process.exitCode=failures?1:0;
