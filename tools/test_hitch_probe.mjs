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
  eventTrace:{schema:'kaminos.webgpu-scheduler-event-trace.v0', clock:'performance.now',
    timingAuthority:'queue-submit-wait', eventProvenance:'observed', events},
  schedStatus:'verified', terminalStatus:'done',
  runId:'fixture-run', sourceIdentity:{status:'verified'}, routeResult:{
    schema:'kaminos.webgpu-route-result.v0', receipt:{
      schema:'kaminos.webgpu-route-receipt.v0', status:'partial', fallbackReason:null,
      requestedRouteId:'moge.depth-normal.webgpu-local.v0',
      effectiveRouteId:'moge.depth-normal.webgpu-local.v0',
      backend:{kind:'webgpu-local',runtime:'browser',adapterName:'fixture',features:[],
        requestedFeatures:[],limits:{maxBufferSize:1024},timestampQuery:'unavailable'},
      runtimeEvidence:{weights:'real',encoderFeatures:'backbone-gpu'},
    }} };
// These fields mirror the retained live 7ae9b66cad8d receipt, not output
// authority. This fixture tests policy; replay of that raw run is conformance.
const trace = events => ({...input.eventTrace, events});
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
  const {result,report}=run('missing',{...input,eventTrace:trace(events.slice(0,2))});
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
  const {result,report}=run('block-range',{...input,eventTrace:trace(blockEvents)});
  assert.equal(result.status,0,result.stderr);
  assert.equal(report.evidence.retirements,1);
});
check('direct neck and decoder tail fences need no redundant retirement event', () => {
  const direct = ['neck-input','decoder-tail'].flatMap((chunk,i)=>[
    {kind:'queue-work-done-start',chunk,phase:'decoder-heads',boundary:i,tMs:10+i*30},
    {kind:'queue-work-done-end',phase:'decoder-heads',boundary:i,tMs:30+i*30},
  ]);
  const {result,report}=run('direct-fences',{...input,eventTrace:trace(direct)});
  assert.equal(result.status,0,result.stderr);
  assert.equal(report.evidence.retirements,2);
});
for (const [name, alter] of [
  ['wrong-route', d=>d.routeResult.receipt.effectiveRouteId='other.route'],
  ['wrong-backend', d=>d.routeResult.receipt.backend.kind='cpu'],
  ['synthetic-weights', d=>d.routeResult.receipt.runtimeEvidence.weights='synthetic'],
  ['fallback-reason', d=>d.routeResult.receipt.fallbackReason='adapter unavailable'],
  ['projected-trace', d=>d.eventTrace.eventProvenance='projected'],
  ['stale-trace', d=>d.eventTrace.events[0].tMs=1],
]) check(`${name} cannot become timing evidence`,()=>{
  const data=structuredClone(input); alter(data);
  data.routeResult.receipt.status='real';
  assert.notEqual(run(name,data).result.status,0);
});
console.log('Evidence artifacts:',root);
process.exitCode=failures?1:0;
