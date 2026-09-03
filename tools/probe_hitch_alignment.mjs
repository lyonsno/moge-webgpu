#!/usr/bin/env node
/**
 * Hitch alignment probe for the moge-live-flame composition page.
 *
 * Records rAF frame timestamps and the cooperative scheduler event trace on
 * the same performance.now clock during one inference run, then attributes
 * each frame hitch (gap > threshold) to the scheduler span it overlaps:
 * which phase, which chunk (vit-block range / neck-input / neck-and-heads /
 * readback), and that span's queue waitMs.
 *
 * Usage: node tools/probe_hitch_alignment.mjs --url <page-url> --out <dir> [--hitch-ms 50]
 */
import puppeteer from 'puppeteer-core';
import { mkdirSync, writeFileSync } from 'fs';
import path from 'path';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const args = process.argv.slice(2);
const arg = (name, dflt) => args.includes(name) ? args[args.indexOf(name) + 1] : dflt;
const url = arg('--url', 'http://localhost:8093/moge-live-flame.html');
const outDir = path.resolve(arg('--out', '/tmp/moge-hitch-probe'));
const hitchMs = Number(arg('--hitch-ms', '50'));
mkdirSync(outDir, { recursive: true });

const browser = await puppeteer.launch({
  executablePath: CHROME_PATH,
  headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--disable-gpu-sandbox', '--no-sandbox'],
  defaultViewport: { width: 1440, height: 900 },
});
const page = await browser.newPage();
try {
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForFunction(() => window.__mogeLiveFlameReady === true, { timeout: 600000 });
  // Let the fire settle so baseline frame cost is honest.
  await new Promise(r => setTimeout(r, 4000));

  const data = await page.evaluate(async () => {
    const frameTimes = [];
    let rafId;
    const tick = now => { frameTimes.push(now); rafId = requestAnimationFrame(tick); };
    rafId = requestAnimationFrame(tick);

    const baselineStart = performance.now();
    await new Promise(r => setTimeout(r, 3000));
    const inferStart = performance.now();
    document.getElementById('ignite').click();
    await new Promise(resolve => {
      const check = () => /done|error/.test(document.getElementById('hud-infer').textContent)
        ? resolve() : setTimeout(check, 200);
      check();
    });
    const inferEnd = performance.now();
    await new Promise(r => setTimeout(r, 1500));
    cancelAnimationFrame(rafId);

    const routeResult = window.__mogeDebug?.webGpuRouteResult || null;
    const eventTrace = routeResult?.receipt?.runtime?.schedulerVerification?.eventTrace || null;
    return { frameTimes, baselineStart, inferStart, inferEnd, eventTrace,
             schedStatus: routeResult?.receipt?.runtime?.schedulerVerification?.status ?? null };
  });

  // --- Offline alignment ---
  const events = data.eventTrace?.events || [];
  // Build spans: consecutive (start,end) pairs of the same kind family.
  const spans = [];
  const open = new Map();
  for (const e of events) {
    const base = String(e.kind).replace(/-(start|end)$/, '');
    const key = `${e.boundary}|${base}`;
    if (e.kind.endsWith('-start')) open.set(key, e);
    else if (e.kind.endsWith('-end') && open.has(key)) {
      const s = open.get(key); open.delete(key);
      spans.push({
        phase: e.phase, kind: base, t0: s.tMs, t1: e.tMs,
        waitMs: e.waitMs ?? e.yieldMs ?? (e.tMs - s.tMs),
        firstBlock: s.firstBlock, lastBlock: s.lastBlock, chunk: s.chunk,
      });
    }
  }

  const gaps = [];
  for (let i = 1; i < data.frameTimes.length; i++) {
    const dt = data.frameTimes[i] - data.frameTimes[i - 1];
    if (dt >= hitchMs) gaps.push({ at: data.frameTimes[i - 1], gapMs: dt });
  }
  const attributed = gaps.map(g => {
    const overlapping = spans.filter(s => s.t0 < g.at + g.gapMs && s.t1 > g.at)
      .sort((a, b) => (Math.min(b.t1, g.at + g.gapMs) - Math.max(b.t0, g.at))
                    - (Math.min(a.t1, g.at + g.gapMs) - Math.max(a.t0, g.at)));
    const top = overlapping[0] || null;
    const during = g.at >= data.inferStart && g.at <= data.inferEnd;
    return {
      atMs: Math.round(g.at - data.inferStart), gapMs: Math.round(g.gapMs), duringInference: during,
      culprit: top ? {
        phase: top.phase, kind: top.kind, waitMs: Math.round(top.waitMs ?? 0),
        blocks: Number.isFinite(top.firstBlock) ? `${top.firstBlock}-${top.lastBlock}` : (top.chunk || null),
      } : (during ? 'no-overlapping-scheduler-span' : 'outside-inference'),
    };
  });

  // Aggregate: worst chunks by queue waitMs.
  const chunkWaits = spans
    .filter(s => s.kind === 'queue-work-done' || s.kind === 'readback-wait')
    .sort((a, b) => b.waitMs - a.waitMs)
    .slice(0, 12)
    .map(s => ({ phase: s.phase, blocks: Number.isFinite(s.firstBlock) ? `${s.firstBlock}-${s.lastBlock}` : (s.chunk || s.kind), waitMs: Math.round(s.waitMs) }));

  const baselineGaps = gaps.filter(g => g.at < data.inferStart).length;
  const inferGaps = attributed.filter(a => a.duringInference);
  const summary = {
    schedStatus: data.schedStatus,
    inferenceMs: Math.round(data.inferEnd - data.inferStart),
    totalFrames: data.frameTimes.length,
    hitchThresholdMs: hitchMs,
    baselineHitches: baselineGaps,
    inferenceHitches: inferGaps.length,
    worstQueueWaits: chunkWaits,
    hitches: inferGaps.slice(0, 25),
  };
  writeFileSync(path.join(outDir, 'hitch-report.json'), JSON.stringify({ summary, spans, attributed }, null, 2));
  console.log(JSON.stringify(summary, null, 2));
} finally {
  await browser.close();
}
