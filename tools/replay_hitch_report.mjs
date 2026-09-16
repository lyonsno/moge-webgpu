#!/usr/bin/env node
// Reinterpret retained raw inputs without rerunning the GPU or overwriting them.
import { readFileSync, writeFileSync, realpathSync } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { analyzeHitches } from './hitch_analysis.mjs';

const [input, output] = process.argv.slice(2);
if (!input || !output) throw new Error('Usage: replay_hitch_report.mjs INPUT OUTPUT');
if (realpathSync(input) === path.resolve(output)) throw new Error('Replay must not overwrite its source');
const bytes = readFileSync(input), original = JSON.parse(bytes);
const analysis = analyzeHitches(original.raw, original.summary?.hitchThresholdMs ?? 50);
const replay = { status: analysis.evidence.errors.length ? 'failed' : 'complete',
  interpretation: 'historical-raw-replay', measurementAuthority: 'headless-relative-timing-only',
  source: { path: realpathSync(input), sha256: createHash('sha256').update(bytes).digest('hex'),
    originalStatus: original.status, identity: original.raw?.sourceIdentity ?? null,
    sourceRecheck: 'not-established-by-this-replay' }, ...analysis };
writeFileSync(output, JSON.stringify(replay, null, 2), {flag:'wx'});
console.log(JSON.stringify({status:replay.status,evidence:replay.evidence,summary:replay.summary},null,2));
process.exitCode = analysis.evidence.errors.length ? 1 : 0;
