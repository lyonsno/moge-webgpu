#!/bin/bash
# GPU Greenroom job wrapper: full cooperative-scheduling verification battery.
# Usage: run_coop_verification.sh <output_dir>
# Runs from the repo/worktree root (job type sets cwd). Exit nonzero on any failure.
set -uo pipefail
OUT="${1:?output dir required}"
mkdir -p "$OUT"
PORT=5186

node tools/test_scheduler_receipt_unit.mjs > "$OUT/unit.log" 2>&1
UNIT=$?

npx vite --port $PORT > "$OUT/vite.log" 2>&1 &
VITE_PID=$!
sleep 3

node tools/test_cooperative_route.mjs --port $PORT > "$OUT/cooperative.log" 2>&1
COOP=$?
node tools/test_depth_parity.mjs --port $PORT > "$OUT/parity.log" 2>&1
PARITY=$?

kill $VITE_PID 2>/dev/null

echo "{\"unit\": $UNIT, \"cooperative\": $COOP, \"parity\": $PARITY}" | tee "$OUT/results.json"
[ $UNIT -eq 0 ] && [ $COOP -eq 0 ] && [ $PARITY -eq 0 ]
