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

# --strictPort: if 5186 is busy, fail rather than silently serving elsewhere.
npx vite --port $PORT --strictPort > "$OUT/vite.log" 2>&1 &
VITE_PID=$!
# Readiness probe — a dead or unbound server must fail loud with a named phase,
# not let the harnesses run against nothing.
READY=0
for _ in $(seq 1 30); do
  if curl -sf "http://localhost:$PORT/" > /dev/null; then READY=1; break; fi
  sleep 1
done
if [ "$READY" -ne 1 ]; then
  echo "{\"phase\": \"vite-startup\", \"error\": \"dev server never became ready on port $PORT\"}" | tee "$OUT/results.json"
  kill $VITE_PID 2>/dev/null
  exit 1
fi
echo "effective route: vite on port $PORT (strictPort)" >> "$OUT/vite.log"

node tools/test_cooperative_route.mjs --port $PORT > "$OUT/cooperative.log" 2>&1
COOP=$?
node tools/test_depth_parity.mjs --port $PORT > "$OUT/parity.log" 2>&1
PARITY=$?
# The route-receipt harness asserts stub-route semantics; it opts into stub
# mode explicitly via ?forceStub=1 (real/hosted weights would otherwise load).
node tools/test_webgpu_route_receipt.mjs --port $PORT > "$OUT/route_receipt.log" 2>&1
RECEIPT=$?

kill $VITE_PID 2>/dev/null

echo "{\"unit\": $UNIT, \"cooperative\": $COOP, \"parity\": $PARITY, \"route_receipt\": $RECEIPT}" | tee "$OUT/results.json"
[ $UNIT -eq 0 ] && [ $COOP -eq 0 ] && [ $PARITY -eq 0 ] && [ $RECEIPT -eq 0 ]
