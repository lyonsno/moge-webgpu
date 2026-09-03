#!/bin/bash
# GPU Greenroom job: run the cooperative route harness and emit live receipts.
# Usage: run_live_receipts.sh <output_dir>
set -uo pipefail
OUT="${1:?output dir required}"
mkdir -p "$OUT"
PORT=5187
npx vite --port $PORT --strictPort > "$OUT/vite.log" 2>&1 &
VITE_PID=$!
READY=0
for _ in $(seq 1 30); do
  if curl -sf "http://localhost:$PORT/" > /dev/null; then READY=1; break; fi
  sleep 1
done
if [ "$READY" -ne 1 ]; then
  echo '{"phase": "vite-startup", "error": "dev server never ready"}' | tee "$OUT/results.json"
  kill $VITE_PID 2>/dev/null; exit 1
fi
node tools/test_cooperative_route.mjs --port $PORT --emit-receipts "$OUT/moge-route-result.json" > "$OUT/cooperative.log" 2>&1
RC=$?
kill $VITE_PID 2>/dev/null
echo "{\"cooperative\": $RC}" | tee "$OUT/results.json"
exit $RC
