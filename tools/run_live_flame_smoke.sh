#!/bin/bash
# GPU Greenroom job: serve the kaminos composition page and run the visual
# smoke witness (fire + cooperative MoGe on one device).
set -uo pipefail
OUT="${1:?output dir required}"
mkdir -p "$OUT"
KAMINOS_ROOT="/private/tmp/kaminos-moge-live-flame-compose"
PORT=8093
python3 "$KAMINOS_ROOT/serve.py" $PORT > "$OUT/serve.log" 2>&1 &
SERVE_PID=$!
READY=0
for _ in $(seq 1 20); do
  if curl -sf "http://localhost:$PORT/moge-live-flame.html" > /dev/null; then READY=1; break; fi
  sleep 1
done
if [ "$READY" -ne 1 ]; then
  echo '{"phase": "serve-startup", "error": "kaminos server never ready"}' | tee "$OUT/results.json"
  kill $SERVE_PID 2>/dev/null; exit 1
fi
node tools/smoke_live_flame_page.mjs --url "http://localhost:$PORT/moge-live-flame.html" --out "$OUT" > "$OUT/smoke.log" 2>&1
RC=$?
kill $SERVE_PID 2>/dev/null
echo "{\"smoke\": $RC}" | tee "$OUT/results.json"
exit $RC
