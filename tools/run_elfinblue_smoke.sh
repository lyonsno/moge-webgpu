#!/bin/bash
# GPU Greenroom job: smoke the live Kaminos app route for a mounted basin with
# the MoGe composition module injected. URL comes from the consumer's mount dir.
set -uo pipefail
OUT="${1:?output dir required}"
mkdir -p "$OUT"
KAMINOS_ROOT="/private/tmp/kaminos-moge-live-flame-elfinblue-0916"
URL_FILE="$KAMINOS_ROOT/artifacts/basin-mounts/elfinblue-fuckeryyy.moge-url.txt"
PORT=8096
KAMINOS_VOLUME_SETTINGS_STORE="$KAMINOS_ROOT/artifacts/basin-mounts/settings-store" python3 "$KAMINOS_ROOT/serve.py" $PORT > "$OUT/serve.log" 2>&1 &
SERVE_PID=$!
READY=0
for _ in $(seq 1 20); do
  curl -sf "http://127.0.0.1:$PORT/index.html" > /dev/null && { READY=1; break; }
  sleep 1
done
if [ "$READY" -ne 1 ]; then
  echo '{"phase": "serve-startup", "error": "kaminos server never ready"}' | tee "$OUT/results.json"
  kill $SERVE_PID 2>/dev/null; exit 1
fi
URL="$(sed "s|http://127.0.0.1:8094|http://127.0.0.1:$PORT|" "$URL_FILE")"
echo "effective route: $URL" > "$OUT/route.txt"
node tools/smoke_live_flame_page.mjs --url "$URL" --out "$OUT" > "$OUT/smoke.log" 2>&1
RC=$?
kill $SERVE_PID 2>/dev/null
echo "{\"smoke\": $RC}" | tee "$OUT/results.json"
exit $RC
