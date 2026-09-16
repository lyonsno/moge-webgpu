#!/bin/bash
# GPU Greenroom job: hitch-alignment probe against the mounted basin app route.
set -uo pipefail
OUT="${1:?output dir required}"
mkdir -p "$OUT"
KAMINOS_ROOT="/private/tmp/kaminos-moge-live-flame-elfinblue-0916"
URL_FILE="$KAMINOS_ROOT/artifacts/basin-mounts/elfinblue-fuckeryyy.moge-url.txt"
PORT=8097
KAMINOS_VOLUME_SETTINGS_STORE="$KAMINOS_ROOT/artifacts/basin-mounts/settings-store" python3 "$KAMINOS_ROOT/serve.py" $PORT > "$OUT/serve.log" 2>&1 &
SERVE_PID=$!
for _ in $(seq 1 20); do curl -sf "http://127.0.0.1:$PORT/index.html" > /dev/null && break; sleep 1; done
URL="$(sed "s|http://127.0.0.1:8094|http://127.0.0.1:$PORT|" "$URL_FILE")"
echo "effective route: $URL" > "$OUT/route.txt"
node tools/probe_hitch_alignment.mjs --url "$URL" --out "$OUT" > "$OUT/probe.log" 2>&1
RC=$?
kill $SERVE_PID 2>/dev/null
echo "{\"probe\": $RC}" | tee "$OUT/results.json"
exit $RC
