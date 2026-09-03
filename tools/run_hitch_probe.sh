#!/bin/bash
# GPU Greenroom job: hitch-alignment probe against the composition page.
set -uo pipefail
OUT="${1:?output dir required}"
mkdir -p "$OUT"
# The page server may already be running for the operator on 8093; reuse it.
if ! curl -sf "http://localhost:8093/moge-live-flame.html" > /dev/null; then
  python3 /private/tmp/kaminos-moge-live-flame-compose/serve.py 8093 > "$OUT/serve.log" 2>&1 &
  for _ in $(seq 1 20); do
    curl -sf "http://localhost:8093/moge-live-flame.html" > /dev/null && break
    sleep 1
  done
fi
node tools/probe_hitch_alignment.mjs --url "http://localhost:8093/moge-live-flame.html" --out "$OUT" > "$OUT/probe.log" 2>&1
RC=$?
tail -5 "$OUT/probe.log"
echo "{\"probe\": $RC}" | tee "$OUT/results.json"
exit $RC
