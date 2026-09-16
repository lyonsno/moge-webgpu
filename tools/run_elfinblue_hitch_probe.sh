#!/bin/bash
# GPU Greenroom job: hitch-alignment probe against the mounted basin app route.
set -eu
exec python3 "$(dirname "$0")/run_elfinblue_hitch_probe.py" "$@"
