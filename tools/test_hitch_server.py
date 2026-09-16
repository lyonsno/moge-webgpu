"""Real loopback collision test; no browser or GPU work."""
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
work = Path(tempfile.mkdtemp(prefix='moge-owned-server-'))
with socket.socket() as occupied:
    occupied.bind(('127.0.0.1', 0))
    occupied.listen()
    port = occupied.getsockname()[1]
    source = (root / 'tools/run_elfinblue_hitch_probe.sh').read_text()
    # Baseline only: the predecessor has no caller-owned port parameter.
    wrapper = root / 'tools/run_elfinblue_hitch_probe.sh'
    if 'PORT=8097' in source:
        source = source.replace('PORT=8097', f'PORT={port}')
        wrapper = work / 'wrapper.sh'
        wrapper.write_text(source)
    fakebin = work / 'bin'
    fakebin.mkdir()
    marker = work / 'browser-started'
    node = fakebin / 'node'
    node.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 0\n')
    node.chmod(0o755)
    # The predecessor's reachability check is supplied separately: it says a
    # listener answered even though the newly launched server failed to bind.
    curl = fakebin / 'curl'
    curl.write_text('#!/bin/sh\nexit 0\n')
    curl.chmod(0o755)
    out = work / 'out'
    result = subprocess.run(['bash', str(wrapper), str(out), '--port', str(port)],
        cwd=root, env={**os.environ, 'PATH': str(fakebin) + ':' + os.environ['PATH']}, capture_output=True, text=True)
    assert result.returncode != 0, 'occupied port incorrectly succeeded'
    assert not marker.exists(), 'browser/inference started after bind failure'
    report = json.loads((out / 'results.json').read_text())
    assert report['failure']['phase'] == 'server-start', report
    assert report['probe'] is None, report
    print('PASS occupied port rejects before browser; artifacts:', work)

# Source comparison is exercised separately so bind-failure remains first.
out = work / 'stale-bundle'
wrong_bundle = work / 'wrong-bundle.js'
wrong_bundle.write_text('// deliberately not the consumed producer\n')
result = subprocess.run(['bash', str(wrapper), str(out), '--port', '0', '--admission-only', '--moge-bundle', str(wrong_bundle)],
    cwd=root, capture_output=True, text=True)
consumer = Path('/private/tmp/kaminos-moge-live-flame-elfinblue-0916/lib/moge-inference.js')
assert result.returncode != 0, 'stale consumer bundle incorrectly admitted'
assert json.loads((out / 'results.json').read_text())['failure']['phase'] == 'source-admission'
print('PASS mismatched producer/consumer bundle rejected')
out = work / 'admitted'
result = subprocess.run(['bash', str(wrapper), str(out), '--port', '0', '--admission-only', '--moge-bundle', str(consumer)],
    cwd=root, capture_output=True, text=True)
assert result.returncode == 0, result.stderr
report = json.loads((out / 'results.json').read_text())
assert report['status'] == 'admission-only' and report['probe'] is None
assert report['serverExit'] is not None
print('PASS canonical host bind/header/bundle admission and cleanup (no inference)')
