"""Owned-server probe runner; a bind failure never starts the browser."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid
from urllib.parse import urlsplit, urlunsplit

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('out', type=Path)
parser.add_argument('--kaminos-root', type=Path, default=Path('/private/tmp/kaminos-moge-live-flame-elfinblue-0916'))
parser.add_argument('--url-file', type=Path)
parser.add_argument('--port', type=int, default=8097)
parser.add_argument('--moge-bundle', type=Path, help='Expected producer bundle (default: this checkout dist-lib/moge-inference.js)')
parser.add_argument('--admission-only', action='store_true', help='Bind/source check only; never claims inference evidence')
args = parser.parse_args()
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
root = args.kaminos_root.resolve()
tools = Path(__file__).resolve().parent
run_id = str(uuid.uuid4())
report = {'runId': run_id, 'status': 'running', 'phase': 'configuration', 'probe': None,
          'kaminosRoot': str(root), 'mogeRoot': str(tools.parent), 'requestedPort': args.port}
server = None
probe = None

def save():
    (out / 'results.json').write_text(json.dumps(report, indent=2))

def git(*argv):
    return subprocess.check_output(['git', '-C', str(root), *argv], text=True).strip()

def interrupted(signum, frame):
    raise RuntimeError(f'Interrupted by signal {signum}')

signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
save()
# Invalidate a preceding success even when failure precedes probe launch.
(out / 'hitch-report.json').write_text(json.dumps({'status': 'not-started', 'runId': run_id, 'raw': None}))
try:
    url_file = args.url_file or root / 'artifacts/basin-mounts/elfinblue-fuckeryyy.moge-url.txt'
    requested = urlsplit(url_file.read_text().strip())
    if requested.scheme != 'http' or requested.hostname not in ('localhost', '127.0.0.1'):
        raise ValueError('Owned-server probe requires a loopback HTTP route')
    identity = {'runId': run_id, 'root': str(root), 'commit': git('rev-parse', 'HEAD'),
                'workingTree': git('status', '--porcelain'), 'files': {}}
    for name in ('serve.py', 'index.html', 'moge-live-flame-inject.mjs', 'moge-live-flame-shared.mjs', 'lib/moge-inference.js'):
        identity['files'][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    report['phase'] = 'server-start'; save()
    read_fd, write_fd = os.pipe()
    try:
        with (out / 'serve.log').open('w') as log:
            server = subprocess.Popen([sys.executable, str(tools / 'serve_hitch_route.py'), str(root),
                str(args.port), run_id, str(write_fd)], cwd=root, stdout=log, stderr=subprocess.STDOUT,
                pass_fds=(write_fd,), env={**os.environ, 'KAMINOS_VOLUME_SETTINGS_STORE': str(root / 'artifacts/basin-mounts/settings-store')})
    finally:
        os.close(write_fd)
    with os.fdopen(read_fd) as stream:
        acknowledgement = stream.readline()
    if not acknowledgement:
        raise RuntimeError(f'Owned server failed before bind acknowledgement; exit={server.wait()}; see serve.log')
    bound = json.loads(acknowledgement)
    if bound['pid'] != server.pid or bound['runId'] != run_id or server.poll() is not None:
        raise RuntimeError('Owned server acknowledgement mismatch')
    url = urlunsplit(('http', f"127.0.0.1:{bound['address'][1]}", requested.path, requested.query, requested.fragment))
    identity.update({'status': 'owned-server-bound', 'server': bound, 'url': url})
    manifest = out / 'source-manifest.json'
    manifest.write_text(json.dumps(identity, indent=2))
    report.update({'phase': 'source-admission', 'sourceIdentity': identity, 'url': url}); save()
    bundle = (args.moge_bundle or tools.parent / 'dist-lib/moge-inference.js').resolve()
    identity['producerBundle'] = {'path': str(bundle), 'sha256': hashlib.sha256(bundle.read_bytes()).hexdigest()}
    manifest.write_text(json.dumps(identity, indent=2))
    if identity['producerBundle']['sha256'] != identity['files']['lib/moge-inference.js']:
        raise RuntimeError('Consumer bundle differs from requested producer bundle; rebuild/vendor before inference')
    from urllib.request import urlopen
    with urlopen(url) as response:
        if response.headers.get('X-Moge-Witness-Run') != run_id:
            raise RuntimeError('HTTP response is not from owned server')
    if args.admission_only:
        report['status'] = 'admission-only'
    else:
        report['phase'] = 'probe'; save()
        with (out / 'probe.log').open('w') as log:
            probe = subprocess.Popen(['node', str(tools / 'probe_hitch_alignment.mjs'), '--url', url,
                '--out', str(out), '--run-id', run_id, '--source-manifest', str(manifest)],
                cwd=tools.parent, stdout=log, stderr=subprocess.STDOUT)
            report['probe'] = probe.wait()
        evidence = json.loads((out / 'hitch-report.json').read_text())
        if report['probe'] or evidence.get('runId') != run_id or evidence.get('status') != 'complete':
            raise RuntimeError('Probe failed or did not return current complete evidence; see hitch-report.json')
        if server.poll() is not None:
            raise RuntimeError('Owned server exited during observation')
        report['status'] = 'complete'
except Exception as error:
    report.update({'status': 'failed', 'failure': {'phase': report['phase'], 'message': str(error)}})
    print(report['failure'], file=sys.stderr)
finally:
    save()
    for process in (probe, server):
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
    report['serverExit'] = server.returncode if server else None
    save()
sys.exit(1 if report['status'] == 'failed' else 0)
