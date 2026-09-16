"""Run canonical Kaminos serve.py, acknowledging only our successful bind."""
import http.server
import json
import os
from pathlib import Path
import runpy
import sys

root, port, run_id, ready_fd = sys.argv[1:]
server_type = http.server.ThreadingHTTPServer
end_headers = http.server.BaseHTTPRequestHandler.end_headers

class OwnedServer(server_type):
    def __init__(self, *args, **kwargs):
        address, *rest = args
        args = (('127.0.0.1', address[1]), *rest)
        super().__init__(*args, **kwargs)  # A competing listener fails here.
        with os.fdopen(int(ready_fd), 'w') as stream:
            stream.write(json.dumps({'pid': os.getpid(), 'address': self.server_address, 'runId': run_id}) + '\n')

def identified_headers(self):
    self.send_header('X-Moge-Witness-Run', run_id)
    self.send_header('Cache-Control', 'no-store')
    end_headers(self)

http.server.ThreadingHTTPServer = OwnedServer
http.server.BaseHTTPRequestHandler.end_headers = identified_headers
os.chdir(root)
sys.path.insert(0, root)
sys.argv = [str(Path(root) / 'serve.py'), port]
runpy.run_path(sys.argv[0], run_name='__main__')
