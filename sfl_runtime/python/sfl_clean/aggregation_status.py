"""Small, read-only status endpoint; separate from the aggregation lock."""
import json
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread
from time import time


class AggregationStatus:
    def __init__(self, quorum: int):
        self._lock = Lock()
        self._events = deque(maxlen=64)
        self._state = {}
        self._clients = {}
        self.publish('IDLE', 1, 0, quorum)

    def publish(self, phase: str, round_number: int, received: int, quorum: int):
        with self._lock:
            self._state = dict(phase=phase, round=round_number, received=received,
                               quorum=quorum, updatedAt=time())
            self._events.append(dict(self._state))

    def snapshot(self):
        with self._lock:
            return {**self._state, 'events': list(self._events),
                    'clients': [dict(value) for value in self._clients.values()]}

    def client(self, client_id, **fields):
        """Record successful protocol activity, not an inferred network connection."""
        with self._lock:
            self._clients.setdefault(client_id, {'id': client_id}).update(fields, lastSeen=time())


def serve_status(status: AggregationStatus, bind: str, run_id: str):
    host, port = bind.rsplit(':', 1)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/status':
                self.send_error(404)
                return
            data = json.dumps({'runId': run_id, **status.snapshot()}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer((host, int(port)), Handler)
    server.daemon_threads = True
    Thread(target=server.serve_forever, name='aggregation-status', daemon=True).start()
    return server
