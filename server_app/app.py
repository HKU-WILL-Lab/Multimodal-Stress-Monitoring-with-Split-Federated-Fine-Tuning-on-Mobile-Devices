"""Local desktop dashboard for the two independent SFL services."""
from __future__ import annotations

import argparse
import atexit
import json
import mimetypes
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
import time
from urllib.request import urlopen
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import webbrowser

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.parent / 'sfl_runtime' / 'python'
sys.path.insert(0, str(RUNTIME))
from sfl_clean.config import load_config
from deployment import Deployment


def endpoint(bind):
    host, port = bind.rsplit(':', 1)
    return ('127.0.0.1' if host in ('0.0.0.0', 'localhost') else host, int(port))


def listening(bind):
    try:
        with socket.create_connection(endpoint(bind), timeout=.15):
            return True
    except OSError:
        return False


class Desktop:
    def __init__(self, config, status_port=50053):
        self.lock = threading.RLock()
        self.path = Path(config).resolve()
        self.config = load_config(self.path)
        self.raw = json.loads(self.path.read_text(encoding='utf-8'))
        self.status_port = status_port
        self.processes = {}
        self.stopped = set()
        self.output = ROOT / 'runtime' / self.config.run_id
        # Run IDs become local directory names, never arbitrary paths.
        if not self.config.run_id.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Use letters, digits, hyphens or underscores for run_id')
        self.output.mkdir(parents=True, exist_ok=True)
        self.deployment = Deployment(self)

    def new_config(self, path):
        config = load_config(path)
        with self.lock:
            for role in ('main', 'federated'):
                self.stop(role)
            self.path = Path(path).resolve()
            self.config = config
            self.raw = json.loads(self.path.read_text(encoding='utf-8'))
            self.processes = {}
            self.stopped = set()
            self.output = ROOT / 'runtime' / config.run_id
            self.output.mkdir(parents=True, exist_ok=True)

    def start(self, role):
        with self.lock:
            if role not in ('main', 'federated'):
                raise ValueError('Unknown role')
            if role in self.processes and self.processes[role].poll() is None:
                return
            # A fresh run is required after a process has used its model/round state.
            if role in self.processes:
                raise ValueError('This service has already run. Choose a new run configuration before restarting.')
            bind = self.config.suffix_rpc.bind if role == 'main' else self.config.coordinator_rpc.bind
            if listening(bind) or (role == 'federated' and listening(f'127.0.0.1:{self.status_port}')):
                raise ValueError('Service port is already occupied; stop its owner or choose a different port.')
            if role == 'main' and self.config.metrics_path.exists() and self.config.metrics_path.stat().st_size:
                raise ValueError('Metrics already exist for this run. Use a new run_id and metrics_path.')
            if (self.config.checkpoint_root / self.config.run_id).exists():
                raise ValueError('Checkpoints already exist for this run. Use a new run_id.')
            module = 'cli_suffix' if role == 'main' else 'cli_coordinator'
            args = [sys.executable, '-u', '-m', 'sfl_clean.' + module, '--config', str(self.path)]
            if role == 'federated':
                args += ['--status-bind', f'127.0.0.1:{self.status_port}']
            env = dict(os.environ, PYTHONPATH=str(RUNTIME), PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
            # pythonw hosts the window; python.exe is used for captured service logs.
            # stop() terminates its owned process tree, including venv launchers.
            if os.name == 'nt':
                args[0] = str(Path(sys.executable).with_name('python.exe'))
            with (self.output / (role + '.log')).open('wb') as log:
                self.processes[role] = subprocess.Popen(args, cwd=self.path.parent, env=env,
                    stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

    def stop(self, role):
        with self.lock:
            proc = self.processes.get(role)
            if proc and proc.poll() is None:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                self.stopped.add(role)

    def close(self):
        self.deployment.close()
        for role in ('main', 'federated'):
            self.stop(role)

    def snapshot(self):
        with self.lock:
            services = {}
            for role in ('main', 'federated'):
                proc = self.processes.get(role)
                bind = self.config.suffix_rpc.bind if role == 'main' else self.config.coordinator_rpc.bind
                state = 'Stopped'
                if proc:
                    state = ('Ready' if listening(bind) else 'Starting') if proc.poll() is None else ('Stopped' if role in self.stopped else 'Failed')
                log = self.output / (role + '.log')
                lines = []
                if log.exists():
                    with log.open('rb') as f:
                        f.seek(max(0, log.stat().st_size - 64000))
                        lines = f.read().decode('utf-8', errors='replace').splitlines()[-200:]
                services[role] = dict(state=state, bind=bind, logs=lines)
            records = []
            if self.config.metrics_path.exists():
                for line in self.config.metrics_path.read_text(encoding='utf-8').splitlines():
                    try:
                        item = json.loads(line)
                        if item.get('run_id') == self.config.run_id:
                            records.append(item)
                    except json.JSONDecodeError:
                        pass  # Writer may be midway through the last JSONL record.
            fed = dict(phase='DISCONNECTED', events=[], clients=[])
            if services['federated']['state'] == 'Ready':
                try:
                    with urlopen(f'http://127.0.0.1:{self.status_port}/status', timeout=.5) as response:
                        candidate = json.load(response)
                    if candidate.get('runId') == self.config.run_id:
                        fed = candidate
                except (OSError, ValueError):
                    pass
            return dict(application='MobiWellbeing Server', runId=self.config.run_id, config=self.raw, services=services,
                        metrics=records, aggregation=fed, deployment=self.deployment.snapshot())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--port', default=8765, type=int)
    parser.add_argument('--status-port', default=50053, type=int)
    parser.add_argument('--no-window', action='store_true')
    args = parser.parse_args()
    origin = f'http://127.0.0.1:{args.port}'
    def open_window():
        edge = Path(os.environ.get('PROGRAMFILES(X86)', '')) / 'Microsoft/Edge/Application/msedge.exe'
        if os.name == 'nt' and edge.exists():
            subprocess.Popen([str(edge), '--app=' + origin, '--window-size=1500,1000'])
        else:
            webbrowser.open(origin)

    if listening(f'127.0.0.1:{args.port}'):
        with urlopen(origin + '/api/state', timeout=2) as response:
            running = json.load(response)
        legacy = (running.get('runId') == load_config(args.config).run_id
                  and set(running.get('services', {})) == {'main', 'federated'})
        if running.get('application') != 'MobiWellbeing Server' and not legacy:
            raise ValueError('Application port is occupied by another program')
        if not args.no_window:
            open_window()
        return
    app = Desktop(args.config, args.status_port)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, content, kind='application/json'):
            if not isinstance(content, bytes):
                content = json.dumps(content, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(content)

        def allowed(self):
            return self.headers.get('Host') == f'127.0.0.1:{args.port}'

        def do_GET(self):
            if not self.allowed():
                return self.respond(403, {'error': 'Invalid host'})
            if self.path == '/api/state':
                return self.respond(200, dict(app.snapshot(), token=token))
            if self.path == '/api/devices':
                try:
                    return self.respond(200, app.deployment.devices())
                except Exception as error:
                    return self.respond(400, {'error': str(error)})
            relative = self.path.split('?', 1)[0].lstrip('/') or 'main.html'
            file = (ROOT / 'static' / relative).resolve()
            if not file.is_relative_to((ROOT / 'static').resolve()) or not file.is_file():
                return self.respond(404, {'error': 'Not found'})
            self.respond(200, file.read_bytes(), mimetypes.guess_type(file)[0] or 'application/octet-stream')

        def do_POST(self):
            if not self.allowed() or self.headers.get('Origin') != origin or self.headers.get('X-App-Token') != token:
                return self.respond(403, {'error': 'Invalid application request'})
            try:
                if self.path in ('/api/deploy', '/api/train'):
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 4096:
                        raise ValueError('Invalid deployment request')
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise ValueError('Expected deployment settings')
                    app.deployment.begin(payload, start=self.path == '/api/train')
                    return self.respond(200, {'ok': True})
                if self.path == '/api/quit':
                    self.respond(200, {'ok': True})
                    threading.Thread(target=server.shutdown, daemon=True).start()
                    return
                parts = self.path.strip('/').split('/')
                if len(parts) != 3 or parts[0] != 'api' or parts[1] not in ('main', 'federated') or parts[2] not in ('start', 'stop'):
                    return self.respond(404, {'error': 'Not found'})
                if app.deployment.snapshot()['phase'] == 'Working':
                    raise ValueError('Wait for the current deployment job to finish')
                getattr(app, parts[2])(parts[1])
                self.respond(200, {'ok': True})
            except Exception as error:
                self.respond(400, {'error': str(error)})

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.daemon_threads = True
    atexit.register(app.close)
    if not args.no_window:
        open_window()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
        server.server_close()


if __name__ == '__main__':
    main()
