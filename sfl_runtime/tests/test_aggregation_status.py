import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from urllib.request import ProxyHandler, build_opener

import pytest
import torch

import sfl_clean.coordinator as module
from sfl_clean.aggregation_status import serve_status
from sfl_clean.coordinator import RoundCoordinator


def test_status_remains_readable_during_aggregation(monkeypatch):
    coordinator = RoundCoordinator(total_rounds=1, submission_quorum=2)
    server = serve_status(coordinator.status, '127.0.0.1:0', 'status-test')
    opener = build_opener(ProxyHandler({}))
    def read():
        with opener.open(f'http://127.0.0.1:{server.server_port}/status', timeout=2) as response:
            return json.load(response)
    entered, release = Event(), Event()
    real_average = module.weighted_average
    def slow_average(states):
        entered.set()
        assert release.wait(5)
        return real_average(states)
    monkeypatch.setattr(module, 'weighted_average', slow_average)
    try:
        assert read()['phase'] == 'IDLE'
        coordinator.bootstrap('a', {'p.lora_A': torch.zeros(1)})
        assert read()['phase'] == 'WAITING'
        coordinator.submit('a', 1, 1, {'p.lora_A': torch.ones(1)})
        assert read()['received'] == 1
        with ThreadPoolExecutor() as worker:
            task = worker.submit(coordinator.submit, 'b', 1, 1, {'p.lora_A': torch.ones(1)})
            try:
                assert entered.wait(3)
                current = read()
                assert current['phase'] == 'AGGREGATING'
                assert current['received'] == current['quorum'] == 2
            finally:
                release.set()
            task.result(timeout=3)
        final = read()
        assert final['phase'] == 'COMPLETE'
        assert [e['phase'] for e in final['events']] == ['IDLE','WAITING','WAITING','AGGREGATING','COMPLETE']
        assert final['runId'] == 'status-test'
    finally:
        release.set()
        server.shutdown()
        server.server_close()


def test_failed_checkpoint_is_not_reported_complete(monkeypatch):
    coordinator = RoundCoordinator(total_rounds=1, submission_quorum=1)
    coordinator.bootstrap('a', {'p.lora_A': torch.zeros(1)})
    def fail(*_):
        raise OSError('disk unavailable')
    monkeypatch.setattr(coordinator, '_save_checkpoint', fail)
    with pytest.raises(OSError):
        coordinator.submit('a', 1, 1, {'p.lora_A': torch.ones(1)})
    assert coordinator.status.snapshot()['phase'] == 'FAILED'
