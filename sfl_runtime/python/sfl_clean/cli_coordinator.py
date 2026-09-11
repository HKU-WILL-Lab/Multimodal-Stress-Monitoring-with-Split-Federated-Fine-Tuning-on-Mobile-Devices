"""Run the prefix round coordinator service."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .coordinator import RoundCoordinator
from .grpc_services import CoordinatorRpcService, make_server
from .proto_runtime import load_bindings
from .aggregation_status import serve_status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--status-bind", help="Optional read-only aggregation status HTTP address, e.g. 127.0.0.1:50053")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    pb2, pb2_grpc = load_bindings()
    coordinator = RoundCoordinator(
        total_rounds=config.training.total_rounds,
        submission_quorum=config.training.submission_quorum,
        checkpoint_root=config.checkpoint_root / config.run_id,
    )
    service = CoordinatorRpcService(pb2, coordinator)
    server = make_server(
        config.coordinator_rpc.bind,
        config.coordinator_rpc.max_message_bytes,
        config.coordinator_rpc.worker_threads,
    )
    pb2_grpc.add_RoundCoordinatorServicer_to_server(service, server)
    server.start()
    status_server = serve_status(coordinator.status, arguments.status_bind, config.run_id) if arguments.status_bind else None
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=5.0).wait()
    finally:
        if status_server:
            status_server.shutdown()
            status_server.server_close()


if __name__ == "__main__":
    main()
