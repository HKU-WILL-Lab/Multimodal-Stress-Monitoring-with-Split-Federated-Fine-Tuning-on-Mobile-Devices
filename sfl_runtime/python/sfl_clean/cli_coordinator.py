"""Run the prefix round coordinator service."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .coordinator import RoundCoordinator
from .grpc_services import CoordinatorRpcService, make_server
from .proto_runtime import load_bindings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
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
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=5.0).wait()


if __name__ == "__main__":
    main()

