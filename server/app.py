# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Adaptive Supply Chain RL Environment.

Endpoints:
    POST /reset  — Reset the environment (accepts optional task/seed params)
    POST /step   — Execute an action
    GET  /state  — Get current environment state
    GET  /schema — Get action/observation schemas
    WS   /ws     — WebSocket endpoint for persistent sessions

Usage:
    # Development (auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import SupplyChainAction, SupplyChainObservation
    from .asc_agent_under_demand_uncertainity_rl_env_environment import (
        AscAgentUnderDemandUncertainityRlEnvironment,
    )
except (ModuleNotFoundError, ImportError):
    from models import SupplyChainAction, SupplyChainObservation
    from server.asc_agent_under_demand_uncertainity_rl_env_environment import (
        AscAgentUnderDemandUncertainityRlEnvironment,
    )


app = create_app(
    AscAgentUnderDemandUncertainityRlEnvironment,
    SupplyChainAction,
    SupplyChainObservation,
    env_name="asc_agent_under_demand_uncertainity_rl_env",
    max_concurrent_envs=4,  # supports concurrent WebSocket sessions
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution via uv run or python -m."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Adaptive Supply Chain RL Environment server")
    parser.add_argument("--host", type=str, default=host)
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
