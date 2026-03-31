"""Uvicorn server launcher for the cloud inference service."""

from __future__ import annotations

import argparse
import os

import uvicorn

from unisplit.shared.constants import DEFAULT_CLOUD_HOST, DEFAULT_CLOUD_PORT


def main() -> None:
    parser = argparse.ArgumentParser(description="UniSplit Cloud Inference Server")
    parser.add_argument("--host", default=None, help="Bind host")
    parser.add_argument("--port", type=int, default=None, help="Bind port")
    parser.add_argument("--config", default="configs/cloud.yaml", help="Config file path")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    host = args.host or os.environ.get("UNISPLIT_CLOUD_HOST", DEFAULT_CLOUD_HOST)
    port = args.port or int(os.environ.get("UNISPLIT_CLOUD_PORT", str(DEFAULT_CLOUD_PORT)))

    # Set config path as env var for the app factory to pick up
    os.environ["UNISPLIT_CONFIG_PATH"] = args.config

    uvicorn.run(
        "unisplit.cloud.server:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level="info",
    )


# App instance for uvicorn import
def _get_config_path() -> str | None:
    return os.environ.get("UNISPLIT_CONFIG_PATH", "configs/cloud.yaml")


from unisplit.cloud.app import create_app  # noqa: E402
app = create_app(_get_config_path())


if __name__ == "__main__":
    main()
