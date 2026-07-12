"""Model-free health handshake for the local dynamic-tools socket."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
from typing import Any, NamedTuple

PROTOCOL = "dynamic-tools-v1"
REQUEST = b'{"type":"health"}\n'
MAX_RESPONSE_BYTES = 512


class HealthResult(NamedTuple):
    protocol: str
    status: str


class HealthError(ValueError):
    """Stable config-flow validation error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


async def async_validate_health(socket_path: str, timeout_s: float) -> HealthResult:
    """Require a ready dynamic-tools-v1 daemon at ``socket_path``."""
    if not isinstance(socket_path, str) or not socket_path.startswith("/"):
        raise HealthError("invalid_socket")
    writer: Any | None = None
    try:
        async with asyncio.timeout(timeout_s):
            reader, writer = await asyncio.open_unix_connection(socket_path)
            writer.write(REQUEST)
            await writer.drain()
            line = await reader.readline()
    except TimeoutError as err:
        raise HealthError("timeout") from err
    except OSError as err:
        raise HealthError("cannot_connect") from err
    finally:
        if writer is not None:
            writer.close()
            with suppress(OSError):
                await writer.wait_closed()

    if not line or len(line) > MAX_RESPONSE_BYTES or not line.endswith(b"\n"):
        raise HealthError("invalid_response")
    try:
        response = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise HealthError("invalid_response") from err
    if not isinstance(response, dict) or response.get("type") != "health.result":
        raise HealthError("invalid_response")
    if response.get("protocol") != PROTOCOL:
        raise HealthError("unsupported_protocol")
    if response.get("status") != "ready":
        raise HealthError("not_ready")
    return HealthResult(protocol=PROTOCOL, status="ready")
