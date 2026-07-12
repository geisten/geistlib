#!/usr/bin/env python3
"""Model-free health-client contract tests."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "integrations/home-assistant/custom_components/geist_conversation/health.py"
SPEC = importlib.util.spec_from_file_location("geist_health_contract", PATH)
assert SPEC is not None and SPEC.loader is not None
health = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(health)


class Reader:
    def __init__(self, response: object) -> None:
        self.response = response

    async def readline(self) -> bytes:
        if isinstance(self.response, bytes):
            return self.response
        return json.dumps(self.response, separators=(",", ":")).encode() + b"\n"


class Writer:
    def __init__(self) -> None:
        self.payload = b""
        self.closed = False

    def write(self, payload: bytes) -> None:
        self.payload += payload

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass


async def expect(code: str, response: object | None = None, error: Exception | None = None) -> None:
    async def connect(_path: str):
        if error is not None:
            raise error
        return Reader(response), Writer()

    original = health.asyncio.open_unix_connection
    health.asyncio.open_unix_connection = connect
    try:
        await health.async_validate_health("/config/geist.sock", 0.1)
    except health.HealthError as err:
        assert err.code == code, (err.code, code)
    else:
        raise AssertionError(f"expected {code}")
    finally:
        health.asyncio.open_unix_connection = original


async def checks() -> None:
    writer = Writer()

    async def ready(_path: str):
        return Reader({"type": "health.result", "protocol": "dynamic-tools-v1",
                       "status": "ready"}), writer

    original = health.asyncio.open_unix_connection
    health.asyncio.open_unix_connection = ready
    try:
        await health.async_validate_health("/config/geist.sock", 0.1)
    finally:
        health.asyncio.open_unix_connection = original
    assert writer.payload == health.REQUEST and writer.closed

    await expect("cannot_connect", error=FileNotFoundError())
    await expect("timeout", error=TimeoutError())
    await expect("invalid_response", response=b"not-json\n")
    await expect("unsupported_protocol", response={"type": "health.result",
                 "protocol": "other", "status": "ready"})
    await expect("not_ready", response={"type": "health.result",
                 "protocol": "dynamic-tools-v1", "status": "loading"})
    try:
        await health.async_validate_health("relative.sock", 0.1)
    except health.HealthError as err:
        assert err.code == "invalid_socket"
    else:
        raise AssertionError("relative path accepted")


if __name__ == "__main__":
    asyncio.run(checks())
    print("ha_health: pass")
