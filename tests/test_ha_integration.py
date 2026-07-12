#!/usr/bin/env python3
"""Model-free contract checks for the Home Assistant integration artifact."""

from __future__ import annotations

import importlib.util
import json
import socket
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPONENT = (
    ROOT / "integrations/home-assistant/custom_components/geist_conversation"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_registry():
    path = COMPONENT / "registry.py"
    spec = importlib.util.spec_from_file_location("geist_registry_contract", path)
    require(spec is not None and spec.loader is not None, "registry module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_transport():
    path = COMPONENT / "transport.py"
    spec = importlib.util.spec_from_file_location("geist_transport_contract", path)
    require(spec is not None and spec.loader is not None, "transport module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_transport_roundtrip() -> None:
    transport = load_transport()

    class FakeSocket:
        def __init__(self) -> None:
            self.timeout = None
            self.path = None
            self.sent = b""
            self.shutdown_how = None
            self.responses = iter(["Licht ".encode(), "ist an.\n".encode(), b""])

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def settimeout(self, timeout: float) -> None:
            self.timeout = timeout

        def connect(self, path: str) -> None:
            self.path = path

        def sendall(self, data: bytes) -> None:
            self.sent += data

        def shutdown(self, how: int) -> None:
            self.shutdown_how = how

        def recv(self, _cap: int) -> bytes:
            return next(self.responses)

    fake = FakeSocket()
    original_socket = transport.socket.socket
    transport.socket.socket = lambda *_args: fake
    try:
        answer = transport.ask_geist("/config/geist.sock", "Licht\nim Flur", 2)
    finally:
        transport.socket.socket = original_socket

    require(fake.timeout == 2, "transport timeout")
    require(fake.path == "/config/geist.sock", "Unix-socket path")
    require(fake.sent == b"Licht im Flur\n", "utterance is one newline-terminated line")
    require(fake.shutdown_how == socket.SHUT_WR, "client half-closes its request")
    require(answer == "Licht ist an.", "chunked EOF-framed response")


def main() -> None:
    expected = {
        "__init__.py",
        "config_flow.py",
        "const.py",
        "conversation.py",
        "ha_executor.py",
        "manifest.json",
        "policy.py",
        "protocol_v2.py",
        "registry.py",
        "transport.py",
    }
    require(expected <= {p.name for p in COMPONENT.iterdir()}, "component is complete")

    manifest = json.loads((COMPONENT / "manifest.json").read_text())
    require(manifest["domain"] == "geist_conversation", "manifest domain")
    require(manifest["iot_class"] == "local_push", "manifest transport class")
    require(manifest["requirements"] == [], "component has no Python dependencies")

    setup_source = (COMPONENT / "__init__.py").read_text()
    require("async_setup_registry_sync" in setup_source,
            "component activates exposed-entity registry synchronization")

    registry = load_registry()
    line = registry.format_line(
        "light.ceiling",
        "Wohnzimmer Deckenlicht",
        ["Deckenlicht"],
        "Wohnzimmer",
    )
    require(line is not None and line.startswith("light.ceiling | light |"), "light line")
    require("Wohnzimmer" in line and "Deckenlicht" in line, "area and alias included")
    require(registry.format_line("camera.door", "Door", None, None) is None,
            "unsupported domains stay unavailable")
    sanitized = registry.format_line("sensor.temp", "Temp | Bad", None, "Bad")
    require(sanitized == "sensor.temp | sensor | Temp Bad, Bad", "delimiter sanitized")
    check_transport_roundtrip()

    print("ha_integration: artifact + exposed-registry contract pass")


if __name__ == "__main__":
    main()
