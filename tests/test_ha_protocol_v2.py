#!/usr/bin/env python3
"""Model-free Python protocol-v2 and HA policy boundary tests."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import types
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "integrations/home-assistant/custom_components/geist_conversation"


def load(name: str):
    package_name = "geist_contract"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(COMPONENT)]
        sys.modules[package_name] = package
    path = COMPONENT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = load("protocol_v2")
policy = load("policy")
ha_executor = load("ha_executor")


def expect_error(code: str, function, *args) -> None:
    try:
        function(*args)
    except protocol.ProtocolError as err:
        assert err.code == code, (err.code, code)
    else:
        raise AssertionError(f"expected ProtocolError({code})")


class FakeHA:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.fail = False
        self.scalar = False

    def get_state(self, entity_id: str):
        self.calls.append(("get_state", entity_id))
        if self.fail:
            raise OSError("offline")
        return "bad" if self.scalar else {"state": "off"}

    def call_service(self, domain: str, service: str, entity_id: str, arguments: dict):
        self.calls.append(("call_service", domain, service, entity_id, arguments))
        if self.fail:
            raise OSError("offline")
        return []


class FakeState:
    def __init__(self) -> None:
        self.state = "heat"
        self.attributes = {
            "temperature": 21,
            "current_temperature": 20.5,
            "unit_of_measurement": "°C",
            "friendly_name": "Private room name",
            "secret_nested": {"token": "must not leave HA"},
        }


class FakeStates:
    def __init__(self) -> None:
        self.value = FakeState()

    def get(self, entity_id: str):
        return self.value if entity_id in {"light.flur", "climate.bad"} else None


class FakeServices:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.error: Exception | None = None

    async def async_call(self, domain, service, data, **kwargs) -> None:
        self.calls.append((domain, service, data, kwargs))
        if self.error is not None:
            raise self.error


class FakeHass:
    def __init__(self) -> None:
        self.states = FakeStates()
        self.services = FakeServices()


class Unauthorized(Exception):
    pass


def base_call(store, **changes):
    frame = {
        "version": 2,
        "request_id": "exec-1",
        "type": "tool.call",
        "operation": "get_state",
        "entity_id": "light.flur",
        "domain": "light",
        "registry_version": store.version,
    }
    frame.update(changes)
    return frame


def check_codec() -> None:
    frame = {"version": 2, "request_id": "req-1", "type": "hello"}
    wire = protocol.encode_frame(frame)
    payload = b'{"version":2,"request_id":"req-1","type":"hello"}'
    assert wire == struct.pack(">I", len(payload)) + payload
    decoded, consumed = protocol.decode_frame(wire + wire)
    assert decoded == frame and consumed == len(wire)

    expect_error("need_more", protocol.decode_frame, wire[:3])
    expect_error("need_more", protocol.decode_frame, wire[:-1])
    expect_error("invalid_json", protocol.decode_frame, b"\0\0\0\0")
    expect_error(
        "frame_too_large",
        protocol.decode_frame,
        struct.pack(">I", protocol.MAX_PAYLOAD + 1),
    )
    duplicate = b'{"version":2,"request_id":"x","type":"hello","type":"health"}'
    expect_error(
        "duplicate_field",
        protocol.decode_frame,
        struct.pack(">I", len(duplicate)) + duplicate,
    )
    invalid_utf8 = b'{"version":2,"request_id":"\xc0\xaf","type":"hello"}'
    expect_error(
        "invalid_json",
        protocol.decode_frame,
        struct.pack(">I", len(invalid_utf8)) + invalid_utf8,
    )
    expect_error(
        "invalid_field",
        protocol.encode_frame,
        {"version": True, "request_id": "x", "type": "hello"},
    )
    expect_error(
        "unknown_type",
        protocol.encode_frame,
        {"version": 2, "request_id": "x", "type": "shell.exec"},
    )
    expect_error(
        "invalid_json",
        protocol.encode_frame,
        {"version": 2, "request_id": "x", "type": "hello", "value": float("nan")},
    )


def check_policy() -> None:
    store = policy.ExposureStore()
    assert store.replace({"light.flur", "climate.bad", "lock.front"}) == 1
    fake = FakeHA()

    get = base_call(store)
    result = policy.handle_tool_call(get, store, fake)
    assert result["status"] == "ok" and result["result"] == {"state": "off"}
    assert fake.calls == [("get_state", "light.flur")]

    set_light = base_call(
        store,
        operation="call_service",
        service="turn_on",
        arguments={"brightness_pct": 40},
    )
    result = policy.handle_tool_call(set_light, store, fake)
    assert result["status"] == "ok"
    assert fake.calls[-1] == (
        "call_service",
        "light",
        "turn_on",
        "light.flur",
        {"brightness_pct": 40},
    )

    validated_set = policy.validate_tool_call(set_light, store)
    set_light["arguments"]["brightness_pct"] = 200
    policy.execute_tool_call(validated_set, store, fake)
    assert fake.calls[-1][-1] == {"brightness_pct": 40}

    before = len(fake.calls)
    unexposed = base_call(store, entity_id="light.keller")
    assert policy.handle_tool_call(unexposed, store, fake)["status"] == "denied"
    assert len(fake.calls) == before

    stale = base_call(store, registry_version=store.version - 1)
    assert policy.handle_tool_call(stale, store, fake)["status"] == "denied"
    assert len(fake.calls) == before

    # Validate, then unexpose before execution: the final boundary rechecks.
    validated = policy.validate_tool_call(get, store)
    store.replace({"climate.bad", "lock.front"})
    try:
        policy.execute_tool_call(validated, store, fake)
    except policy.PolicyError as err:
        assert err.status == "denied"
    else:
        raise AssertionError("unexpose race must fail closed")
    assert len(fake.calls) == before

    store.replace({"light.flur", "climate.bad", "lock.front"})
    wrong_domain = base_call(store, domain="switch")
    assert policy.handle_tool_call(wrong_domain, store, fake)["status"] == "invalid_request"

    bad_request_id = base_call(store, request_id="bad id")
    assert policy.handle_tool_call(bad_request_id, store, fake)["status"] == "invalid_request"

    injected = base_call(store, service="unlock")
    assert policy.handle_tool_call(injected, store, fake)["status"] == "denied"

    bad_brightness = base_call(
        store,
        operation="call_service",
        service="turn_on",
        arguments={"brightness_pct": True},
    )
    assert policy.handle_tool_call(bad_brightness, store, fake)["status"] == "denied"

    climate = base_call(
        store,
        operation="call_service",
        entity_id="climate.bad",
        domain="climate",
        service="set_temperature",
        arguments={"temperature": 101},
    )
    assert policy.handle_tool_call(climate, store, fake)["status"] == "denied"

    lock = base_call(
        store,
        operation="call_service",
        entity_id="lock.front",
        domain="lock",
        service="unlock",
        arguments={},
    )
    assert policy.handle_tool_call(lock, store, fake)["status"] == "denied"
    assert policy.handle_tool_call(lock, store, fake, allow_high_impact=True)["status"] == "ok"

    fake.fail = True
    current_get = base_call(store)
    assert policy.handle_tool_call(current_get, store, fake)["status"] == "unavailable"
    fake.fail = False
    fake.scalar = True
    assert policy.handle_tool_call(current_get, store, fake)["status"] == "invalid_request"


async def check_ha_executor() -> None:
    store = policy.ExposureStore()
    store.replace({"light.flur", "climate.bad"})
    hass = FakeHass()
    context = object()
    executor = ha_executor.HomeAssistantExecutor(hass, context)

    climate_get = base_call(
        store, entity_id="climate.bad", domain="climate"
    )
    result = await ha_executor.async_handle_tool_call(climate_get, store, executor)
    assert result["status"] == "ok"
    assert result["result"] == {
        "state": "heat",
        "attributes": {
            "temperature": 21,
            "current_temperature": 20.5,
            "unit_of_measurement": "°C",
        },
    }
    assert "friendly_name" not in result["result"]["attributes"]

    service = base_call(
        store,
        operation="call_service",
        entity_id="climate.bad",
        domain="climate",
        service="set_temperature",
        arguments={"temperature": 22},
    )
    result = await ha_executor.async_handle_tool_call(service, store, executor)
    assert result["status"] == "ok"
    assert hass.services.calls[-1] == (
        "climate",
        "set_temperature",
        {"entity_id": "climate.bad", "temperature": 22},
        {"blocking": True, "context": context},
    )

    validated = policy.validate_tool_call(service, store)
    store.replace({"light.flur"})
    try:
        await ha_executor.async_execute_tool_call(validated, store, executor)
    except policy.PolicyError as err:
        assert err.status == "denied"
    else:
        raise AssertionError("async unexpose race must fail closed")
    assert len(hass.services.calls) == 1

    store.replace({"light.flur", "climate.bad"})
    current_service = dict(service, registry_version=store.version)
    hass.services.error = Unauthorized("no permission")
    result = await ha_executor.async_handle_tool_call(current_service, store, executor)
    assert result["status"] == "denied"
    hass.services.error = OSError("offline")
    result = await ha_executor.async_handle_tool_call(current_service, store, executor)
    assert result["status"] == "unavailable"

    hass.services.error = None
    hass.states.value = None
    current_get = dict(climate_get, registry_version=store.version)
    result = await ha_executor.async_handle_tool_call(current_get, store, executor)
    assert result["status"] == "unavailable"


def main() -> None:
    check_codec()
    check_policy()
    asyncio.run(check_ha_executor())
    print("ha_protocol_v2: framing + exposure/action policy + HA executor pass")


if __name__ == "__main__":
    main()
