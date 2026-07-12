#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "integrations/home-assistant/custom_components/geist_conversation"


def load(name: str):
    package_name = "geist_dynamic_contract"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(COMPONENT)]
        sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", COMPONENT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


policy = load("policy")
dynamic = load("dynamic_tools_v1")
session = load("dynamic_session_v1")


class Executor:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def get_state(self, entity_id: str):
        self.calls.append(("get", entity_id))
        return {"state": "off"}

    async def async_call_service(self, domain: str, service: str, entity_id: str, arguments: dict):
        self.calls.append((domain, service, entity_id, arguments))
        return []


class SlowExecutor(Executor):
    async def async_call_service(self, domain: str, service: str, entity_id: str, arguments: dict):
        await asyncio.sleep(1)
        return []


class Reader:
    def __init__(self, frames: list[dict]) -> None:
        self.frames = iter(frames)

    async def readline(self) -> bytes:
        try:
            return json.dumps(next(self.frames), separators=(",", ":")).encode() + b"\n"
        except StopIteration:
            return b""


class Writer:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    def write(self, payload: bytes) -> None:
        self.frames.append(json.loads(payload))

    async def drain(self) -> None:
        pass


async def checks() -> None:
    exposure = policy.ExposureStore(frozenset({"light.kitchen", "sensor.outdoor", "lock.front"}), 7)
    names = {tool["name"] for tool in dynamic.build_dynamic_tools(exposure)}
    assert {"HassGetState", "HassTurnOn", "HassTurnOff", "HassSetBrightness"} <= names
    assert "HassUnlock" not in names
    assert "HassUnlock" in {tool["name"] for tool in dynamic.build_dynamic_tools(
        exposure, allow_high_impact=True)}
    executor = Executor()
    frame = {"type": "tool.call", "call_id": "1", "name": "HassSetBrightness",
             "arguments": {"name": "light.kitchen", "brightness_pct": 42}}
    result = await dynamic.async_handle_dynamic_tool_call(
        frame, exposure, executor, registry_version=7)
    assert result == {"type": "tool.result", "call_id": "1", "status": "ok", "result": []}
    assert executor.calls == [("light", "turn_on", "light.kitchen", {"brightness_pct": 42})]
    try:
        await dynamic.async_handle_dynamic_tool_call(
            {"type": "tool.call", "call_id": "2", "name": "HassUnlock",
             "arguments": {"name": "lock.front"}}, exposure, executor, registry_version=7)
    except policy.PolicyError as err:
        assert err.status == "denied"
    else:
        raise AssertionError("offered-name gate failed")
    exposure.replace({"sensor.outdoor"})
    try:
        await dynamic.async_handle_dynamic_tool_call(frame, exposure, executor, registry_version=7)
    except policy.PolicyError as err:
        assert err.status == "denied"
    else:
        raise AssertionError("revoked exposure executed")

    exposure = policy.ExposureStore(frozenset({"light.kitchen"}), 3)
    writer = Writer()
    session_executor = Executor()
    answer = await session.async_dynamic_session(
        Reader([{"type": "tool.call", "call_id": "a", "name": "HassTurnOn",
                 "arguments": {"name": "light.kitchen"}},
                {"type": "tool.call", "call_id": "b", "name": "HassGetState",
                 "arguments": {"name": "light.kitchen"}},
                {"type": "conversation.result", "text": "Kitchen is on."}]),
        writer, "Turn on kitchen and report its state", exposure, session_executor,
        timeout_s=1, max_tool_steps=3)
    assert answer == "Kitchen is on."
    assert writer.frames[0]["tools"] and writer.frames[0]["max_tool_steps"] == 3
    assert writer.frames[1] == {"type": "tool.result", "call_id": "a", "status": "ok", "result": []}
    assert writer.frames[2] == {"type": "tool.result", "call_id": "b", "status": "ok",
                                "result": {"state": "off"}}

    try:
        await session.async_dynamic_session(
            Reader([{"type": "tool.call", "call_id": "1", "name": "HassTurnOn",
                     "arguments": {"name": "light.kitchen"}},
                    {"type": "tool.call", "call_id": "2", "name": "HassTurnOff",
                     "arguments": {"name": "light.kitchen"}}]),
            Writer(), "Do two things", exposure, Executor(), timeout_s=1, max_tool_steps=1)
    except session.ProtocolError as err:
        assert err.code == "tool_budget_exceeded"
    else:
        raise AssertionError("tool budget was not enforced")

    timeout_writer = Writer()
    try:
        await session.async_dynamic_session(
            Reader([{"type": "tool.call", "call_id": "slow", "name": "HassTurnOn",
                     "arguments": {"name": "light.kitchen"}}]),
            timeout_writer, "Turn on kitchen", exposure, SlowExecutor(),
            timeout_s=0.01, max_tool_steps=2)
    except session.ProtocolError as err:
        assert err.code == "timeout"
    else:
        raise AssertionError("timeout did not cancel session")
    assert timeout_writer.frames[-1] == {
        "type": "cancel", "call_id": "slow", "reason": "timeout"}


if __name__ == "__main__":
    asyncio.run(checks())
    print("ha_dynamic_tools_v1: pass")
