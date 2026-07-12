"""Bidirectional newline-JSON session using HA as a dynamic-tools-v1 host."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from .dynamic_tools_v1 import async_handle_dynamic_tool_call, build_dynamic_tools
from .policy import ExposureStore, PolicyError


class ProtocolError(ValueError):
    """Stable fail-closed error for the dynamic JSON session."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code

MAX_LINE_BYTES = 131072


class RequestGate:
    """Zero-queue admission gate for one resident model runtime."""

    def __init__(self) -> None:
        self._active = False

    def enter(self) -> None:
        if self._active:
            raise ProtocolError("busy")
        self._active = True

    def leave(self) -> None:
        self._active = False


async def _write(writer: Any, frame: dict[str, Any]) -> None:
    payload = json.dumps(frame, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(payload) > MAX_LINE_BYTES:
        raise ProtocolError("frame_too_large")
    writer.write(payload)
    await writer.drain()


async def _read(reader: Any) -> dict[str, Any]:
    line = await reader.readline()
    if not line:
        raise ProtocolError("connection_closed")
    if len(line) > MAX_LINE_BYTES or not line.endswith(b"\n"):
        raise ProtocolError("frame_too_large")
    try:
        frame = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ProtocolError("invalid_json") from err
    if not isinstance(frame, dict):
        raise ProtocolError("invalid_json")
    return frame


async def async_dynamic_session(
    reader: Any,
    writer: Any,
    utterance: str,
    exposure: ExposureStore,
    executor: Any,
    *,
    timeout_s: float,
    max_tool_steps: int = 4,
    allow_high_impact: bool = False,
    language: str = "",
    context: str = "",
) -> str:
    if not isinstance(utterance, str) or not utterance.strip() or not 1 <= max_tool_steps <= 16:
        raise ProtocolError("invalid_session")
    version = exposure.version
    request = {"input": utterance.strip(), "max_tool_steps": max_tool_steps,
               "tools": build_dynamic_tools(exposure, allow_high_impact=allow_high_impact)}
    if language:
        request["language"] = language
    if context:
        request["context"] = context
    calls = 0
    active_call_id: str | None = None
    try:
        async with asyncio.timeout(timeout_s):
            await _write(writer, request)
            while True:
                frame = await _read(reader)
                if frame.get("type") == "conversation.result":
                    if set(frame) != {"type", "text"} or not isinstance(frame.get("text"), str):
                        raise ProtocolError("invalid_conversation_result")
                    return frame["text"].strip()
                calls += 1
                if calls > max_tool_steps:
                    raise ProtocolError("tool_budget_exceeded")
                active_call_id = frame.get("call_id") if isinstance(frame.get("call_id"), str) else None
                try:
                    result = await async_handle_dynamic_tool_call(
                        frame, exposure, executor, registry_version=version,
                        allow_high_impact=allow_high_impact)
                except PolicyError as err:
                    call_id = frame.get("call_id", "invalid")
                    result = {"type": "tool.result", "call_id": call_id,
                              "status": err.status, "result": {}}
                await _write(writer, result)
                active_call_id = None
    except TimeoutError as err:
        if active_call_id is not None:
            try:
                await _write(writer, {"type": "cancel", "call_id": active_call_id,
                                      "reason": "timeout"})
            except (OSError, ProtocolError):
                pass
        raise ProtocolError("timeout") from err
    except asyncio.CancelledError:
        if active_call_id is not None:
            try:
                await asyncio.shield(_write(
                    writer, {"type": "cancel", "call_id": active_call_id,
                             "reason": "client_cancelled"}))
            except (OSError, ProtocolError):
                pass
        raise


async def async_ask_geist_dynamic(
    socket_path: str,
    utterance: str,
    exposure: ExposureStore,
    executor: Any,
    *,
    timeout_s: float,
    max_tool_steps: int = 4,
    allow_high_impact: bool = False,
    language: str = "",
    context: str = "",
) -> str:
    reader, writer = await asyncio.open_unix_connection(socket_path)
    try:
        return await async_dynamic_session(reader, writer, utterance, exposure, executor,
                                           timeout_s=timeout_s, max_tool_steps=max_tool_steps,
                                           allow_high_impact=allow_high_impact,
                                           language=language, context=context)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass
