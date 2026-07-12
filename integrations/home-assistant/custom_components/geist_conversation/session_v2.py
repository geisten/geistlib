"""Opt-in bidirectional protocol-v2 conversation session.

This is not wired into ``conversation.py`` yet. It is the shared stream loop
for a future Unix-socket and private app-network transport.
"""

from __future__ import annotations

import asyncio
import re
import struct
import uuid
from typing import Any

from .ha_executor import HomeAssistantExecutor, async_handle_tool_call
from .policy import ExposureStore
from .protocol_v2 import MAX_PAYLOAD, PREFIX_BYTES, ProtocolError, decode_frame, encode_frame

MAX_UTTERANCE_BYTES = 4096
MAX_RESULT_BYTES = 8192
DEFAULT_MAX_TOOL_CALLS = 8
LOCALE = re.compile(r"^[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})*$")


async def async_read_frame(reader: Any) -> dict[str, Any]:
    try:
        prefix = await reader.readexactly(PREFIX_BYTES)
    except asyncio.IncompleteReadError as err:
        raise ProtocolError("connection_closed") from err
    payload_len = struct.unpack(">I", prefix)[0]
    if payload_len > MAX_PAYLOAD:
        raise ProtocolError("frame_too_large")
    if payload_len == 0:
        raise ProtocolError("invalid_json")
    try:
        payload = await reader.readexactly(payload_len)
    except asyncio.IncompleteReadError as err:
        raise ProtocolError("connection_closed") from err
    frame, consumed = decode_frame(prefix + payload)
    if consumed != PREFIX_BYTES + payload_len:
        raise ProtocolError("invalid_json")
    return frame


async def async_write_frame(writer: Any, frame: dict[str, Any]) -> None:
    writer.write(encode_frame(frame))
    await writer.drain()


async def _async_cancel(writer: Any, request_id: str, reason: str) -> None:
    try:
        await async_write_frame(
            writer,
            {
                "version": 2,
                "request_id": request_id,
                "type": "cancel",
                "reason": reason,
            },
        )
    except (OSError, ProtocolError):
        pass


def _conversation_result(frame: dict[str, Any], request_id: str) -> str:
    if (
        set(frame) != {"version", "request_id", "type", "text"}
        or frame["type"] != "conversation.result"
        or frame["request_id"] != request_id
        or not isinstance(frame["text"], str)
        or not frame["text"].strip()
        or len(frame["text"].encode("utf-8")) > MAX_RESULT_BYTES
    ):
        raise ProtocolError("invalid_conversation_result")
    return frame["text"].strip()


async def _async_session(
    reader: Any,
    writer: Any,
    utterance: str,
    locale: str,
    exposure: ExposureStore,
    executor: HomeAssistantExecutor,
    request_id: str,
    max_tool_calls: int,
    allow_high_impact: bool,
) -> str:
    await async_write_frame(
        writer,
        {
            "version": 2,
            "request_id": request_id,
            "type": "conversation.start",
            "utterance": utterance,
            "locale": locale,
            "registry_version": exposure.version,
            "max_tool_calls": max_tool_calls,
        },
    )
    tool_calls = 0
    seen_tool_ids: set[str] = set()
    while True:
        frame = await async_read_frame(reader)
        if frame["type"] == "conversation.result":
            return _conversation_result(frame, request_id)
        if frame["type"] != "tool.call":
            raise ProtocolError("unexpected_frame")
        tool_calls += 1
        if frame["request_id"] in seen_tool_ids:
            raise ProtocolError("duplicate_request_id")
        seen_tool_ids.add(frame["request_id"])
        if tool_calls > max_tool_calls:
            await async_write_frame(
                writer,
                {
                    "version": 2,
                    "request_id": frame["request_id"],
                    "type": "tool.result",
                    "status": "denied",
                    "result": {},
                },
            )
            raise ProtocolError("tool_budget_exceeded")
        result = await async_handle_tool_call(
            frame,
            exposure,
            executor,
            allow_high_impact=allow_high_impact,
        )
        await async_write_frame(writer, result)


async def async_conversation_session(
    reader: Any,
    writer: Any,
    utterance: str,
    locale: str,
    exposure: ExposureStore,
    executor: HomeAssistantExecutor,
    *,
    timeout_s: float,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    allow_high_impact: bool = False,
    request_id: str | None = None,
) -> str:
    try:
        utterance_bytes = len(utterance.encode("utf-8")) if isinstance(utterance, str) else 0
    except UnicodeEncodeError as err:
        raise ProtocolError("invalid_session") from err
    if (
        not isinstance(utterance, str)
        or not utterance.strip()
        or utterance_bytes > MAX_UTTERANCE_BYTES
        or not isinstance(locale, str)
        or LOCALE.fullmatch(locale) is None
        or len(locale) > 32
        or isinstance(max_tool_calls, bool)
        or not isinstance(max_tool_calls, int)
        or not 1 <= max_tool_calls <= 32
        or isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or timeout_s <= 0
    ):
        raise ProtocolError("invalid_session")
    if request_id is None:
        request_id = f"conv-{uuid.uuid4().hex}"
    try:
        async with asyncio.timeout(timeout_s):
            return await _async_session(
                reader,
                writer,
                utterance.strip(),
                locale,
                exposure,
                executor,
                request_id,
                max_tool_calls,
                allow_high_impact,
            )
    except TimeoutError as err:
        await _async_cancel(writer, request_id, "timeout")
        raise ProtocolError("timeout") from err
    except asyncio.CancelledError:
        await asyncio.shield(_async_cancel(writer, request_id, "client_cancelled"))
        raise
    except ProtocolError as err:
        await _async_cancel(writer, request_id, err.code)
        raise


async def async_ask_geist_v2(
    socket_path: str,
    utterance: str,
    locale: str,
    exposure: ExposureStore,
    executor: HomeAssistantExecutor,
    *,
    timeout_s: float,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    allow_high_impact: bool = False,
) -> str:
    reader, writer = await asyncio.open_unix_connection(socket_path)
    try:
        return await async_conversation_session(
            reader,
            writer,
            utterance,
            locale,
            exposure,
            executor,
            timeout_s=timeout_s,
            max_tool_calls=max_tool_calls,
            allow_high_impact=allow_high_impact,
        )
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass
