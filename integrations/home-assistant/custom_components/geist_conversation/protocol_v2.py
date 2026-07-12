"""Dependency-free Geist Home protocol-v2 framing for Home Assistant."""

from __future__ import annotations

import json
import re
import struct
from typing import Any

VERSION = 2
PREFIX_BYTES = 4
MAX_PAYLOAD = 1024 * 1024
MAX_DEPTH = 32
REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
FRAME_TYPES = {
    "hello",
    "health",
    "registry.replace",
    "conversation.start",
    "tool.call",
    "tool.result",
    "conversation.result",
    "cancel",
}


class ProtocolError(ValueError):
    """A stable, fail-closed protocol error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("duplicate_field")
        result[key] = value
    return result


def _constant(_value: str) -> None:
    raise ProtocolError("invalid_json")


def _depth(value: Any, current: int = 0) -> int:
    if current > MAX_DEPTH:
        raise ProtocolError("invalid_json")
    if isinstance(value, dict):
        for child in value.values():
            _depth(child, current + 1)
    elif isinstance(value, list):
        for child in value:
            _depth(child, current + 1)
    return current


def validate_envelope(frame: Any) -> dict[str, Any]:
    if not isinstance(frame, dict):
        raise ProtocolError("invalid_json")
    missing = {"version", "request_id", "type"} - frame.keys()
    if missing:
        raise ProtocolError("missing_field")
    version = frame["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ProtocolError("invalid_field")
    if version != VERSION:
        raise ProtocolError("unsupported_version")
    request_id = frame["request_id"]
    if not isinstance(request_id, str) or REQUEST_ID.fullmatch(request_id) is None:
        raise ProtocolError("invalid_field")
    frame_type = frame["type"]
    if not isinstance(frame_type, str):
        raise ProtocolError("invalid_field")
    if frame_type not in FRAME_TYPES:
        raise ProtocolError("unknown_type")
    _depth(frame)
    return frame


def encode_frame(frame: dict[str, Any]) -> bytes:
    validate_envelope(frame)
    try:
        payload = json.dumps(
            frame,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as err:
        raise ProtocolError("invalid_json") from err
    if not payload or len(payload) > MAX_PAYLOAD:
        raise ProtocolError("frame_too_large")
    return struct.pack(">I", len(payload)) + payload


def decode_frame(data: bytes) -> tuple[dict[str, Any], int]:
    if not isinstance(data, bytes):
        raise ProtocolError("invalid_arg")
    if len(data) < PREFIX_BYTES:
        raise ProtocolError("need_more")
    payload_len = struct.unpack(">I", data[:PREFIX_BYTES])[0]
    if payload_len > MAX_PAYLOAD:
        raise ProtocolError("frame_too_large")
    if payload_len == 0:
        raise ProtocolError("invalid_json")
    total = PREFIX_BYTES + payload_len
    if len(data) < total:
        raise ProtocolError("need_more")
    try:
        text = data[PREFIX_BYTES:total].decode("utf-8", errors="strict")
        frame = json.loads(
            text,
            object_pairs_hook=_object,
            parse_constant=_constant,
        )
    except ProtocolError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as err:
        raise ProtocolError("invalid_json") from err
    return validate_envelope(frame), total
