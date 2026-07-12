"""HA-owned exposure and action policy for protocol-v2 tool calls.

This module deliberately imports no Home Assistant package. The integration
adapts HA states/services to ``StateServiceExecutor``; contract tests use a
fake. Authorization is checked both during validation and immediately before
execution so an unexpose or registry replacement fails closed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

ENTITY_ID = re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+$")
REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
READ_DOMAINS = {
    "light",
    "switch",
    "climate",
    "cover",
    "media_player",
    "lock",
    "fan",
    "sensor",
    "binary_sensor",
}


class PolicyError(ValueError):
    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


class StateServiceExecutor(Protocol):
    def get_state(self, entity_id: str) -> dict[str, Any]: ...

    def call_service(
        self, domain: str, service: str, entity_id: str, arguments: dict[str, Any]
    ) -> list[Any] | dict[str, Any]: ...


@dataclass
class ExposureStore:
    entities: frozenset[str] = field(default_factory=frozenset)
    version: int = 0

    def replace(self, entities: set[str]) -> int:
        self.entities = frozenset(entities)
        self.version += 1
        return self.version

    def contains(self, entity_id: str) -> bool:
        return entity_id in self.entities


@dataclass(frozen=True)
class ValidatedToolCall:
    request_id: str
    operation: str
    entity_id: str
    domain: str
    service: str | None
    arguments: dict[str, Any]
    registry_version: int


def _number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _empty(arguments: dict[str, Any]) -> bool:
    return len(arguments) == 0


def _validate_action(
    domain: str,
    service: str,
    arguments: dict[str, Any],
    allow_high_impact: bool,
) -> None:
    if domain == "light":
        if service == "turn_off" and _empty(arguments):
            return
        if service == "turn_on" and (
            _empty(arguments)
            or (
                set(arguments) == {"brightness_pct"}
                and _number(arguments["brightness_pct"])
                and 0 <= arguments["brightness_pct"] <= 100
            )
        ):
            return
    elif domain in {"switch", "media_player"}:
        if service in {"turn_on", "turn_off"} and _empty(arguments):
            return
    elif domain == "climate":
        if (
            service == "set_temperature"
            and set(arguments) == {"temperature"}
            and _number(arguments["temperature"])
            and -50 <= arguments["temperature"] <= 100
        ):
            return
    elif domain == "cover":
        if service in {"open_cover", "close_cover"} and _empty(arguments):
            return
    elif domain == "lock" and allow_high_impact:
        if service in {"lock", "unlock"} and _empty(arguments):
            return
    raise PolicyError("denied")


def validate_tool_call(
    frame: dict[str, Any],
    exposure: ExposureStore,
    *,
    allow_high_impact: bool = False,
) -> ValidatedToolCall:
    if frame.get("version") != 2 or frame.get("type") != "tool.call":
        raise PolicyError("invalid_request")
    request_id = frame.get("request_id")
    operation = frame.get("operation")
    entity_id = frame.get("entity_id")
    domain = frame.get("domain")
    registry_version = frame.get("registry_version")
    if (
        not isinstance(request_id, str)
        or REQUEST_ID.fullmatch(request_id) is None
        or operation not in {"get_state", "call_service"}
        or not isinstance(entity_id, str)
        or ENTITY_ID.fullmatch(entity_id) is None
        or not isinstance(domain, str)
        or entity_id.split(".", 1)[0] != domain
        or isinstance(registry_version, bool)
        or not isinstance(registry_version, int)
        or registry_version < 0
    ):
        raise PolicyError("invalid_request")
    if registry_version != exposure.version or not exposure.contains(entity_id):
        raise PolicyError("denied")

    base = {"version", "request_id", "type", "operation", "entity_id", "domain", "registry_version"}
    if operation == "get_state":
        if set(frame) != base or domain not in READ_DOMAINS:
            raise PolicyError("denied")
        return ValidatedToolCall(
            request_id, operation, entity_id, domain, None, {}, registry_version
        )

    if set(frame) != base | {"service", "arguments"}:
        raise PolicyError("invalid_request")
    service = frame.get("service")
    arguments = frame.get("arguments")
    if not isinstance(service, str) or not isinstance(arguments, dict):
        raise PolicyError("invalid_request")
    _validate_action(domain, service, arguments, allow_high_impact)
    return ValidatedToolCall(
        request_id,
        operation,
        entity_id,
        domain,
        service,
        dict(arguments),
        registry_version,
    )


def execute_tool_call(
    call: ValidatedToolCall,
    exposure: ExposureStore,
    executor: StateServiceExecutor,
) -> dict[str, Any]:
    # Final action-boundary check closes the validate/execute race.
    if call.registry_version != exposure.version or not exposure.contains(call.entity_id):
        raise PolicyError("denied")
    try:
        if call.operation == "get_state":
            result: Any = executor.get_state(call.entity_id)
        else:
            assert call.service is not None
            result = executor.call_service(
                call.domain, call.service, call.entity_id, dict(call.arguments)
            )
    except PolicyError:
        raise
    except Exception as err:
        raise PolicyError("unavailable") from err
    if not isinstance(result, (dict, list)):
        raise PolicyError("invalid_request")
    return {
        "version": 2,
        "request_id": call.request_id,
        "type": "tool.result",
        "status": "ok",
        "result": result,
    }


def handle_tool_call(
    frame: dict[str, Any],
    exposure: ExposureStore,
    executor: StateServiceExecutor,
    *,
    allow_high_impact: bool = False,
) -> dict[str, Any]:
    request_id = frame.get("request_id")
    if not isinstance(request_id, str):
        request_id = "invalid"
    try:
        call = validate_tool_call(
            frame, exposure, allow_high_impact=allow_high_impact
        )
        return execute_tool_call(call, exposure, executor)
    except PolicyError as err:
        return {
            "version": 2,
            "request_id": request_id,
            "type": "tool.result",
            "status": err.status,
            "result": {},
        }
