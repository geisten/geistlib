"""Home Assistant adapter for the host-neutral dynamic-tools-v1 contract."""

from __future__ import annotations

import re
from typing import Any

from .policy import ExposureStore, PolicyError

CALL_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
TURN_DOMAINS = {"light", "switch", "media_player", "fan"}


def _name_schema(entities: list[str]) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string"}
    if len(entities) <= 32:
        schema["enum"] = entities
    return schema


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def build_dynamic_tools(exposure: ExposureStore, *, allow_high_impact: bool = False) -> list[dict[str, Any]]:
    """Build only capabilities available in this request's exposure snapshot."""
    entities = sorted(exposure.entities)
    by_domain: dict[str, list[str]] = {}
    for entity_id in entities:
        by_domain.setdefault(entity_id.split(".", 1)[0], []).append(entity_id)
    tools: list[dict[str, Any]] = []
    if entities:
        tools.append(_tool("HassGetState", "Read one currently exposed Home Assistant entity",
                           {"name": _name_schema(entities)}, ["name"]))
    turnable = sorted(entity for domain in TURN_DOMAINS for entity in by_domain.get(domain, []))
    if turnable:
        tools.append(_tool("HassTurnOn", "Turn on one currently exposed Home Assistant entity",
                           {"name": _name_schema(turnable)}, ["name"]))
        tools.append(_tool("HassTurnOff", "Turn off one currently exposed Home Assistant entity",
                           {"name": _name_schema(turnable)}, ["name"]))
    if by_domain.get("light"):
        tools.append(_tool("HassSetBrightness", "Set brightness of an exposed light in percent",
                           {"name": _name_schema(by_domain["light"]),
                            "brightness_pct": {"type": "integer", "minimum": 0, "maximum": 100}},
                           ["name", "brightness_pct"]))
    if by_domain.get("climate"):
        tools.append(_tool("HassSetTemperature", "Set target temperature of an exposed climate entity",
                           {"name": _name_schema(by_domain["climate"]),
                            "temperature": {"type": "number", "minimum": -50, "maximum": 100}},
                           ["name", "temperature"]))
    if by_domain.get("cover"):
        cover = {"name": _name_schema(by_domain["cover"])}
        tools.append(_tool("HassOpenCover", "Open an exposed cover", cover, ["name"]))
        tools.append(_tool("HassCloseCover", "Close an exposed cover", cover, ["name"]))
    if allow_high_impact and by_domain.get("lock"):
        lock = {"name": _name_schema(by_domain["lock"])}
        tools.append(_tool("HassLock", "Lock an exposed lock", lock, ["name"]))
        tools.append(_tool("HassUnlock", "Unlock an exposed lock", lock, ["name"]))
    return tools


def _offered_names(tools: list[dict[str, Any]]) -> set[str]:
    return {tool["name"] for tool in tools}


async def async_handle_dynamic_tool_call(
    frame: dict[str, Any],
    exposure: ExposureStore,
    executor: Any,
    *,
    registry_version: int,
    allow_high_impact: bool = False,
) -> dict[str, Any]:
    """Validate again at the HA action boundary, execute, and correlate result."""
    call_id = frame.get("call_id")
    name = frame.get("name")
    arguments = frame.get("arguments")
    if (
        set(frame) != {"type", "call_id", "name", "arguments"}
        or frame.get("type") != "tool.call"
        or not isinstance(call_id, str)
        or CALL_ID.fullmatch(call_id) is None
        or not isinstance(name, str)
        or not isinstance(arguments, dict)
    ):
        raise PolicyError("invalid_request")
    tools = build_dynamic_tools(exposure, allow_high_impact=allow_high_impact)
    if name not in _offered_names(tools) or registry_version != exposure.version:
        raise PolicyError("denied")
    if set(arguments) - {"name", "brightness_pct", "temperature"}:
        raise PolicyError("invalid_request")
    entity_id = arguments.get("name")
    if not isinstance(entity_id, str) or not exposure.contains(entity_id):
        raise PolicyError("denied")
    domain = entity_id.split(".", 1)[0]
    if hasattr(executor, "is_exposed") and not executor.is_exposed(entity_id):
        raise PolicyError("denied")

    if name == "HassGetState" and set(arguments) == {"name"}:
        result = executor.get_state(entity_id)
    else:
        service: str
        service_args: dict[str, Any] = {}
        if name in {"HassTurnOn", "HassTurnOff"} and set(arguments) == {"name"}:
            if domain not in TURN_DOMAINS:
                raise PolicyError("denied")
            service = "turn_on" if name == "HassTurnOn" else "turn_off"
        elif name == "HassSetBrightness" and set(arguments) == {"name", "brightness_pct"}:
            value = arguments["brightness_pct"]
            if domain != "light" or isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
                raise PolicyError("denied")
            service, service_args = "turn_on", {"brightness_pct": value}
        elif name == "HassSetTemperature" and set(arguments) == {"name", "temperature"}:
            value = arguments["temperature"]
            if domain != "climate" or isinstance(value, bool) or not isinstance(value, (int, float)) or not -50 <= value <= 100:
                raise PolicyError("denied")
            service, service_args = "set_temperature", {"temperature": value}
        elif name in {"HassOpenCover", "HassCloseCover"} and set(arguments) == {"name"}:
            if domain != "cover":
                raise PolicyError("denied")
            service = "open_cover" if name == "HassOpenCover" else "close_cover"
        elif name in {"HassLock", "HassUnlock"} and set(arguments) == {"name"}:
            if domain != "lock" or not allow_high_impact:
                raise PolicyError("denied")
            service = "lock" if name == "HassLock" else "unlock"
        else:
            raise PolicyError("invalid_request")
        # No await before this final exposure/version check: capability revocation wins.
        if (
            registry_version != exposure.version
            or not exposure.contains(entity_id)
            or (hasattr(executor, "is_exposed") and not executor.is_exposed(entity_id))
        ):
            raise PolicyError("denied")
        result = await executor.async_call_service(domain, service, entity_id, service_args)
    if not isinstance(result, (dict, list)):
        raise PolicyError("invalid_request")
    return {"type": "tool.result", "call_id": call_id, "status": "ok", "result": result}
