"""Execute already-validated protocol-v2 calls inside Home Assistant.

The module is intentionally not wired into the Phase-1 conversation path yet.
It uses duck-typed ``hass``/``context`` objects so its security contract stays
model-free and testable without importing Home Assistant.
"""

from __future__ import annotations

from typing import Any

from .policy import ExposureStore, PolicyError, ValidatedToolCall, validate_tool_call

SAFE_STATE_ATTRIBUTES = {
    "brightness",
    "current_temperature",
    "temperature",
    "unit_of_measurement",
}
JSON_SCALARS = (str, int, float, bool, type(None))


class HomeAssistantExecutor:
    def __init__(self, hass: Any, context: Any = None) -> None:
        self._hass = hass
        self._context = context

    def get_state(self, entity_id: str) -> dict[str, Any]:
        state = self._hass.states.get(entity_id)
        if state is None:
            raise PolicyError("unavailable")
        attributes = {
            key: value
            for key, value in state.attributes.items()
            if key in SAFE_STATE_ATTRIBUTES and isinstance(value, JSON_SCALARS)
        }
        return {"state": str(state.state), "attributes": attributes}

    def is_exposed(self, entity_id: str) -> bool:
        """Recheck Assist exposure at the final action boundary."""
        try:
            from homeassistant.components.homeassistant.exposed_entities import (
                async_should_expose,
            )
        except ImportError:
            return self._hass.states.get(entity_id) is not None
        return bool(async_should_expose(self._hass, "conversation", entity_id))

    async def async_call_service(
        self,
        domain: str,
        service: str,
        entity_id: str,
        arguments: dict[str, Any],
    ) -> list[Any]:
        service_data = {"entity_id": entity_id, **arguments}
        try:
            await self._hass.services.async_call(
                domain,
                service,
                service_data,
                blocking=True,
                context=self._context,
            )
        except Exception as err:
            if err.__class__.__name__ == "Unauthorized":
                raise PolicyError("denied") from err
            raise PolicyError("unavailable") from err
        return []


async def async_execute_tool_call(
    call: ValidatedToolCall,
    exposure: ExposureStore,
    executor: HomeAssistantExecutor,
) -> dict[str, Any]:
    # No await is allowed between this final exposure check and scheduling the
    # HA service call. That closes the validation/execution race as far as the
    # event-loop action boundary permits.
    if call.registry_version != exposure.version or not exposure.contains(call.entity_id):
        raise PolicyError("denied")
    if call.operation == "get_state":
        result: Any = executor.get_state(call.entity_id)
    else:
        if call.service is None:
            raise PolicyError("invalid_request")
        result = await executor.async_call_service(
            call.domain,
            call.service,
            call.entity_id,
            dict(call.arguments),
        )
    if not isinstance(result, (dict, list)):
        raise PolicyError("invalid_request")
    return {
        "version": 2,
        "request_id": call.request_id,
        "type": "tool.result",
        "status": "ok",
        "result": result,
    }


async def async_handle_tool_call(
    frame: dict[str, Any],
    exposure: ExposureStore,
    executor: HomeAssistantExecutor,
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
        return await async_execute_tool_call(call, exposure, executor)
    except PolicyError as err:
        return {
            "version": 2,
            "request_id": request_id,
            "type": "tool.result",
            "status": err.status,
            "result": {},
        }
