"""Redacted config-entry diagnostics."""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import CONF_SOCKET, DEFAULT_SOCKET, HEALTH_TIMEOUT_S
from .health import HealthError, async_validate_health


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Return protocol health only; never reveal paths, addresses or HA state."""
    del hass
    try:
        result = await async_validate_health(
            entry.data.get(CONF_SOCKET, DEFAULT_SOCKET), HEALTH_TIMEOUT_S
        )
    except HealthError as err:
        health = {"status": "error", "error": err.code}
    else:
        health = {"status": result.status, "protocol": result.protocol}
    return {"configured": True, "health": health}
