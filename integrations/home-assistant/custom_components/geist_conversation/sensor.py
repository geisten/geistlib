"""Diagnostic health entity and Repairs lifecycle."""

from __future__ import annotations

from datetime import timedelta

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_SOCKET, DEFAULT_SOCKET, DOMAIN, HEALTH_TIMEOUT_S
from .health import HealthError, async_validate_health

SCAN_INTERVAL = timedelta(seconds=30)
ISSUE_PREFIX = "runtime_health_"


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    async_add_entities([GeistHealthSensor(entry)], update_before_add=True)


class GeistHealthSensor(SensorEntity):
    """Expose protocol readiness without leaking connection details."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_has_entity_name = True
    _attr_translation_key = "health"
    _attr_should_poll = True

    def __init__(self, entry: ConfigEntry) -> None:
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_health"
        self._attr_native_value = "unknown"
        self._attr_extra_state_attributes = {"protocol": "dynamic-tools-v1"}
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="Geist",
            manufacturer="Geist",
            model="Local inference service",
        )

    @property
    def _issue_id(self) -> str:
        return f"{ISSUE_PREFIX}{self._entry.entry_id}"

    async def async_update(self) -> None:
        socket_path = self._entry.data.get(CONF_SOCKET, DEFAULT_SOCKET)
        try:
            result = await async_validate_health(socket_path, HEALTH_TIMEOUT_S)
        except HealthError as err:
            self._attr_native_value = err.code
            self._attr_extra_state_attributes = {"protocol": "dynamic-tools-v1"}
            ir.async_create_issue(
                self.hass,
                DOMAIN,
                self._issue_id,
                is_fixable=False,
                severity=ir.IssueSeverity.ERROR,
                translation_key="runtime_health",
                translation_placeholders={"error": err.code},
            )
            return

        self._attr_native_value = result.status
        self._attr_extra_state_attributes = {"protocol": result.protocol}
        ir.async_delete_issue(self.hass, DOMAIN, self._issue_id)
