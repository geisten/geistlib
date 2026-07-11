"""Config flow: single instance, created from the YAML import (no UI form —
the socket path lives in configuration.yaml next to the daemon that owns it)."""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigFlow, ConfigFlowResult

from .const import CONF_SOCKET, DEFAULT_SOCKET, DOMAIN


class GeistConversationConfigFlow(ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_import(self, import_data: dict[str, Any]) -> ConfigFlowResult:
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured()
        return self.async_create_entry(
            title="geist",
            data={CONF_SOCKET: import_data.get(CONF_SOCKET, DEFAULT_SOCKET)},
        )

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Adding via UI just uses the default socket path."""
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured()
        return self.async_create_entry(title="geist", data={CONF_SOCKET: DEFAULT_SOCKET})
