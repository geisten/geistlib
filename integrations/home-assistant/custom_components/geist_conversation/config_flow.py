"""UI config and reconfigure flow with a model-free daemon health check."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigFlow, ConfigFlowResult

from .const import CONF_SOCKET, DEFAULT_SOCKET, DOMAIN, TIMEOUT_S
from .health import HealthError, async_validate_health


def _schema(default: str) -> vol.Schema:
    return vol.Schema({vol.Required(CONF_SOCKET, default=default): str})


class GeistConversationConfigFlow(ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Validate the local dynamic protocol before creating the entry."""
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured()
        errors: dict[str, str] = {}
        if user_input is not None:
            socket_path = user_input[CONF_SOCKET]
            try:
                await async_validate_health(socket_path, TIMEOUT_S)
            except HealthError as err:
                errors["base"] = err.code
            else:
                return self.async_create_entry(
                    title="geist", data={CONF_SOCKET: socket_path}
                )
        return self.async_show_form(
            step_id="user", data_schema=_schema(DEFAULT_SOCKET), errors=errors
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Validate and atomically reload a changed socket path."""
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}
        current = entry.data.get(CONF_SOCKET, DEFAULT_SOCKET)
        if user_input is not None:
            socket_path = user_input[CONF_SOCKET]
            try:
                await async_validate_health(socket_path, TIMEOUT_S)
            except HealthError as err:
                errors["base"] = err.code
            else:
                return self.async_update_reload_and_abort(
                    entry, data_updates={CONF_SOCKET: socket_path}
                )
        return self.async_show_form(
            step_id="reconfigure", data_schema=_schema(current), errors=errors
        )
