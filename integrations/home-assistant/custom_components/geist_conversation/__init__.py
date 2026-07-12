"""geist Conversation — route Home Assistant Assist utterances to a resident
geist agent over its Unix socket (see `geist agent --serve` / DEPLOY.md).

Home Assistant owns exposure, policy, and action execution. Geist receives an
immutable request-scoped toolset over the local Unix socket and returns typed
tool calls; it never receives Home Assistant credentials.

Start the daemon on the host, e.g.:
    ./geist-home --serve ~/ha-config/geist.sock

Then pick "geist" as the conversation agent of your Assist pipeline
(Settings -> Voice assistants), or POST to /api/conversation/process with
its agent id.
"""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ConfigType

PLATFORMS = [Platform.CONVERSATION, Platform.SENSOR]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the integration; configuration is UI-only."""
    del hass, config
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
