"""geist Conversation — route Home Assistant Assist utterances to a resident
geist agent over its Unix socket (see `geist agent --serve` / DEPLOY.md).

Home Assistant owns exposure, policy, and action execution. Geist receives an
immutable request-scoped toolset over the local Unix socket and returns typed
tool calls; it never receives Home Assistant credentials.

configuration.yaml (imported into a config entry on first start):

    geist_conversation:
      socket: /config/geist.sock   # the daemon's --serve path as the HA
                                   # container sees it (config volume)

Start the daemon on the host, e.g.:
    ./geist-home --serve ~/ha-config/geist.sock

Then pick "geist" as the conversation agent of your Assist pipeline
(Settings -> Voice assistants), or POST to /api/conversation/process with
its agent id.
"""

from __future__ import annotations

import logging

import voluptuous as vol

from homeassistant.config_entries import SOURCE_IMPORT, ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType

from .const import CONF_SOCKET, DEFAULT_SOCKET, DOMAIN

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.CONVERSATION]

CONFIG_SCHEMA = vol.Schema(
    {
        DOMAIN: vol.Schema(
            {vol.Optional(CONF_SOCKET, default=DEFAULT_SOCKET): cv.string}
        )
    },
    extra=vol.ALLOW_EXTRA,
)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Import the YAML block into a config entry (modern conversation agents
    are entities owned by an entry — the legacy default-agent override is
    gone)."""
    if DOMAIN in config and not hass.config_entries.async_entries(DOMAIN):
        hass.async_create_task(
            hass.config_entries.flow.async_init(
                DOMAIN, context={"source": SOURCE_IMPORT}, data=config[DOMAIN]
            )
        )
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
