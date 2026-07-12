"""Resolve the entities currently exposed to Home Assistant Assist."""

from __future__ import annotations


SUPPORTED_DOMAINS = {
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

ASSISTANT = "conversation"


def _exposed_entity_ids(hass) -> list[str]:
    from homeassistant.components.homeassistant.exposed_entities import (
        async_should_expose,
    )

    return [
        state.entity_id
        for state in hass.states.async_all()
        if async_should_expose(hass, ASSISTANT, state.entity_id)
    ]


def exposed_entity_ids(hass) -> set[str]:
    """Return the supported entity ids exposed for this Assist request."""
    return {
        entity_id
        for entity_id in _exposed_entity_ids(hass)
        if entity_id.split(".", 1)[0] in SUPPORTED_DOMAINS
    }
