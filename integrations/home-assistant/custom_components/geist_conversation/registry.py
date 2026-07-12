"""Build a geist device registry from Home Assistant's exposed entities.

geist resolves devices DETERMINISTICALLY against a registry of
`entity_id | domain | aliases` lines (see agent_home.h). Instead of
hand-maintaining that file, derive it from HA itself: for every entity EXPOSED
to Assist, the aliases are

    friendly name  +  the entity's HA aliases  +  its AREA name  +  a generic
    domain word ("licht"/"light", "rollladen", ...)

so a request naming a room resolves (the area name is an alias), the room-status
listing uses real HA areas, and a bare "das Licht" stays deliberately ambiguous
across the lights (the generic word) — matching the hand-registry semantics.

Task #2 builds the registry text; task #3 pushes it to the daemon over the
REGISTRY control frame and re-pushes on change.
"""

from __future__ import annotations

import logging
import socket
from datetime import timedelta

_LOGGER = logging.getLogger(__name__)

# First bytes of a REGISTRY control frame (see agent_main.h / HOME.md §4).
CTL_REGISTRY = b"\x01REGISTRY\n"
# Safety-net re-push (a daemon restart drops the in-memory registry; an
# expose-toggle fires no registry event) — bounded staleness without polling HA.
REPUSH_INTERVAL = timedelta(minutes=5)
# Coalesce a burst of registry changes into one push.
DEBOUNCE_S = 2.0

# The assistant key HA's expose settings use for Assist / conversation agents.
ASSISTANT = "conversation"

# Domains geist can command; sensor/binary_sensor are read-only status. Anything
# exposed outside this set is skipped — geist has no tool for it.
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

# A generic noun per domain so a bare "das Licht" matches every light (the
# deliberate-ambiguity answer), while "Licht im Wohnzimmer" disambiguates via the
# area alias. Conservative on purpose — one or two common German/English nouns.
DOMAIN_WORDS = {
    "light": "licht, light, lampe",
    "switch": "schalter, switch",
    "climate": "heizung, thermostat, klima",
    "cover": "rollladen, jalousie, cover",
    "media_player": "musik, radio, lautsprecher",
    "lock": "schloss, lock",
    "fan": "ventilator, lüfter, fan",
}


def format_line(
    entity_id: str,
    friendly: str | None,
    ha_aliases: list[str] | None,
    area: str | None,
) -> str | None:
    """Assemble one `entity | domain | aliases` registry line, or None if the
    entity yields no usable alias. PURE — no hass, so it is unit-testable.

    Aliases are de-duplicated case-insensitively and stripped of the '|' field
    delimiter; the generic domain word is appended so bare-noun requests stay
    ambiguous rather than resolving to the alphabetically-first device.
    """
    domain = entity_id.split(".", 1)[0]
    if domain not in SUPPORTED_DOMAINS:
        return None
    parts: list[str] = []
    if friendly:
        parts.append(friendly)
    if ha_aliases:
        parts.extend(ha_aliases)
    if area:
        parts.append(area)
    if domain in DOMAIN_WORDS:
        parts.append(DOMAIN_WORDS[domain])

    seen: list[str] = []
    lowered: set[str] = set()
    for phrase in parts:
        if not isinstance(phrase, str):
            continue  # HA can hand back a ComputedNameType sentinel, not a str
        phrase = " ".join(phrase.replace("|", " ").split())  # strip '|' + collapse spaces
        if phrase and phrase.lower() not in lowered:
            seen.append(phrase)
            lowered.add(phrase.lower())
    if not seen:
        return None
    return f"{entity_id} | {domain} | {', '.join(seen)}"


def _exposed_entity_ids(hass) -> list[str]:
    """Entity ids exposed to Assist. Falls back to all states on older cores
    without the exposed-entities helper."""
    try:
        from homeassistant.components.homeassistant.exposed_entities import (
            async_should_expose,
        )
    except ImportError:
        return [s.entity_id for s in hass.states.async_all()]
    return [
        s.entity_id
        for s in hass.states.async_all()
        if async_should_expose(hass, ASSISTANT, s.entity_id)
    ]


def exposed_entity_ids(hass) -> set[str]:
    """Current Assist exposure filtered to domains supported by the adapter."""
    return {
        entity_id
        for entity_id in _exposed_entity_ids(hass)
        if entity_id.split(".", 1)[0] in SUPPORTED_DOMAINS
    }


def _area_name(hass, ent_reg, area_reg, dev_reg, entity_id: str) -> str | None:
    """The entity's area name: its own area, else its device's area, else None."""
    entry = ent_reg.async_get(entity_id)
    if entry is None:
        return None
    area_id = entry.area_id
    if area_id is None and entry.device_id:
        dev = dev_reg.async_get(entry.device_id)
        area_id = dev.area_id if dev else None
    if area_id is None:
        return None
    area = area_reg.async_get_area(area_id)
    return area.name if area else None


def build_registry(hass) -> str:
    """Return the geist registry as newline-separated `entity | domain | aliases`
    lines — the body of a REGISTRY control frame. Call from the event loop
    (registry access is loop-bound, not blocking I/O)."""
    from homeassistant.helpers import (
        area_registry as ar,
        device_registry as dr,
        entity_registry as er,
    )

    ent_reg = er.async_get(hass)
    area_reg = ar.async_get(hass)
    dev_reg = dr.async_get(hass)

    lines: list[str] = []
    for entity_id in _exposed_entity_ids(hass):
        try:
            entry = ent_reg.async_get(entity_id)
            state = hass.states.get(entity_id)
            friendly = state.attributes.get("friendly_name") if state else None
            ha_aliases = sorted(entry.aliases) if entry and entry.aliases else None
            area = _area_name(hass, ent_reg, area_reg, dev_reg, entity_id)
            line = format_line(entity_id, friendly, ha_aliases, area)
            if line:
                lines.append(line)
        except Exception:  # one odd entity must not blank the whole registry
            _LOGGER.debug("geist registry: skipping %s", entity_id, exc_info=True)
    return "\n".join(lines)


def _push_sync(sock_path: str, body: str, timeout: float) -> str:
    """Blocking: send a REGISTRY control frame, return the daemon's reply
    ("ok: <n> devices"). Runs in an executor — it is socket I/O."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(sock_path)
        s.sendall(CTL_REGISTRY + body.encode("utf-8"))
        s.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            b = s.recv(4096)
            if not b:
                break
            chunks.append(b)
    return b"".join(chunks).decode("utf-8", errors="replace").strip()


async def async_push_registry(hass, sock_path: str, timeout: float) -> str:
    """Build the registry (in the loop) and push it to the daemon (in an
    executor). Returns the daemon's reply."""
    body = build_registry(hass)  # registry access is loop-bound, not I/O
    return await hass.async_add_executor_job(_push_sync, sock_path, body, timeout)


def async_setup_registry_sync(hass, sock_path: str, timeout: float):
    """Push the registry now, on every entity/area/device change (debounced),
    and periodically (safety net after a daemon restart). Returns an unsub
    callable for entry.async_on_unload."""
    from homeassistant.core import callback
    from homeassistant.helpers import (
        area_registry as ar,
        device_registry as dr,
        entity_registry as er,
    )
    from homeassistant.helpers.event import async_call_later, async_track_time_interval

    state: dict = {"cancel": None}

    async def _push(_now=None) -> None:
        state["cancel"] = None
        try:
            reply = await async_push_registry(hass, sock_path, timeout)
            _LOGGER.info("geist registry synced: %s", reply)
        except OSError as err:
            _LOGGER.warning("geist registry push failed (is the --serve daemon up?): %s", err)

    @callback
    def _schedule(_event=None) -> None:
        if state["cancel"] is not None:
            state["cancel"]()  # coalesce a burst into one push
        state["cancel"] = async_call_later(hass, DEBOUNCE_S, _push)

    unsubs = [
        hass.bus.async_listen(er.EVENT_ENTITY_REGISTRY_UPDATED, _schedule),
        hass.bus.async_listen(ar.EVENT_AREA_REGISTRY_UPDATED, _schedule),
        hass.bus.async_listen(dr.EVENT_DEVICE_REGISTRY_UPDATED, _schedule),
        async_track_time_interval(hass, _push, REPUSH_INTERVAL),
    ]
    hass.async_create_task(_push())  # initial push

    @callback
    def _unsub() -> None:
        for u in unsubs:
            u()
        if state["cancel"] is not None:
            state["cancel"]()

    return _unsub


if __name__ == "__main__":
    # self-check for the pure line assembler (no hass needed)
    assert format_line("switch.tv", None, None, None) == (
        "switch.tv | switch | schalter, switch"
    ), "generic domain word only"
    line = format_line("light.x", "Wohnzimmer Deckenlicht", ["Deckenlicht"], "Wohnzimmer")
    assert line == (
        "light.x | light | Wohnzimmer Deckenlicht, Deckenlicht, Wohnzimmer, licht, light, lampe"
    ), f"friendly + alias + area + generic: {line}"
    assert format_line("camera.door", "Door", None, None) is None, "unsupported domain skipped"
    assert (
        format_line("sensor.temp", "Temp | Bad", None, "Bad")
        == "sensor.temp | sensor | Temp Bad, Bad"
    ), "pipe stripped + spaces collapsed; no generic word for sensor"
    print("registry.format_line: ok")
