"""The conversation entity: one Assist utterance in, one geist answer out."""

from __future__ import annotations

import logging
import socket

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_SOCKET, DEFAULT_SOCKET, TIMEOUT_S

_LOGGER = logging.getLogger(__name__)


def _ask_geist(path: str, text: str) -> str:
    """One utterance, one connection: send the line, half-close, read to EOF."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(TIMEOUT_S)
        s.connect(path)
        s.sendall(text.encode("utf-8").replace(b"\n", b" ") + b"\n")
        s.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            b = s.recv(4096)
            if not b:
                break
            chunks.append(b)
    return b"".join(chunks).decode("utf-8", errors="replace").strip()


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    async_add_entities([GeistConversationEntity(entry)])


class GeistConversationEntity(conversation.ConversationEntity):
    """Assist conversation agent backed by the resident geist daemon."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: ConfigEntry) -> None:
        self._sock_path = entry.data.get(CONF_SOCKET, DEFAULT_SOCKET)
        self._attr_unique_id = entry.entry_id

    @property
    def supported_languages(self) -> list[str] | str:
        return MATCH_ALL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        response = intent.IntentResponse(language=user_input.language)
        try:
            answer = await self.hass.async_add_executor_job(
                _ask_geist, self._sock_path, user_input.text
            )
        except OSError as err:
            _LOGGER.error("geist daemon unreachable at %s: %s", self._sock_path, err)
            response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "geist ist nicht erreichbar (läuft der --serve Daemon?)",
            )
            return conversation.ConversationResult(
                response=response, conversation_id=user_input.conversation_id
            )
        response.async_set_speech(answer or "(keine Antwort)")
        return conversation.ConversationResult(
            response=response, conversation_id=user_input.conversation_id
        )
