"""The conversation entity: one Assist utterance in, one geist answer out."""

from __future__ import annotations

import logging

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_SOCKET, DEFAULT_SOCKET, TIMEOUT_S
from .dynamic_session_v1 import async_ask_geist_dynamic
from .ha_executor import HomeAssistantExecutor
from .policy import ExposureStore
from .protocol_v2 import ProtocolError
from .registry import exposed_entity_ids

_LOGGER = logging.getLogger(__name__)


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
            exposure = ExposureStore(frozenset(exposed_entity_ids(self.hass)), 1)
            answer = await async_ask_geist_dynamic(
                self._sock_path,
                user_input.text,
                exposure,
                HomeAssistantExecutor(self.hass, getattr(user_input, "context", None)),
                timeout_s=TIMEOUT_S,
            )
        except (OSError, ProtocolError) as err:
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
