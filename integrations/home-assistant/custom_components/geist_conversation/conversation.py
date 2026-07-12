"""The conversation entity: one Assist utterance in, one geist answer out."""

from __future__ import annotations

import asyncio
import logging
from time import monotonic
from uuid import uuid4

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_SOCKET, DEFAULT_SOCKET, TIMEOUT_S
from .dynamic_session_v1 import ProtocolError, RequestGate, async_ask_geist_dynamic
from .ha_executor import HomeAssistantExecutor
from .history import HistoryStore, normalize_language
from .policy import ExposureStore
from .exposure import exposed_entity_ids

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
        self._gate = RequestGate()
        self._history = HistoryStore()

    @property
    def supported_languages(self) -> list[str] | str:
        return MATCH_ALL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        response = intent.IntentResponse(language=user_input.language)
        conversation_id = user_input.conversation_id or uuid4().hex
        language = normalize_language(
            user_input.language, getattr(self.hass.config, "language", "en")
        )
        context = self._history.context(user_input.conversation_id)
        started = monotonic()
        try:
            self._gate.enter()
        except ProtocolError:
            _LOGGER.info("geist_request status=busy duration_ms=0")
            response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "Geist is busy. Try again shortly.",
            )
            return conversation.ConversationResult(
                response=response, conversation_id=conversation_id
            )
        try:
            exposure = ExposureStore(frozenset(exposed_entity_ids(self.hass)), 1)
            answer = await async_ask_geist_dynamic(
                self._sock_path,
                user_input.text,
                exposure,
                HomeAssistantExecutor(self.hass, getattr(user_input, "context", None)),
                timeout_s=TIMEOUT_S,
                language=language,
                context=context,
            )
        except asyncio.CancelledError:
            _LOGGER.info(
                "geist_request status=client_cancelled duration_ms=%d",
                int((monotonic() - started) * 1000),
            )
            raise
        except OSError:
            code = "cannot_connect"
            _LOGGER.warning(
                "geist_request status=%s duration_ms=%d",
                code,
                int((monotonic() - started) * 1000),
            )
            response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Geist request failed ({code}).",
            )
            return conversation.ConversationResult(
                response=response, conversation_id=conversation_id
            )
        except ProtocolError as err:
            _LOGGER.warning(
                "geist_request status=%s duration_ms=%d",
                err.code,
                int((monotonic() - started) * 1000),
            )
            response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Geist request failed ({err.code}).",
            )
            return conversation.ConversationResult(
                response=response, conversation_id=conversation_id
            )
        finally:
            self._gate.leave()
        _LOGGER.info(
            "geist_request status=ok duration_ms=%d",
            int((monotonic() - started) * 1000),
        )
        response.async_set_speech(answer or "(keine Antwort)")
        self._history.add(conversation_id, user_input.text, answer)
        return conversation.ConversationResult(
            response=response, conversation_id=conversation_id
        )
