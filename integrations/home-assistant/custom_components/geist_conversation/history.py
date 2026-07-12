"""Bounded in-memory conversation context for the HA adapter."""

from __future__ import annotations

from collections import OrderedDict
import json
import re

MAX_CONVERSATIONS = 32
MAX_TURNS = 4
MAX_CONTEXT_BYTES = 2048
_LANGUAGE = re.compile(r"^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$")


def normalize_language(value: str | None, fallback: str = "en") -> str:
    """Return a bounded BCP-47-like code accepted by the runtime contract."""
    candidate = (value or fallback).replace("_", "-")[:15]
    return candidate if _LANGUAGE.fullmatch(candidate) else fallback


class HistoryStore:
    """LRU conversations with fixed conversation, turn, and byte limits."""

    def __init__(self) -> None:
        self._items: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()

    def context(self, conversation_id: str | None) -> str:
        if not conversation_id or conversation_id not in self._items:
            return ""
        turns = self._items[conversation_id]
        self._items.move_to_end(conversation_id)
        return self._encode(turns)

    def add(self, conversation_id: str | None, user: str, assistant: str) -> None:
        if not conversation_id:
            return
        turns = self._items.setdefault(conversation_id, [])
        self._items.move_to_end(conversation_id)
        turns.append((user, assistant))
        del turns[:-MAX_TURNS]
        while turns and len(self._encode(turns).encode("utf-8")) > MAX_CONTEXT_BYTES:
            del turns[0]
        if not turns:
            self._items.pop(conversation_id, None)
        while len(self._items) > MAX_CONVERSATIONS:
            self._items.popitem(last=False)

    def reset(self, conversation_id: str | None) -> None:
        if conversation_id:
            self._items.pop(conversation_id, None)

    @staticmethod
    def _encode(turns: list[tuple[str, str]]) -> str:
        messages = [
            message
            for user, assistant in turns
            for message in (
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            )
        ]
        return json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
