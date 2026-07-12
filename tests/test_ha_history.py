#!/usr/bin/env python3
"""Model-free language and bounded in-memory history contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "integrations/home-assistant/custom_components/geist_conversation/history.py"
SPEC = importlib.util.spec_from_file_location("geist_history_contract", PATH)
assert SPEC is not None and SPEC.loader is not None
history = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(history)

assert history.normalize_language("de_DE") == "de-DE"
assert history.normalize_language(None, "fr") == "fr"
assert history.normalize_language("../../bad", "en") == "en"

store = history.HistoryStore()
assert store.context(None) == "" and store.context("new") == ""
for number in range(6):
    store.add("a", f"u{number}", f"a{number}")
messages = json.loads(store.context("a"))
assert len(messages) == history.MAX_TURNS * 2
assert messages[0]["content"] == "u2" and messages[-1]["content"] == "a5"
assert len(store.context("a").encode()) <= history.MAX_CONTEXT_BYTES

store.add("b", "other", "answer")
assert "other" not in store.context("a") and "u5" not in store.context("b")
store.reset("a")
assert store.context("a") == ""

for number in range(history.MAX_CONVERSATIONS + 1):
    store.add(f"c{number}", "u", "a")
assert store.context("c0") == ""
assert store.context(f"c{history.MAX_CONVERSATIONS}") != ""

store.add("huge", "x" * 3000, "y" * 3000)
assert store.context("huge") == ""
print("ha_history: language + isolation + turn/byte/LRU/reset bounds pass")
