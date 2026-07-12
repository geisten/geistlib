#!/usr/bin/env python3
"""Static fail-closed contract for HA health entity, Repairs and diagnostics."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "integrations/home-assistant/custom_components/geist_conversation"

sensor = (COMPONENT / "sensor.py").read_text()
diagnostics = (COMPONENT / "diagnostics.py").read_text()
strings = (COMPONENT / "strings.json").read_text()
conversation = (COMPONENT / "conversation.py").read_text()

assert "async_create_issue" in sensor and "async_delete_issue" in sensor
assert "translation_key=\"runtime_health\"" in sensor
assert "IssueSeverity.ERROR" in sensor
assert "utterance" not in diagnostics and "entity" not in diagnostics
assert '"socket"' not in diagnostics
assert '"runtime_health"' in strings and "{error}" in strings
assert "RequestGate" in conversation and "status=busy" in conversation
assert "status=%s duration_ms=%d" in conversation
assert "geist daemon unreachable" not in conversation
print("ha_operability: health entity + Repairs + redacted diagnostics contract pass")
