#!/usr/bin/env python3
"""Model-free contract checks for the Home Assistant integration artifact."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPONENT = (
    ROOT / "integrations/home-assistant/custom_components/geist_conversation"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    expected = {
        "__init__.py",
        "config_flow.py",
        "const.py",
        "conversation.py",
        "dynamic_session_v1.py",
        "dynamic_tools_v1.py",
        "ha_executor.py",
        "health.py",
        "manifest.json",
        "policy.py",
        "strings.json",
        "exposure.py",
    }
    require(expected <= {p.name for p in COMPONENT.iterdir()}, "component is complete")

    manifest = json.loads((COMPONENT / "manifest.json").read_text())
    require(manifest["domain"] == "geist_conversation", "manifest domain")
    require(manifest["iot_class"] == "local_push", "manifest transport class")
    require(manifest["requirements"] == [], "component has no Python dependencies")

    setup_source = (COMPONENT / "__init__.py").read_text()
    require("registry" not in setup_source, "component has no registry-sync compatibility path")
    require("GEIST_HA_TOKEN" not in setup_source, "component does not pass HA credentials")
    config_flow = (COMPONENT / "config_flow.py").read_text()
    require("async_validate_health" in config_flow, "config flow validates daemon health")
    require("async_step_reconfigure" in config_flow, "config flow supports reconfigure")
    require("async_step_import" not in config_flow, "no YAML-import compatibility flow")

    print("ha_integration: dynamic-only artifact contract pass")


if __name__ == "__main__":
    main()
