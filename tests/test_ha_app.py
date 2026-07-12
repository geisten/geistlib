#!/usr/bin/env python3
"""Model-free security and packaging contract for the HA app scaffold."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "home-assistant-apps/geist"

repository = yaml.safe_load((ROOT / "repository.yaml").read_text())
config = yaml.safe_load((APP / "config.yaml").read_text())
dockerfile = (APP / "Dockerfile").read_text()
apparmor = (APP / "apparmor.txt").read_text()
run = (APP / "rootfs/run.sh").read_text()
health = (APP / "rootfs/healthcheck.sh").read_text()
workflow = (ROOT / ".github/workflows/ha-app.yml").read_text()

assert repository["name"] and repository["url"].startswith("https://")
assert config["slug"] == "geist"
assert config["arch"] == ["aarch64", "amd64"]
assert config["apparmor"] is True and config["init"] is False
assert config["map"] == [] and config["ports"] == {}
for key in ("host_network", "host_ipc", "host_dbus", "host_pid", "host_uts",
            "hassio_api", "homeassistant_api", "docker_api", "full_access",
            "audio", "video", "gpio", "usb", "uart", "udev", "stdin", "legacy"):
    assert config[key] is False, key
assert "BUILD_FROM" not in dockerfile
assert "FROM ghcr.io/home-assistant/base:" in dockerfile
assert "HEALTHCHECK" in dockerfile and "EXPOSE" not in dockerfile
assert 'io.hass.arch="${BUILD_ARCH}"' in dockerfile
assert "/data/geist-home" in run and "--serve" in run
assert "GEIST_HA_" not in run and "http" not in run.lower()
assert 'UNIX-CONNECT:/data/geist.sock' in health
assert '"type":"health"' in health and '"protocol":"dynamic-tools-v1"' in health
assert "/data/** rwk" in apparmor
assert "deny /config/**" in apparmor and "deny /run/docker.sock" in apparmor
assert "linux/arm64" in workflow and "linux/amd64" in workflow
assert "push: false" in workflow and "docker/build-push-action@v6" in workflow
print("ha_app: multi-arch protected scaffold + private data/health boundary pass")
