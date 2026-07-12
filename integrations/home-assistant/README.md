# Home Assistant developer preview

The preview uses a local Unix socket, not an HTTP inference server. A resident
`geist-home` process owns the model and bounded agent loop. The Home Assistant
component supplies a per-request dynamic toolset from the current Assist
exposure, executes validated calls inside HA, and returns their results to Geist.

## Prerequisites

- Home Assistant Core or Container on the same Linux host;
- a built `geist-home` binary from the home-agent branch;
- a model embedded in that binary, or the model configuration expected by the
  build;
- a host directory visible as `/config` inside the HA container.

Geist contains no Home Assistant REST client and receives no token. Expose only the entities Geist
may see or control; HA rechecks exposure and user context at execution time.

## Install the custom component

### Guided setup

With a released or locally built `geist-home`, the recommended path is:

```sh
scripts/setup-home-assistant.sh \
  --ha-config /path/to/ha-config \
  --binary /path/to/geist-home \
  --activate
```

### Low-level reproducible installer

The reproducible installer stages the component and a generated systemd unit:

```sh
scripts/install-home-assistant.sh \
  --ha-config /path/to/ha-config \
  --binary /absolute/path/to/geist-home \
  --user "$USER" \
  --work-dir /path/to/geist-workdir
```

Run the printed systemctl command, restart Home Assistant, then add the
integration. Re-running the installer performs an upgrade and records its
backup. Restore the most recent backup with the same arguments plus
`--rollback`. Use `--destdir` for packaging or a non-root staging test.

Manual installation remains possible:

From the repository root, with `/path/to/ha-config` replaced by the host path of
HA's `/config` volume:

```sh
mkdir -p /path/to/ha-config/custom_components/geist_conversation
cp integrations/home-assistant/custom_components/geist_conversation/* \
  /path/to/ha-config/custom_components/geist_conversation/
```

Restart Home Assistant, add the **geist Conversation** integration in the UI,
enter `/config/geist.sock`, and select it as the conversation agent of an Assist
pipeline. The Config Flow verifies the dynamic-tools-v1 health handshake before
creating the entry. For every request the
component derives tools only from Assist-exposed entities in supported domains.

## Run the resident daemon

The dynamic-tools path uses:

```sh
./geist-home --serve /path/to/ha-config/geist.sock
```

The daemon creates the socket mode `0600`. When HA runs in a container, the
socket's host path must be inside a mounted volume and its UID/GID permissions
must permit the container's HA process to connect.

For boot operation, adapt `../systemd/geist-home.service`: binary, working
directory, user, and socket path are deliberately explicit.
Then run:

```sh
sudo cp integrations/systemd/geist-home.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now geist-home
systemctl status geist-home
```

## Acceptance checks

```sh
# Daemon is resident and the socket is private.
systemctl is-active geist-home
stat -c '%a %U %G %n' /path/to/ha-config/geist.sock

# Component files and the dynamic protocol are valid without installing HA.
python3 tests/test_ha_integration.py
```

In Home Assistant, expose one harmless test light, run a status query and a
toggle through Assist, then unexpose it and verify the next request makes
the same request unavailable. Do not begin with locks, covers, or climate
setpoints.

For a deployed host, run the read-only health diagnostic. It checks systemd,
restart count, resident memory, and socket mode without sending an agent request:

```sh
scripts/check-home-assistant.sh --socket /path/to/ha-config/geist.sock
```

## Current protocol

- one newline-delimited JSON request containing `input`, `max_tool_steps`, and
  the tools currently permitted for this Assist request;
- correlated `tool.call` / `tool.result` round trips on the same socket;
- Home Assistant validates name, arguments, exposure and action policy, then
  executes; the dynamic protocol does not consume HA credentials;
- a final `conversation.result` contains the spoken answer;
- non-JSON requests and obsolete control frames are rejected.

This protocol is intentionally local and small. Soak evidence, external beta
feedback, and a full clean-host timing run remain before calling the integration
generally installable.
