# Home Assistant developer preview

The preview uses a local Unix socket, not an HTTP inference server. A resident
`geist-home` process owns the model and bounded home-agent loop; the Home
Assistant custom component supplies the Assist UI/voice boundary and keeps
geist's deterministic device registry synchronized with HA's exposed entities.

## Prerequisites

- Home Assistant Core or Container on the same Linux host;
- a built `geist-home` binary from the home-agent branch;
- a model embedded in that binary, or the model configuration expected by the
  build;
- a Home Assistant long-lived access token for the preview's bounded REST
  tools;
- a host directory visible as `/config` inside the HA container.

The token is a preview limitation. Store it in a mode-0600 environment file and
expose only the entities Geist is allowed to resolve. A future integration may
execute actions inside HA and remove the daemon's token entirely.

## Install the custom component

The reproducible installer stages the component, a mode-0600 environment file,
and a generated systemd unit. It never accepts the HA token on the command line:

```sh
scripts/install-home-assistant.sh \
  --ha-config /path/to/ha-config \
  --binary /absolute/path/to/geist-home \
  --env-file /path/to/geist-home.env \
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

Add the transport to `configuration.yaml`:

```yaml
geist_conversation:
  socket: /config/geist.sock
```

Restart Home Assistant, add the **geist Conversation** integration, and select
it as the conversation agent of an Assist pipeline. The component pushes only
Assist-exposed entities in its supported domains; changes are debounced and the
full registry is re-pushed every five minutes after daemon restarts.

## Run the resident daemon

The deployed Pi preview uses:

```sh
GEIST_HA_URL=http://localhost:8123 \
GEIST_HA_TOKEN='<long-lived-token>' \
./geist-home --serve /path/to/ha-config/geist.sock
```

The daemon creates the socket mode `0600`. When HA runs in a container, the
socket's host path must be inside a mounted volume and its UID/GID permissions
must permit the container's HA process to connect.

For boot operation, adapt `../systemd/geist-home.service`: binary, working
directory, environment file, user, and socket path are deliberately explicit.
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

# Component files and pure registry behavior are valid without installing HA.
python3 tests/test_ha_integration.py
```

In Home Assistant, expose one harmless test light, run a status query and a
toggle through Assist, then unexpose it and verify the next registry sync makes
the same request unavailable. Do not begin with locks, covers, or climate
setpoints.

For a deployed host, run the read-only health diagnostic. It checks systemd,
restart count, resident memory, socket mode, recent registry synchronization,
environment permissions and the authenticated HA API without sending an agent
request:

```sh
scripts/check-home-assistant.sh \
  --socket /path/to/ha-config/geist.sock \
  --env-file /path/to/geist-home.env
```

## Current protocol

- one UTF-8 utterance line per Unix-socket connection;
- client half-closes its write side;
- daemon returns an EOF-framed UTF-8 answer;
- control frame `\x01REGISTRY\n<body>` replaces the in-memory exposed-device
  registry when the body contains at least one valid entry.

This protocol is intentionally local and small. Versioned framing, diagnostics,
upgrade/rollback automation, and clean-host timing are the remaining Phase-1
work before calling the integration generally installable.
