#!/bin/sh
# Install the geist Home Assistant developer preview. Supports DESTDIR for
# hermetic clean-host tests and packaging.
set -eu

usage() {
    cat <<'EOF'
usage: scripts/install-home-assistant.sh [options]

Required:
  --ha-config DIR       host directory mounted as /config in Home Assistant
  --binary PATH         executable geist-home binary

Options:
  --user USER           systemd service user (default: current user)
  --work-dir DIR        daemon working directory (default: binary directory)
  --state-dir DIR       installed env/backups (default: ~/.local/share/geist-home)
  --unit-dir DIR        systemd unit directory (default: /etc/systemd/system)
  --destdir DIR         prefix every destination; for packaging/tests only
  --rollback            restore the most recent backup made by this installer
  --help

The installer does not restart Home Assistant or systemd. It prints the exact
activation commands after a successful real-host installation.
EOF
}

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
component_src="$repo_dir/integrations/home-assistant/custom_components/geist_conversation"
ha_config=
binary=
service_user=$(id -un)
work_dir=
state_dir=${HOME:-/tmp}/.local/share/geist-home
unit_dir=/etc/systemd/system
destdir=
rollback=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --ha-config) ha_config=${2-}; shift 2 ;;
        --binary) binary=${2-}; shift 2 ;;
        --user) service_user=${2-}; shift 2 ;;
        --work-dir) work_dir=${2-}; shift 2 ;;
        --state-dir) state_dir=${2-}; shift 2 ;;
        --unit-dir) unit_dir=${2-}; shift 2 ;;
        --destdir) destdir=${2-}; shift 2 ;;
        --rollback) rollback=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "install-home-assistant: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

required() {
    name=$1
    value=$2
    if [ -z "$value" ]; then
        echo "install-home-assistant: $name is required" >&2
        exit 2
    fi
}

required --ha-config "$ha_config"
required --binary "$binary"

case "$ha_config$binary$state_dir$unit_dir$destdir" in
    *"\n"*) echo "install-home-assistant: newlines are not allowed in paths" >&2; exit 2 ;;
esac

if [ ! -d "$component_src" ]; then
    echo "install-home-assistant: component source missing: $component_src" >&2
    exit 1
fi
if [ ! -x "$binary" ]; then
    echo "install-home-assistant: binary is not executable: $binary" >&2
    exit 1
fi
if ! "$binary" --help 2>&1 | grep -q -- '--serve'; then
    echo "install-home-assistant: binary does not advertise the required --serve mode" >&2
    exit 1
fi
if [ -z "$work_dir" ]; then
    work_dir=$(CDPATH= cd -- "$(dirname -- "$binary")" && pwd)
fi

prefix_path() {
    path=$1
    if [ -n "$destdir" ]; then
        case "$path" in
            /*) printf '%s%s\n' "$destdir" "$path" ;;
            *) printf '%s/%s\n' "$destdir" "$path" ;;
        esac
    else
        printf '%s\n' "$path"
    fi
}

component_dst=$(prefix_path "$ha_config/custom_components/geist_conversation")
state_dst=$(prefix_path "$state_dir")
unit_dst=$(prefix_path "$unit_dir/geist-home.service")
backup_root="$state_dst/backups"
latest_file="$state_dst/latest-backup"

if [ "$rollback" -eq 1 ]; then
    if [ ! -f "$latest_file" ]; then
        echo "install-home-assistant: no rollback backup found" >&2
        exit 1
    fi
    backup=$(sed -n '1p' "$latest_file")
    if [ ! -d "$backup" ]; then
        echo "install-home-assistant: rollback backup is missing: $backup" >&2
        exit 1
    fi
    rm -rf "$component_dst"
    if [ -d "$backup/component" ]; then cp -R "$backup/component" "$component_dst"; fi
    if [ -f "$backup/unit" ]; then cp "$backup/unit" "$unit_dst"; else rm -f "$unit_dst"; fi
    echo "Rolled back geist Home Assistant artifacts from $backup"
    exit 0
fi

mkdir -p "$component_dst" "$state_dst" "$(dirname -- "$unit_dst")" "$backup_root"
backup="$backup_root/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$backup"
if [ -d "$component_dst" ] && [ "$(find "$component_dst" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    cp -R "$component_dst" "$backup/component"
fi
if [ -f "$unit_dst" ]; then cp "$unit_dst" "$backup/unit"; fi

# Copy an explicit allowlist: never deploy __pycache__, editor files, or secrets.
rm -rf "$component_dst"
mkdir -p "$component_dst"
for file in __init__.py config_flow.py const.py conversation.py diagnostics.py \
    dynamic_session_v1.py dynamic_tools_v1.py exposure.py ha_executor.py health.py \
    manifest.json policy.py sensor.py strings.json; do
    cp "$component_src/$file" "$component_dst/$file"
done
cp -R "$component_src/translations" "$component_dst/translations"

socket_path="$ha_config/geist.sock"
cat >"$unit_dst" <<EOF
[Unit]
Description=geist home appliance daemon (Unix socket)
After=network.target

[Service]
User=$service_user
WorkingDirectory=$work_dir
Environment=GEIST_AGENT_TRACE=0
ExecStart=$binary --serve $socket_path
Restart=on-failure
RestartSec=5
NoNewPrivileges=yes

[Install]
WantedBy=multi-user.target
EOF

printf '%s\n' "$backup" >"$latest_file"
cat >"$state_dst/install-manifest.txt" <<EOF
component=$ha_config/custom_components/geist_conversation
unit=$unit_dir/geist-home.service
socket=$socket_path
EOF

echo "Installed geist Home Assistant component: $component_dst"
echo "Installed systemd unit: $unit_dst"
echo "Rollback backup: $backup"
if [ -z "$destdir" ]; then
    echo "Next: sudo systemctl daemon-reload && sudo systemctl enable --now geist-home"
    echo "Then restart Home Assistant and add the geist Conversation integration."
fi
