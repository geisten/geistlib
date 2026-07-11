#!/bin/sh
# Guided one-command setup for the local Home Assistant appliance.
set -eu

usage() {
    cat <<'EOF'
usage: scripts/setup-home-assistant.sh [options]

Options:
  --ha-config DIR   host path mounted as HA /config (prompted if omitted)
  --binary PATH     geist-home binary (auto-detected if omitted)
  --ha-url URL      Home Assistant URL (default: http://localhost:8123)
  --token-file PATH file containing only the HA token
  --user USER       service user (default: current user)
  --activate        run sudo systemctl daemon-reload/enable and restart HA
  --dry-run         detect and validate without writing
  --help

Missing interactive values are prompted. The token is never placed in argv,
generated units, logs, or the repository.
EOF
}

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ha_config=
binary=
ha_url=http://localhost:8123
token_file=
service_user=$(id -un)
activate=0
dry_run=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --ha-config) ha_config=${2-}; shift 2 ;;
        --binary) binary=${2-}; shift 2 ;;
        --ha-url) ha_url=${2-}; shift 2 ;;
        --token-file) token_file=${2-}; shift 2 ;;
        --user) service_user=${2-}; shift 2 ;;
        --activate) activate=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "setup-home-assistant: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

platform=$(uname -s)-$(uname -m)
case "$platform" in
    Linux-aarch64|Linux-arm64) release_platform=linux-arm64 ;;
    Linux-x86_64|Linux-amd64) release_platform=linux-x86_64 ;;
    Darwin-arm64) release_platform=macos-arm64 ;;
    *) release_platform=unsupported ;;
esac

if [ -z "$binary" ]; then
    for candidate in \
        "$root/geist-home" \
        "$root/bin/$("$root/mk/detect-target.sh")/release/tools/geist-home" \
        "$HOME/.local/bin/geist-home" \
        /usr/local/bin/geist-home; do
        if [ -x "$candidate" ]; then binary=$candidate; break; fi
    done
fi
if [ -z "$binary" ] || [ ! -x "$binary" ]; then
    echo "setup-home-assistant: geist-home not found for $release_platform" >&2
    echo "Build it with 'make home' or pass --binary /absolute/path/geist-home." >&2
    exit 1
fi
binary=$(CDPATH= cd -- "$(dirname -- "$binary")" && pwd)/$(basename -- "$binary")
if ! "$binary" --help 2>&1 | grep -q -- '--serve'; then
    echo "setup-home-assistant: binary lacks the required --serve mode: $binary" >&2
    exit 1
fi

if [ -z "$ha_config" ]; then
    if [ ! -t 0 ]; then
        echo "setup-home-assistant: --ha-config is required non-interactively" >&2
        exit 2
    fi
    printf 'Home Assistant host config directory: '
    IFS= read -r ha_config
fi
if [ ! -d "$ha_config" ]; then
    echo "setup-home-assistant: HA config directory does not exist: $ha_config" >&2
    exit 1
fi
ha_config=$(CDPATH= cd -- "$ha_config" && pwd)

printf 'platform=%s release=%s\n' "$platform" "$release_platform"
printf 'binary=%s\nha_config=%s\nha_url=%s\n' "$binary" "$ha_config" "$ha_url"
if [ "$dry_run" -eq 1 ]; then
    echo "setup-home-assistant: dry-run PASS"
    exit 0
fi

state_dir=${HOME:-/tmp}/.local/share/geist-home
mkdir -p "$state_dir"
env_file=$state_dir/geist-home.setup.env
umask 077
if [ -n "$token_file" ]; then
    if [ ! -f "$token_file" ]; then
        echo "setup-home-assistant: token file missing: $token_file" >&2
        exit 1
    fi
    token=$(sed -n '1p' "$token_file")
elif [ -t 0 ]; then
    printf 'Home Assistant long-lived token: '
    stty -echo
    IFS= read -r token
    stty echo
    printf '\n'
else
    echo "setup-home-assistant: --token-file is required non-interactively" >&2
    exit 2
fi
if [ -z "$token" ]; then
    echo "setup-home-assistant: token is empty" >&2
    exit 1
fi
printf 'GEIST_HA_URL=%s\nGEIST_HA_TOKEN=%s\n' "$ha_url" "$token" >"$env_file"
chmod 600 "$env_file"

if command -v curl >/dev/null 2>&1; then
    if ! printf 'header = "Authorization: Bearer %s"\n' "$token" |
        curl -fsS --max-time 10 -o /dev/null --config - "$ha_url/api/"; then
        token=
        echo "setup-home-assistant: authenticated HA API check failed: $ha_url" >&2
        exit 1
    fi
fi
token=

"$root/scripts/install-home-assistant.sh" \
    --ha-config "$ha_config" \
    --binary "$binary" \
    --env-file "$env_file" \
    --user "$service_user" \
    --work-dir "$(dirname -- "$binary")" \
    --state-dir "$state_dir"
rm -f "$env_file"

if [ "$activate" -eq 1 ]; then
    command -v systemctl >/dev/null 2>&1 || {
        echo "setup-home-assistant: --activate requires systemd" >&2
        exit 1
    }
    sudo systemctl daemon-reload
    sudo systemctl enable --now geist-home
    if command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' | grep -qx homeassistant; then
        docker restart homeassistant >/dev/null
    else
        echo "Restart Home Assistant, then add the geist Conversation integration."
    fi
fi

echo "setup-home-assistant: PASS"
