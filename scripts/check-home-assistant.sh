#!/bin/sh
# Read-only health diagnostics for a deployed geist Home Assistant appliance.
set -eu

usage() {
    cat <<'EOF'
usage: scripts/check-home-assistant.sh [options]

Options:
  --service NAME          systemd unit (default: geist-home)
  --socket PATH           Unix socket (default: /config/geist.sock)
  --env-file PATH         GEIST_HA_URL/TOKEN file; enables HA API check
  --registry-age SEC      require a registry push this recently (default: 600)
  --max-restarts N        allowed systemd restart count (default: 3)
  --skip-systemd          check only filesystem and optional HA API
  --help

No state is changed and no agent request is sent through the socket.
EOF
}

service=geist-home
socket_path=/config/geist.sock
env_file=
registry_age=600
max_restarts=3
skip_systemd=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --service) service=${2-}; shift 2 ;;
        --socket) socket_path=${2-}; shift 2 ;;
        --env-file) env_file=${2-}; shift 2 ;;
        --registry-age) registry_age=${2-}; shift 2 ;;
        --max-restarts) max_restarts=${2-}; shift 2 ;;
        --skip-systemd) skip_systemd=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "check-home-assistant: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$registry_age:$max_restarts" in
    *[!0-9:]*|:*|*:) echo "check-home-assistant: ages/counts must be non-negative integers" >&2; exit 2 ;;
esac

failures=0
ok() { printf 'OK   %s\n' "$1"; }
fail() { printf 'FAIL %s\n' "$1" >&2; failures=$((failures + 1)); }
file_mode() {
    stat -c '%a' "$1" 2>/dev/null || stat -f '%Lp' "$1"
}

if [ -S "$socket_path" ]; then
    ok "Unix socket exists: $socket_path"
else
    fail "Unix socket missing: $socket_path"
fi

if [ -e "$socket_path" ]; then
    mode=$(file_mode "$socket_path")
    if [ "$mode" = 600 ]; then
        ok "socket permissions are 0600"
    else
        fail "socket permissions are $mode, expected 600"
    fi
fi

if [ "$skip_systemd" -eq 0 ]; then
    if systemctl is-active --quiet "$service"; then
        ok "systemd service is active: $service"
    else
        fail "systemd service is not active: $service"
    fi

    main_pid=$(systemctl show "$service" -p MainPID --value 2>/dev/null || printf 0)
    restarts=$(systemctl show "$service" -p NRestarts --value 2>/dev/null || printf 999999)
    memory=$(systemctl show "$service" -p MemoryCurrent --value 2>/dev/null || printf unknown)
    case "$main_pid" in
        ''|0|*[!0-9]*) fail "service has no live MainPID" ;;
        *) ok "resident daemon PID=$main_pid memory=$memory" ;;
    esac
    case "$restarts" in
        ''|*[!0-9]*) fail "cannot read systemd restart count" ;;
        *)
            if [ "$restarts" -le "$max_restarts" ]; then
                ok "restart count $restarts <= $max_restarts"
            else
                fail "restart count $restarts > $max_restarts"
            fi
            ;;
    esac

    if journalctl -u "$service" --since "$registry_age seconds ago" --no-pager 2>/dev/null |
        grep -q 'registry push ->'; then
        ok "exposed-entity registry synchronized within ${registry_age}s"
    else
        fail "no registry synchronization in the last ${registry_age}s"
    fi
fi

if [ -n "$env_file" ]; then
    if [ ! -f "$env_file" ]; then
        fail "environment file missing: $env_file"
    else
        env_mode=$(file_mode "$env_file")
        if [ "$env_mode" = 600 ]; then ok "environment permissions are 0600"
        else fail "environment permissions are $env_mode, expected 600"; fi

        ha_url=$(sed -n 's/^GEIST_HA_URL=//p' "$env_file" | tail -n 1)
        ha_token=$(sed -n 's/^GEIST_HA_TOKEN=//p' "$env_file" | tail -n 1)
        if [ -z "$ha_url" ] || [ -z "$ha_token" ]; then
            fail "environment lacks GEIST_HA_URL or GEIST_HA_TOKEN"
        elif command -v curl >/dev/null 2>&1; then
            if printf 'header = "Authorization: Bearer %s"\n' "$ha_token" |
                curl -fsS --max-time 5 -o /dev/null --config - "$ha_url/api/"; then
                ok "Home Assistant authenticated API is reachable"
            else
                fail "Home Assistant authenticated API is unreachable"
            fi
        else
            fail "curl is required for the Home Assistant API check"
        fi
        ha_token=
    fi
fi

if [ "$failures" -ne 0 ]; then
    printf 'geist-home health: %d failure(s)\n' "$failures" >&2
    exit 1
fi
echo "geist-home health: PASS"
