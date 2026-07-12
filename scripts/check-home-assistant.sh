#!/bin/sh
# Read-only health diagnostics for a deployed geist Home Assistant appliance.
set -eu

usage() {
    cat <<'EOF'
usage: scripts/check-home-assistant.sh [options]

Options:
  --service NAME          systemd unit (default: geist-home)
  --socket PATH           Unix socket (default: /config/geist.sock)
  --max-restarts N        allowed systemd restart count (default: 3)
  --skip-systemd          check only the Unix socket
  --help

No state is changed and no agent request is sent through the socket.
EOF
}

service=geist-home
socket_path=/config/geist.sock
max_restarts=3
skip_systemd=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --service) service=${2-}; shift 2 ;;
        --socket) socket_path=${2-}; shift 2 ;;
        --max-restarts) max_restarts=${2-}; shift 2 ;;
        --skip-systemd) skip_systemd=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "check-home-assistant: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$max_restarts" in
    *[!0-9]*|'') echo "check-home-assistant: restart count must be a non-negative integer" >&2; exit 2 ;;
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
fi

if [ "$failures" -ne 0 ]; then
    printf 'geist-home health: %d failure(s)\n' "$failures" >&2
    exit 1
fi
echo "geist-home health: PASS"
