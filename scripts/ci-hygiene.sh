#!/bin/sh
# CI hygiene on the desktop runner host (docs/HARDWARE_CI.md, "CI hygiene").
# The user timer ci-hygiene (scripts/ci-hygiene.timer, every 10 min) calls
# `run`; .gitea/workflows/host-service.yml (service=ci-hygiene) calls
# `deploy` and `status` on the geist-vulkan host runner.
#   run     remove stale action-cache index.lock files; once a day prune docker
#   deploy  install this script and the user units, enable the timer, run once
#   status  timer state and the last log lines
set -eu
H=$HOME/ci-hygiene
LOG=$H/ci-hygiene.log
UNITS=$HOME/.config/systemd/user
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}

say() { echo "$(date '+%F %T') ci-hygiene: $*"; }

# A checkout of an action repository that died mid-way leaves
# ~/.cache/act*/<hash>/.git/index.lock behind, and every later job that needs
# that action fails at "Prepare all required actions" (2026-10-18, -20).
# Removed only when older than 10 min and no git process works in that
# repository (its cwd or its arguments name it).
locks() {
    find "$HOME"/.cache -maxdepth 4 -path "$HOME/.cache/act*/*/.git/index.lock" \
        -type f -mmin +10 2>/dev/null | while read -r lock; do
        repo=${lock%/.git/index.lock}
        busy=
        for pid in $(pgrep -x git 2>/dev/null); do
            cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null || true)
            case "$cwd" in "$repo" | "$repo"/*) busy=$pid ;; esac
            tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -qF "$repo" && busy=$pid
        done
        if [ -n "$busy" ]; then
            say "kept $lock (git pid $busy works there)"
        else
            age=$(( ($(date +%s) - $(stat -c %Y "$lock")) / 60 ))
            rm -f "$lock" && say "removed $lock (${age} min old)"
        fi
    done
}

# Once a day. Without -a both prunes take dangling images only, so the
# runner images (docker.gitea.com/runner-images:*) and every other tagged
# image stay; no --volumes. The image list before and after goes to the log.
prune() {
    stamp=$H/last-prune
    [ -f "$stamp" ] && [ -z "$(find "$stamp" -mmin +1440)" ] && return 0
    command -v docker >/dev/null || { say "no docker"; return 0; }
    docker image ls --format '{{.Repository}}:{{.Tag}}' | grep -v '<none>' | sort > "$H/images.before"
    say "docker system prune (until=72h)"
    docker system prune -f --filter until=72h 2>&1 | tail -n 1 | sed 's/^/    /'
    say "docker image prune (dangling, until=168h)"
    docker image prune -f --filter until=168h 2>&1 | tail -n 1 | sed 's/^/    /'
    docker image ls --format '{{.Repository}}:{{.Tag}}' | grep -v '<none>' | sort > "$H/images.after"
    gone=$(comm -23 "$H/images.before" "$H/images.after" | tr '\n' ' ')
    [ -n "$gone" ] && say "tagged images gone (unexpected): $gone"
    touch "$stamp"
}

case "${1:-status}" in
run)
    mkdir -p "$H"
    locks
    prune
    ;;
deploy)
    mkdir -p "$H" "$UNITS"
    install -m 755 scripts/ci-hygiene.sh "$H/ci-hygiene.sh"
    install -m 644 scripts/ci-hygiene.service scripts/ci-hygiene.timer "$UNITS/"
    systemctl --user daemon-reload
    systemctl --user enable --now ci-hygiene.timer
    systemctl --user start ci-hygiene.service
    say "deployed"
    "$0" status
    ;;
status)
    systemctl --user --no-pager list-timers ci-hygiene.timer || true
    systemctl --user --no-pager status ci-hygiene.service 2>&1 | head -n 4 || true
    tail -n 30 "$LOG" 2>/dev/null || true
    ls -la "$HOME"/.cache/act*/*/.git/index.lock 2>/dev/null || echo "no index.lock in the action caches"
    docker image ls --format '{{.Repository}}:{{.Tag}} {{.Size}}' 2>/dev/null | grep runner-images || true
    ;;
*)
    echo "usage: $0 run|deploy|status"; exit 2 ;;
esac
