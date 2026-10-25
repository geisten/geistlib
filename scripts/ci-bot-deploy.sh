#!/bin/sh
# Installs scripts/ci-bot.py (with scripts/gitea_api.py) as the user service `ci-bot` on the desktop
# runner host (docs/HARDWARE_CI.md, "CI bot"). Runs from
# .gitea/workflows/host-service.yml (service=ci-bot) on the geist-vulkan
# host runner.
#   deploy   install/update the bot (token from $CI_BOT_TOKEN), prove one
#            dry-run poll works, start it (CI_BOT_DRY_RUN=true: in dry-run)
#   rollback stop and disable the service (the previous scripts stay as
#            *.py.prev; deploy an older commit to go back a version)
#   status   service state and the last log lines
set -eu
C=$HOME/ci-bot
TOKEN_FILE=$HOME/.config/ci-bot/token
UNIT=$HOME/.config/systemd/user/ci-bot.service
export XDG_RUNTIME_DIR=/run/user/$(id -u)

say() { echo "ci-bot: $*"; }

case "${1:-status}" in
status)
  systemctl --user --no-pager status ci-bot | head -5 || true
  tail -n 40 "$C/ci-bot.log" 2>/dev/null || true
  ;;
rollback)
  systemctl --user disable --now ci-bot 2>/dev/null || true
  say "stopped and disabled"
  ;;
deploy)
  if [ -z "${CI_BOT_TOKEN:-}" ]; then
    say "the org Actions secret CI_BOT_TOKEN is missing or empty."
    say "Create a Gitea token (scope write:repository) for the bot's user and add it"
    say "as secret CI_BOT_TOKEN in the geisten org (Settings > Actions > Secrets)."
    exit 1
  fi
  command -v git >/dev/null || { say "git not on PATH"; exit 1; }
  mkdir -p "$C" "$(dirname "$TOKEN_FILE")" "$(dirname "$UNIT")"
  (umask 077; printf '%s\n' "$CI_BOT_TOKEN" > "$TOKEN_FILE")
  chmod 600 "$TOKEN_FILE"
  for f in ci-bot.py gitea_api.py; do
    if [ -f "$C/$f" ]; then cp -p "$C/$f" "$C/$f.prev"; fi
  done
  install -m 755 scripts/ci-bot.py "$C/ci-bot.py"
  install -m 644 scripts/gitea_api.py "$C/gitea_api.py"
  # One dry-run poll with the new token and script: API reachable, token
  # valid, git clones work. Its own state file, so the live state is untouched.
  say "smoke: one dry-run poll"
  python3 "$C/ci-bot.py" --once --dry-run --home "$C" --token-file "$TOKEN_FILE" \
      --state "$C/state-smoke.json" > "$C/smoke.log" 2>&1 || true
  sed 's/^/    /' "$C/smoke.log"
  if grep -qE 'listing PRs failed|tick failed|Traceback' "$C/smoke.log" || \
     ! grep -q 'ci-bot start' "$C/smoke.log"; then
    say "smoke poll failed; service not (re)started"
    exit 1
  fi
  DRY=
  [ "${CI_BOT_DRY_RUN:-false}" = true ] && DRY=" --dry-run"
  cat > "$UNIT" <<EOF
[Unit]
Description=CI bot: infra reruns, CHANGELOG rebases, dependent merges (geistlib scripts/ci-bot.py)
After=network-online.target

[Service]
ExecStart=/usr/bin/python3 %h/ci-bot/ci-bot.py --home %h/ci-bot --token-file %h/.config/ci-bot/token$DRY
Restart=always
RestartSec=30
StandardOutput=append:%h/ci-bot/ci-bot.log
StandardError=append:%h/ci-bot/ci-bot.log

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable ci-bot >/dev/null 2>&1
  systemctl --user restart ci-bot
  sleep 5
  systemctl --user is-active ci-bot >/dev/null || { say "service not active"; tail -n 20 "$C/ci-bot.log"; exit 1; }
  say "running${DRY:+ (dry-run)}; log $C/ci-bot.log"
  ;;
*)
  say "usage: $0 deploy|rollback|status"; exit 2 ;;
esac
