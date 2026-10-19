#!/bin/sh
# Installs scripts/git-cache.py on the desktop runner host and points the
# docker runners' job containers at it (docs/HARDWARE_CI.md, "CI git cache").
# Runs from .gitea/workflows/git-cache.yml on the geist-vulkan host runner.
#   deploy   install/update the cache service, wire the docker runners
#   rollback restore the runners' configs from before the cache, stop it
#   status   service state and the last log lines
set -eu
LISTEN=${GIT_CACHE_LISTEN:-192.168.178.43:8417}
C=$HOME/git-cache
TOKEN_FILE=$HOME/.config/git-cache/token
UNIT=$HOME/.config/systemd/user/git-cache.service
RUNNERS="gitea-runner-ci gitea-runner-tsan"
IMAGE=docker.gitea.com/runner-images:ubuntu-24.04
export XDG_RUNTIME_DIR=/run/user/$(id -u)

say() { echo "git-cache: $*"; }

# Restart the docker runners once no job container runs (up to 40 min;
# after that a running job is cut and has to be rerun).
restart_runners() {
  i=0
  while [ -n "$(docker ps -q --filter name=GITEA-ACTIONS-TASK)" ] && [ $i -lt 80 ]; do
    [ $((i % 10)) = 0 ] && say "waiting for $(docker ps -q --filter name=GITEA-ACTIONS-TASK | wc -l) job container(s)"
    sleep 30; i=$((i + 1))
  done
  for r in $RUNNERS; do systemctl --user restart "$r"; done
  sleep 3
  for r in $RUNNERS; do systemctl --user is-active "$r" >/dev/null || { say "$r not active"; return 1; }; done
  say "runners restarted"
}

rollback() {
  for r in $RUNNERS; do
    cfg=$HOME/$r/config.yaml
    [ -f "$cfg.pre-git-cache" ] && cp -p "$cfg.pre-git-cache" "$cfg" && say "$r config restored"
  done
  restart_runners
  systemctl --user disable --now git-cache 2>/dev/null || true
  say "rolled back"
}

case "${1:-status}" in
status)
  systemctl --user --no-pager status git-cache | head -5 || true
  tail -n 30 "$C/git-cache.log" 2>/dev/null || true
  ;;
rollback)
  rollback
  ;;
deploy)
  [ -n "${GIT_CACHE_TOKEN:-}" ] || { say "secret GIT_CACHE_TOKEN missing"; exit 1; }
  mkdir -p "$C" "$(dirname "$TOKEN_FILE")" "$(dirname "$UNIT")"
  (umask 077; printf '%s\n' "$GIT_CACHE_TOKEN" > "$TOKEN_FILE")
  install -m 755 scripts/git-cache.py "$C/git-cache.py"
  cat > "$UNIT" <<EOF
[Unit]
Description=CI git cache for the docker runners' job containers (geistlib scripts/git-cache.py)

[Service]
ExecStart=/usr/bin/python3 %h/git-cache/git-cache.py --listen $LISTEN --root %h/git-cache/repos --token-file %h/.config/git-cache/token
Restart=always
RestartSec=2
StandardOutput=append:%h/git-cache/git-cache.log
StandardError=append:%h/git-cache/git-cache.log

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable git-cache >/dev/null 2>&1
  systemctl --user restart git-cache
  # From a job-like container: info/refs (first call clones the mirror) and
  # a shallow fetch through the insteadOf mapping.
  i=0
  until code=$(docker run --rm "$IMAGE" curl -s -m 120 -o /dev/null -w '%{http_code}' \
        "http://$LISTEN/geisten/geistlib.git/info/refs?service=git-upload-pack") && [ "$code" = 200 ]; do
    i=$((i + 1)); [ $i -gt 5 ] && { say "cache not answering ($code)"; systemctl --user stop git-cache; exit 1; }; sleep 10
  done
  docker run --rm "$IMAGE" sh -c "git init -q /t && cd /t && \
    git -c url.http://$LISTEN/.insteadOf=https://git.geisten.net/ fetch -q --depth=1 \
      https://git.geisten.net/geisten/geistlib.git main && git log --oneline -1 FETCH_HEAD" \
    || { say "shallow fetch through the cache failed"; systemctl --user stop git-cache; exit 1; }
  say "cache answers on $LISTEN"
  # Wire the docker runners: git in their job containers fetches from the
  # cache, pushes still go to Gitea.
  for r in $RUNNERS; do
    cfg=$HOME/$r/config.yaml
    [ -f "$cfg" ] || { say "no $cfg"; continue; }
    [ -f "$cfg.pre-git-cache" ] || cp -p "$cfg" "$cfg.pre-git-cache"
    python3 - "$cfg.pre-git-cache" "$cfg" "$LISTEN" <<'PY'
import sys
src, dst, listen = sys.argv[1:]
keys = [
    ("GIT_CONFIG_COUNT", "2"),
    ("GIT_CONFIG_KEY_0", f"url.http://{listen}/.insteadOf"),
    ("GIT_CONFIG_VALUE_0", "https://git.geisten.net/"),
    ("GIT_CONFIG_KEY_1", "url.https://git.geisten.net/.pushInsteadOf"),
    ("GIT_CONFIG_VALUE_1", "https://git.geisten.net/"),
]
add = [f"    {k}: \"{v}\"\n" for k, v in keys]
lines = open(src).read().splitlines(keepends=True)
out, done = [], False
for i, l in enumerate(lines):
    out.append(l)
    if not done and l.rstrip() == "  envs:":
        out += add; done = True
if not done:
    j = next(i for i, l in enumerate(out) if l.rstrip() == "runner:")
    out[j + 1:j + 1] = ["  # CI git cache (scripts/git-cache.py)\n", "  envs:\n"] + add
open(dst, "w").write("".join(out))
PY
    say "$r config wired"; grep -n 'GIT_CONFIG' "$cfg" | sed 's/^/    /'
  done
  restart_runners || { rollback; exit 1; }
  ;;
*)
  say "usage: $0 deploy|rollback|status"; exit 2 ;;
esac
