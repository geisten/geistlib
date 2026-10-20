#!/bin/sh
# Installs scripts/git-cache.py on a runner host and points that host's
# runners at it (docs/HARDWARE_CI.md, "CI git cache"). Runs from
# .gitea/workflows/git-cache.yml on the host's own host-mode runner.
#   deploy   install/update the cache service, wire the runners
#   rollback restore the runners' configs from before the cache, stop it
#   status   service state and the last log lines
# GIT_CACHE_HOST picks the host profile:
#   desktop  (geist-vulkan) user service on the LAN address; docker runners
#            gitea-runner-ci and gitea-runner-tsan (user units)
#   pi5      (pi5) system service on 127.0.0.1 and the docker bridge;
#            docker runner gitea-runner-ci (bridge address) and host runner
#            gitea-runner (127.0.0.1), both system units. The deploy job runs
#            on gitea-runner itself, so that one restarts after the job ends.
set -eu
HOST=${GIT_CACHE_HOST:-desktop}
C=$HOME/git-cache
TOKEN_FILE=$HOME/.config/git-cache/token
IMAGE=docker.gitea.com/runner-images:ubuntu-24.04

say() { echo "git-cache: $*"; }

case "$HOST" in
desktop)
  LISTEN=192.168.178.43:8417
  RUNNERS="gitea-runner-ci:$LISTEN gitea-runner-tsan:$LISTEN"
  UNIT=$HOME/.config/systemd/user/git-cache.service
  export XDG_RUNTIME_DIR=/run/user/$(id -u)
  sc() { systemctl --user "$@"; }
  ;;
pi5)
  LISTEN="127.0.0.1:8417 172.17.0.1:8417"
  RUNNERS="gitea-runner-ci:172.17.0.1:8417 gitea-runner:127.0.0.1:8417"
  UNIT=/etc/systemd/system/git-cache.service
  sc() { sudo -n systemctl "$@"; }
  ;;
*)
  say "unknown GIT_CACHE_HOST $HOST"; exit 2 ;;
esac

# Restart the runners once idle. On the desktop: no job container runs (up
# to 40 min; after that a running job is cut and has to be rerun). On the
# Pi this job holds the board lock, which every job on both runners holds
# too, so gitea-runner-ci is idle now; gitea-runner runs this job and is
# restarted by a transient unit once the lock is free and it has no child.
restart_runners() {
  if [ "$HOST" = pi5 ]; then
    sc restart gitea-runner-ci
    sleep 3
    sc is-active gitea-runner-ci >/dev/null || { say "gitea-runner-ci not active"; return 1; }
    cat > "$C/restart-host-runner.sh" <<'EOF'
#!/bin/sh
# git-cache-deploy.sh: restart the host runner after the deploy job ended.
pid=$(systemctl show -p MainPID --value gitea-runner)
i=0
until flock -n /tmp/pi-heavy.lock sh -c "[ -z \"\$(pgrep -P $pid)\" ] && sudo -n systemctl restart gitea-runner"; do
  i=$((i + 1)); [ $i -gt 4320 ] && { echo "$(date '+%F %T') host runner restart: gave up"; exit 1; }
  sleep 5
done
echo "$(date '+%F %T') host runner gitea-runner restarted"
EOF
    sudo -n systemd-run --quiet --collect --uid="$(id -u)" --gid="$(id -g)" \
      --unit="git-cache-host-restart-$(date +%s)" \
      -p StandardOutput=append:"$C/git-cache.log" -p StandardError=append:"$C/git-cache.log" \
      /bin/sh "$C/restart-host-runner.sh"
    say "gitea-runner-ci restarted; gitea-runner restarts after this job"
    return 0
  fi
  i=0
  while [ -n "$(docker ps -q --filter name=GITEA-ACTIONS-TASK)" ] && [ $i -lt 80 ]; do
    [ $((i % 10)) = 0 ] && say "waiting for $(docker ps -q --filter name=GITEA-ACTIONS-TASK | wc -l) job container(s)"
    sleep 30; i=$((i + 1))
  done
  for r in $RUNNERS; do sc restart "${r%%:*}"; done
  sleep 3
  for r in $RUNNERS; do sc is-active "${r%%:*}" >/dev/null || { say "${r%%:*} not active"; return 1; }; done
  say "runners restarted"
}

rollback() {
  for r in $RUNNERS; do
    cfg=$HOME/${r%%:*}/config.yaml
    [ -f "$cfg.pre-git-cache" ] && cp -p "$cfg.pre-git-cache" "$cfg" && say "${r%%:*} config restored"
  done
  restart_runners
  sc disable --now git-cache 2>/dev/null || true
  say "rolled back"
}

case "${1:-status}" in
status)
  sc --no-pager status git-cache | head -5 || true
  for r in $RUNNERS; do
    say "${r%%:*}: $(grep -c 'GIT_CONFIG_' "$HOME/${r%%:*}/config.yaml" 2>/dev/null || true) GIT_CONFIG lines," \
        "running since $(sc show -p ActiveEnterTimestamp --value "${r%%:*}" 2>/dev/null || true)"
  done
  tail -n 30 "$C/git-cache.log" 2>/dev/null || true
  ;;
rollback)
  rollback
  ;;
deploy)
  [ -n "${GIT_CACHE_TOKEN:-}" ] || { say "secret GIT_CACHE_TOKEN missing"; exit 1; }
  mkdir -p "$C" "$(dirname "$TOKEN_FILE")"
  (umask 077; printf '%s\n' "$GIT_CACHE_TOKEN" > "$TOKEN_FILE")
  install -m 755 scripts/git-cache.py "$C/git-cache.py"
  args=""
  for l in $LISTEN; do args="$args --listen $l"; done
  if [ "$HOST" = pi5 ]; then
    unit_user="User=$(id -un)
After=network-online.target docker.service
Wants=network-online.target"
    home=$HOME
  else
    unit_user=""
    home=%h
  fi
  cat > "$C/git-cache.service" <<EOF
[Unit]
Description=CI git cache for the runners' jobs (geistlib scripts/git-cache.py)
$unit_user

[Service]
ExecStart=/usr/bin/python3 $home/git-cache/git-cache.py$args --root $home/git-cache/repos --token-file $home/.config/git-cache/token
Restart=always
RestartSec=2
StandardOutput=append:$home/git-cache/git-cache.log
StandardError=append:$home/git-cache/git-cache.log

[Install]
WantedBy=$([ "$HOST" = pi5 ] && echo multi-user.target || echo default.target)
EOF
  if [ "$HOST" = pi5 ]; then
    sudo -n install -m 644 "$C/git-cache.service" "$UNIT"
  else
    mkdir -p "$(dirname "$UNIT")"; install -m 644 "$C/git-cache.service" "$UNIT"
  fi
  sc daemon-reload
  sc enable git-cache >/dev/null 2>&1
  sc restart git-cache
  # Every listen address: info/refs (the first call clones the mirror) from
  # where the jobs connect — a job-like container for the docker address,
  # this host for 127.0.0.1 — and a shallow fetch through the insteadOf
  # mapping.
  for l in $LISTEN; do
    case "$l" in 127.*) run="sh -c" ;; *) run="docker run --rm $IMAGE sh -c" ;; esac
    i=0
    until code=$($run "curl -s -m 600 -o /dev/null -w '%{http_code}' \
          'http://$l/geisten/geistlib.git/info/refs?service=git-upload-pack'") && [ "$code" = 200 ]; do
      i=$((i + 1)); [ $i -gt 5 ] && { say "cache not answering on $l ($code)"; sc stop git-cache; exit 1; }; sleep 10
    done
    $run "t=\$(mktemp -d) && git init -q \$t && cd \$t && \
      git -c url.http://$l/.insteadOf=https://git.geisten.net/ fetch -q --depth=1 \
        https://git.geisten.net/geisten/geistlib.git main && git log --oneline -1 FETCH_HEAD; rm -rf \$t" \
      || { say "shallow fetch through $l failed"; sc stop git-cache; exit 1; }
    say "cache answers on $l"
  done
  # Wire the runners: git in their jobs fetches from the cache, pushes still
  # go to Gitea.
  for r in $RUNNERS; do
    name=${r%%:*}; listen=${r#*:}
    cfg=$HOME/$name/config.yaml
    [ -f "$cfg" ] || { say "no $cfg"; continue; }
    [ -f "$cfg.pre-git-cache" ] || cp -p "$cfg" "$cfg.pre-git-cache"
    python3 - "$cfg.pre-git-cache" "$cfg" "$listen" <<'PY'
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
    say "$name config wired"; grep -n 'GIT_CONFIG' "$cfg" | sed 's/^/    /'
  done
  restart_runners || { rollback; exit 1; }
  ;;
*)
  say "usage: $0 deploy|rollback|status"; exit 2 ;;
esac
