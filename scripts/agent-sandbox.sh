#!/bin/sh
# agent-sandbox.sh — run Claude Code on untrusted input in a throwaway
# container that holds no repository credential (.gitea/workflows/agent.yml,
# AGENT.md §9).
#
# Why a container of its own: the CI job containers have the host's docker
# socket (jobs `docker run -v "$PWD:/src"`), which is root on the runner
# host. Model-written code must not run there. This script starts a sibling
# container without the socket, copies the inputs in with `docker cp` (no
# bind mount of a host path), runs `claude` as uid 1000 with no
# capabilities and no-new-privileges, and copies /work/out back. Nothing the
# model leaves behind runs outside that container; the caller treats
# /work/out as data (scripts/agent_gate.py checks it).
#
# The only secret inside is CLAUDE_CODE_OAUTH_TOKEN, passed to the one
# `docker exec` that runs claude: the model needs it to work. Its
# subprocesses get a scrubbed environment (CLAUDE_CODE_SUBPROCESS_ENV_SCRUB),
# but code running as the same uid can still read it from /proc. It grants
# Claude usage only (no repository access) and is revocable; docs in
# AGENT.md §9.
#
# Usage: agent-sandbox.sh <in-dir> <out-dir> <packages> -- <claude args...>
#   <in-dir>    copied to /work (the repository at /work/src, the prompt at
#               /work/prompt.md, anything else the prompt names)
#   <out-dir>   receives /work/out (created empty before claude runs)
#   <packages>  apt packages besides git and ca-certificates; "+clang-format"
#               adds the pinned clang-format 22 from pip
#   claude's stdout goes to <out-dir>/stdout.md, stdin is /work/prompt.md.
#   After claude, if /work/post.sh exists it runs in the same sandbox, same
#   uid (e.g. `git format-patch` into /work/out).
# Env: CLAUDE_CODE_OAUTH_TOKEN; CLAUDE_BIN (default: download the pinned
#      release below and check its sha256); AGENT_TIMEOUT (seconds, default
#      5100); AGENT_IMAGE (default ubuntu:24.04).
set -eu
dl=

# Claude Code, pinned: version and the manifest's sha256 per platform
# (https://downloads.claude.ai/claude-code-releases/<v>/manifest.json,
# checked against the downloaded binaries). Raise both together.
CLAUDE_VERSION=2.1.285
CLAUDE_SHA256_X64=33dad1ec615a2e08cc78b494f05c110e49916de2c79d78ec8799ebf46b233d29
CLAUDE_SHA256_ARM64=24fac77749bed3d91365d6b6915aa4b824e14318ecb6bc17adbc192f01c9173d

in=$1 out=$2 packages=$3
shift 3
[ "${1:-}" = "--" ] && shift
: "${CLAUDE_CODE_OAUTH_TOKEN:?}"
if [ -z "${CLAUDE_BIN:-}" ]; then
  case $(docker version -f '{{.Server.Arch}}') in
    amd64) plat=linux-x64 sum=$CLAUDE_SHA256_X64 ;;
    arm64) plat=linux-arm64 sum=$CLAUDE_SHA256_ARM64 ;;
    *) echo "agent-sandbox: no pinned claude for this docker host" >&2; exit 1 ;;
  esac
  CLAUDE_BIN=$(mktemp) dl=$CLAUDE_BIN
  curl -fsSL --retry 3 -o "$CLAUDE_BIN" \
    "https://downloads.claude.ai/claude-code-releases/$CLAUDE_VERSION/$plat/claude"
  echo "$sum  $CLAUDE_BIN" | sha256sum -c - >/dev/null ||
    { echo "agent-sandbox: claude $CLAUDE_VERSION $plat: checksum mismatch" >&2; exit 1; }
fi
image=${AGENT_IMAGE:-ubuntu:24.04}
timeout=${AGENT_TIMEOUT:-5100}

cid=$(docker create --init --network bridge \
        --cap-drop ALL --cap-add CHOWN --cap-add DAC_OVERRIDE --cap-add FOWNER \
        --cap-add SETUID --cap-add SETGID --security-opt no-new-privileges \
        --memory 16g --pids-limit 4096 --cpus 8 \
        "$image" sleep infinity)
trap 'docker rm -f "$cid" >/dev/null 2>&1; rm -f "${dl:-}"' EXIT INT TERM
docker start "$cid" >/dev/null

# Root phase: packages only. No secret is in the container yet.
docker exec -u 0 "$cid" mkdir -p /work
docker cp "$in/." "$cid:/work/"
docker cp "$CLAUDE_BIN" "$cid:/usr/local/bin/claude"
apt='' pipcf=0
for p in $packages; do
  case $p in +clang-format) pipcf=1 ;; *) apt="$apt $p" ;; esac
done
docker exec -u 0 -e DEBIAN_FRONTEND=noninteractive -e APT="$apt" -e PIPCF="$pipcf" "$cid" sh -euc '
  chmod 0755 /usr/local/bin/claude
  id -u ubuntu >/dev/null 2>&1 || useradd -m -u 1000 ubuntu
  apt-get -o Acquire::Retries=3 update -qq
  apt-get install -qq -y --no-install-recommends ca-certificates git $APT $([ "$PIPCF" = 0 ] || echo python3-pip) >/dev/null 2>&1
  [ "$PIPCF" = 0 ] || python3 -m pip install -q --break-system-packages clang-format==22.1.5
  mkdir -p /work/out
  chown -R 1000:1000 /work
'
# Fail closed: the model's uid must not reach a docker daemon.
if docker exec -u 1000:1000 "$cid" sh -c 'test -e /var/run/docker.sock || test -e /run/docker.sock'; then
  echo "agent-sandbox: a docker socket is visible inside the sandbox" >&2; exit 1
fi

# Model phase: uid 1000, no capabilities (a non-root uid gets none, and
# no-new-privileges blocks setuid binaries), only the OAuth token.
set +e
docker exec -i -u 1000:1000 -w /work/src \
  -e HOME=/home/ubuntu -e CLAUDE_CODE_OAUTH_TOKEN \
  -e CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1 -e IS_SANDBOX=1 -e DISABLE_AUTOUPDATER=1 \
  "$cid" timeout "$timeout" claude "$@" < "$in/prompt.md" > "$out.stdout.tmp"
rc=$?
set -e
# Nothing the model started outlives it (a background process could still
# be writing into /work/out while it is copied).
killall1000() { docker exec -u 1000:1000 "$cid" sh -c 'kill -KILL -1 2>/dev/null; true'; }
killall1000
if docker exec -u 0 "$cid" test -f /work/post.sh; then
  docker exec -u 1000:1000 -w /work/src -e HOME=/home/ubuntu "$cid" sh /work/post.sh ||
    { [ "$rc" -ne 0 ] || rc=1; }
  killall1000
fi
mkdir -p "$out"
docker cp "$cid:/work/out/." "$out/"
mv "$out.stdout.tmp" "$out/stdout.md"
exit "$rc"
