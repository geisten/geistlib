#!/bin/sh
# Disposable real-Home-Assistant acceptance test for the geist home boundary.
# Requires an existing HA container image and a geist-home binary with an
# embedded model. It never talks to the host's productive HA instance.
set -eu

image=${HA_IMAGE:-ghcr.io/home-assistant/home-assistant:stable}
port=${HA_TEST_PORT:-18123}
binary=${GEIST_HOME_BINARY:-}
container=geist-ha-live-$$
tmp=${TMPDIR:-/tmp}/geist-ha-live-$$
base=http://127.0.0.1:$port
daemon_pid=

if [ -z "$binary" ] || [ ! -x "$binary" ]; then
    echo "test_home_assistant_live: set GEIST_HOME_BINARY to an executable geist-home" >&2
    exit 2
fi
command -v docker >/dev/null 2>&1 || {
    echo "test_home_assistant_live: docker is required" >&2
    exit 2
}
command -v curl >/dev/null 2>&1 || {
    echo "test_home_assistant_live: curl is required" >&2
    exit 2
}
command -v python3 >/dev/null 2>&1 || {
    echo "test_home_assistant_live: python3 is required" >&2
    exit 2
}

cleanup() {
    if [ -n "$daemon_pid" ]; then
        kill "$daemon_pid" >/dev/null 2>&1 || true
        wait "$daemon_pid" 2>/dev/null || true
    fi
    docker rm -f "$container" >/dev/null 2>&1 || true
    # HA writes some config files as the container's root user. Remove them
    # through the same image so a non-root host runner can clean its temp dir.
    if [ -d "$tmp" ]; then
        docker run --rm --entrypoint sh -v "$tmp:/work" "$image" \
            -c 'rm -rf /work/config /work/registry.txt /work/*.json /work/*.txt' \
            >/dev/null 2>&1 || true
    fi
    rm -rf "$tmp"
}
trap cleanup EXIT HUP INT TERM
mkdir -p "$tmp/config"

cat >"$tmp/config/configuration.yaml" <<'EOF'
default_config:

homeassistant:
  name: geist disposable acceptance
  latitude: 52.52
  longitude: 13.405
  elevation: 34
  unit_system: metric
  time_zone: Europe/Berlin

input_boolean:
  exposed_light_power:
    name: Exposed light power
  hidden_light_power:
    name: Hidden light power

template:
  - light:
      - name: Flur
        unique_id: geist_exposed_test_light
        state: "{{ is_state('input_boolean.exposed_light_power', 'on') }}"
        turn_on:
          - action: input_boolean.turn_on
            target:
              entity_id: input_boolean.exposed_light_power
        turn_off:
          - action: input_boolean.turn_off
            target:
              entity_id: input_boolean.exposed_light_power
      - name: Keller
        unique_id: geist_hidden_test_light
        state: "{{ is_state('input_boolean.hidden_light_power', 'on') }}"
        turn_on:
          - action: input_boolean.turn_on
            target:
              entity_id: input_boolean.hidden_light_power
        turn_off:
          - action: input_boolean.turn_off
            target:
              entity_id: input_boolean.hidden_light_power
EOF

cat >"$tmp/registry.txt" <<'EOF'
light.flur | light | flurlicht, licht flur, hallway light
EOF

docker run -d --name "$container" \
    -p "127.0.0.1:$port:8123" \
    -v "$tmp/config:/config" \
    "$image" >/dev/null

i=0
until curl -fsS "$base/api/onboarding" >"$tmp/onboarding.json" 2>/dev/null; do
    i=$((i + 1))
    if [ "$i" -ge 180 ]; then
        echo "test_home_assistant_live: HA did not become ready" >&2
        docker logs "$container" >&2 || true
        exit 1
    fi
    sleep 1
done

client_id=http://localhost/
curl -fsS -X POST "$base/api/onboarding/users" \
    -H 'Content-Type: application/json' \
    --data '{"name":"geist test","username":"geist-test","password":"geist-test-only","client_id":"http://localhost/","language":"en"}' \
    >"$tmp/user.json"

auth_code=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["auth_code"])' "$tmp/user.json")
curl -fsS -X POST "$base/auth/token" \
    -H 'Content-Type: application/x-www-form-urlencoded' \
    --data-urlencode 'grant_type=authorization_code' \
    --data-urlencode "code=$auth_code" \
    --data-urlencode "client_id=$client_id" \
    >"$tmp/token.json"
token=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["access_token"])' "$tmp/token.json")
chmod 600 "$tmp/token.json"

api_get_state() {
    curl -fsS "$base/api/states/$1" -H "Authorization: Bearer $token" |
        python3 -c 'import json,sys; print(json.load(sys.stdin)["state"])'
}

assert_state() {
    entity=$1
    expected=$2
    actual=$(api_get_state "$entity")
    if [ "$actual" != "$expected" ]; then
        echo "test_home_assistant_live: $entity expected=$expected actual=$actual" >&2
        exit 1
    fi
}

# Wait for both template entities; onboarding can respond just before all
# startup tasks have settled.
i=0
until api_get_state light.flur >/dev/null 2>&1 &&
    api_get_state light.keller >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -ge 60 ]; then
        echo "test_home_assistant_live: template lights did not appear" >&2
        docker logs "$container" >&2 || true
        exit 1
    fi
    sleep 1
done

# Fresh input_boolean helpers may restore as unknown. Initialize the two
# fixture backends through HA's real service API, then wait for template state.
curl -fsS -X POST "$base/api/services/input_boolean/turn_off" \
    -H "Authorization: Bearer $token" \
    -H 'Content-Type: application/json' \
    --data '{"entity_id":["input_boolean.exposed_light_power","input_boolean.hidden_light_power"]}' \
    >/dev/null
i=0
until [ "$(api_get_state light.flur)" = off ] &&
    [ "$(api_get_state light.keller)" = off ]; do
    i=$((i + 1))
    if [ "$i" -ge 30 ]; then
        echo "test_home_assistant_live: template lights did not settle to off" >&2
        exit 1
    fi
    sleep 1
done

assert_state light.flur off
assert_state light.keller off

GEIST_HA_URL="$base" GEIST_HA_TOKEN="$token" \
    GEIST_HOME_REGISTRY="$tmp/registry.txt" \
    "$binary" "Schalte das Licht im Flur ein" >"$tmp/exposed-response.txt"
assert_state light.flur on
assert_state light.keller off

# Measure warm product latency through the resident Unix-socket daemon. Each
# sample includes model routing, the authenticated HA service call and answer.
sock=$tmp/geist.sock
GEIST_HA_URL="$base" GEIST_HA_TOKEN="$token" \
    GEIST_HOME_REGISTRY="$tmp/registry.txt" \
    "$binary" --serve "$sock" >"$tmp/daemon.log" 2>&1 &
daemon_pid=$!
i=0
while [ ! -S "$sock" ]; do
    i=$((i + 1))
    if [ "$i" -ge 60 ]; then
        echo "test_home_assistant_live: geist daemon socket did not appear" >&2
        cat "$tmp/daemon.log" >&2
        exit 1
    fi
    sleep 1
done

ask_timed() {
    python3 -c 'import socket,sys,time
s=socket.socket(socket.AF_UNIX); s.settimeout(60); start=time.monotonic(); s.connect(sys.argv[1]); s.sendall(sys.argv[2].encode()+b"\n"); s.shutdown(socket.SHUT_WR); data=b""
while True:
    chunk=s.recv(4096)
    if not chunk: break
    data+=chunk
open(sys.argv[3],"wb").write(data)
print(round((time.monotonic()-start)*1000))' "$sock" "$1" "$2"
}

# Warm-up is intentionally excluded from the distribution.
ask_timed "Ist das Licht im Flur an?" "$tmp/warm-response.txt" >/dev/null
: >"$tmp/latency-ms.txt"
i=1
while [ "$i" -le 10 ]; do
    if [ $((i % 2)) -eq 1 ]; then
        request="Schalte das Licht im Flur aus"
        expected=off
    else
        request="Schalte das Licht im Flur ein"
        expected=on
    fi
    ask_timed "$request" "$tmp/latency-response-$i.txt" >>"$tmp/latency-ms.txt"
    assert_state light.flur "$expected"
    i=$((i + 1))
done
latency_summary=$(python3 -c 'import math,statistics,sys
v=sorted(map(int,open(sys.argv[1]).read().split()))
print(f"samples={len(v)} p50_ms={round(statistics.median(v))} p95_ms={v[math.ceil(.95*len(v))-1]} min_ms={v[0]} max_ms={v[-1]}")' "$tmp/latency-ms.txt")

# The hidden entity is present in HA but absent from the geist registry. The
# request must not mutate it, irrespective of the model's wording.
ask_timed "Schalte das Licht im Keller ein" "$tmp/hidden-response.txt" >/dev/null
assert_state light.keller off

printf 'live_ha: PASS exposed=%s hidden=%s\n' \
    "$(api_get_state light.flur)" \
    "$(api_get_state light.keller)"
printf 'live_ha_latency: %s\n' "$latency_summary"
