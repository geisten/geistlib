#!/bin/sh
# Model-free contract for guided HA setup detection and non-secret dry-run.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
tmp=${TMPDIR:-/tmp}/geist-ha-setup-$$
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
mkdir -p "$tmp/ha"
tmp=$(CDPATH= cd -- "$tmp" && pwd)
printf '#!/bin/sh\nprintf "usage: geist-home --serve SOCKET\\n"\n' >"$tmp/geist-home"
chmod 755 "$tmp/geist-home"

output=$("$root/scripts/setup-home-assistant.sh" \
    --ha-config "$tmp/ha" \
    --binary "$tmp/geist-home" \
    --dry-run)
printf '%s\n' "$output" | grep -q '^platform='
printf '%s\n' "$output" | grep -q "^binary=$tmp/geist-home$"
printf '%s\n' "$output" | grep -q "^ha_config=$tmp/ha$"
printf '%s\n' "$output" | grep -q '^setup-home-assistant: dry-run PASS$'
test ! -e "$tmp/ha/custom_components/geist_conversation"

echo "ha_setup: guided detection dry-run pass"
