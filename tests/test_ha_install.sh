#!/bin/sh
# Hermetic clean-host install, upgrade, and rollback acceptance test.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
tmp=${TMPDIR:-/tmp}/geist-ha-install-$$
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
mkdir -p "$tmp/bin" "$tmp/source"

# The installer validates executability, not a model. This fixture represents a
# release-provided geist-home; daemon behavior is covered by the branch's C tests.
printf '#!/bin/sh\nprintf "usage: geist-home --serve SOCKET\\n"\n' >"$tmp/bin/geist-home"
chmod 755 "$tmp/bin/geist-home"
printf 'GEIST_HA_URL=http://localhost:8123\nGEIST_HA_TOKEN=test-only\n' >"$tmp/source/home.env"
chmod 600 "$tmp/source/home.env"

install_cmd="$root/scripts/install-home-assistant.sh"
common="--ha-config /srv/ha-config --binary $tmp/bin/geist-home --env-file $tmp/source/home.env --user geist --work-dir /srv/geist --state-dir /var/lib/geist-home --unit-dir /etc/systemd/system --destdir $tmp/root"

# shellcheck disable=SC2086
"$install_cmd" $common
component="$tmp/root/srv/ha-config/custom_components/geist_conversation"
unit="$tmp/root/etc/systemd/system/geist-home.service"
env="$tmp/root/var/lib/geist-home/geist-home.env"

test -f "$component/manifest.json"
test -f "$component/registry.py"
test -f "$component/transport.py"
test ! -e "$component/__pycache__"
case "$(uname -s)" in
    Darwin | *BSD) env_mode=$(stat -f '%Lp' "$env") ;;
    *) env_mode=$(stat -c '%a' "$env") ;;
esac
test "$env_mode" = 600
grep -q '^ExecStart=.*/geist-home --serve /srv/ha-config/geist.sock$' "$unit"
grep -q '^EnvironmentFile=/var/lib/geist-home/geist-home.env$' "$unit"

# Create a recognizable installed v1, then upgrade; the installer must back it up.
printf 'old-version\n' >"$component/old-marker"
# shellcheck disable=SC2086
"$install_cmd" $common
test ! -e "$component/old-marker"

# shellcheck disable=SC2086
"$install_cmd" $common --rollback
test -f "$component/old-marker"

echo "ha_install: clean install + secret mode + upgrade rollback pass"
