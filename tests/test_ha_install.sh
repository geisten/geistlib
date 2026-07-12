#!/bin/sh
# Hermetic clean-host install, upgrade, and rollback acceptance test.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
tmp=${TMPDIR:-/tmp}/geist-ha-install-$$
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
mkdir -p "$tmp/bin"

# The installer validates executability, not a model. This fixture represents a
# release-provided geist-home; daemon behavior is covered by the branch's C tests.
printf '#!/bin/sh\nprintf "usage: geist-home --serve SOCKET\\n"\n' >"$tmp/bin/geist-home"
chmod 755 "$tmp/bin/geist-home"

install_cmd="$root/scripts/install-home-assistant.sh"
common="--ha-config /srv/ha-config --binary $tmp/bin/geist-home --user geist --work-dir /srv/geist --state-dir /var/lib/geist-home --unit-dir /etc/systemd/system --destdir $tmp/root"

# shellcheck disable=SC2086
"$install_cmd" $common
component="$tmp/root/srv/ha-config/custom_components/geist_conversation"
unit="$tmp/root/etc/systemd/system/geist-home.service"

test -f "$component/manifest.json"
test -f "$component/exposure.py"
test -f "$component/health.py"
test -f "$component/history.py"
test -f "$component/sensor.py"
test -f "$component/diagnostics.py"
test -f "$component/translations/de.json"
test ! -e "$component/__pycache__"
grep -q '^ExecStart=.*/geist-home --serve /srv/ha-config/geist.sock$' "$unit"
! grep -q '^EnvironmentFile=' "$unit"

# Create a recognizable installed v1, then upgrade; the installer must back it up.
printf 'old-version\n' >"$component/old-marker"
# shellcheck disable=SC2086
"$install_cmd" $common
test ! -e "$component/old-marker"

# shellcheck disable=SC2086
"$install_cmd" $common --rollback
test -f "$component/old-marker"

echo "ha_install: clean install + upgrade rollback pass"
