#!/bin/sh
# The CI git cache on the Mac (docs/HARDWARE_CI.md, "CI git cache"):
# scripts/git-cache.py on 127.0.0.1:8417 for the host-mode runners mac-light
# and mac-m1max, run by the LaunchAgent net.geisten.git-cache.
#   install  copy git-cache.py and this script to ~/git-cache, (re)load the
#            LaunchAgent. It starts once ~/.config/git-cache/token exists.
#   run      the LaunchAgent's program: start the cache and, once it serves
#            geistlib, write ~/git-cache/ci.gitconfig (insteadOf to the
#            cache, pushInsteadOf back to Gitea); remove it when stopping.
#   status   agent state and the last log lines
# The runners include that file through their envs (GIT_CONFIG_COUNT=1,
# GIT_CONFIG_KEY_0=include.path, GIT_CONFIG_VALUE_0=~/git-cache/ci.gitconfig);
# a missing file is ignored, so jobs fetch from Gitea while the cache is down
# or has no token.
set -eu
C=$HOME/git-cache
TOKEN_FILE=$HOME/.config/git-cache/token
INC=$C/ci.gitconfig
LISTEN=127.0.0.1:8417
LABEL=net.geisten.git-cache
PLIST=$HOME/Library/LaunchAgents/$LABEL.plist

say() { echo "$(date '+%F %T') git-cache: $*"; }

case "${1:-status}" in
install)
  here=$(cd "$(dirname "$0")" && pwd)
  mkdir -p "$C/repos" "$(dirname "$TOKEN_FILE")"
  chmod 700 "$(dirname "$TOKEN_FILE")"
  install -m 755 "$here/git-cache.py" "$C/git-cache.py"
  [ "$here/git-cache-mac.sh" = "$C/git-cache-mac.sh" ] || install -m 755 "$here/git-cache-mac.sh" "$C/git-cache-mac.sh"
  cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/sh</string><string>$C/git-cache-mac.sh</string><string>run</string></array>
  <key>KeepAlive</key>
  <dict><key>PathState</key><dict><key>$TOKEN_FILE</key><true/></dict></dict>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>GIT_CONFIG_NOSYSTEM</key><string>1</string>
    <key>GIT_CONFIG_GLOBAL</key><string>/dev/null</string>
    <key>GIT_TERMINAL_PROMPT</key><string>0</string>
  </dict>
  <key>StandardOutPath</key><string>$C/git-cache.log</string>
  <key>StandardErrorPath</key><string>$C/git-cache.log</string>
</dict>
</plist>
EOF
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$PLIST"
  echo "installed; $( [ -f "$TOKEN_FILE" ] && echo "token present" || echo "waiting for $TOKEN_FILE")"
  ;;
run)
  # No system/global git config: the keychain helper must not hand the
  # upstream fetch another token.
  rm -f "$INC"
  /usr/bin/python3 "$C/git-cache.py" --listen "$LISTEN" --root "$C/repos" --token-file "$TOKEN_FILE" &
  pid=$!
  trap 'rm -f "$INC"; kill $pid 2>/dev/null' EXIT
  trap 'exit 143' TERM INT HUP
  sleep 2 & wait $!
  # Back off: a wrong token is a failed login each time, and Gitea's host
  # bans an address after a few in a row.
  delay=30
  until [ "$(curl -s -m 900 -o /dev/null -w '%{http_code}' \
        "http://$LISTEN/geisten/geistlib.git/info/refs?service=git-upload-pack")" = 200 ]; do
    kill -0 $pid 2>/dev/null || exit 1
    say "not serving geistlib yet (token?), jobs fetch from Gitea; retry in $delay s"
    sleep $delay & wait $!
    delay=$((delay * 2 > 3600 ? 3600 : delay * 2))
  done
  printf '[url "http://%s/"]\n\tinsteadOf = https://git.geisten.net/\n[url "https://git.geisten.net/"]\n\tpushInsteadOf = https://git.geisten.net/\n' \
    "$LISTEN" > "$INC.tmp" && mv "$INC.tmp" "$INC"
  say "serving; jobs fetch through $INC"
  wait $pid
  ;;
status)
  launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null | grep -E '^\s*(state|pid|last exit code) =' || echo "$LABEL not loaded"
  [ -f "$TOKEN_FILE" ] && echo "token present" || echo "no token file"
  [ -f "$INC" ] && echo "jobs wired ($INC present)" || echo "jobs not wired ($INC absent)"
  tail -n 20 "$C/git-cache.log" 2>/dev/null || true
  ;;
*)
  echo "usage: $0 install|run|status"; exit 2 ;;
esac
