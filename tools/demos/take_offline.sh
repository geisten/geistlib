#!/usr/bin/env bash
# take_offline.sh — record assets/demo-offline-box.gif: the whole shell runs
# under `unshare -n` (a network namespace with NO network), the failing ping
# is the in-frame proof. Reconstruction from PR #214 (issue #218).
# Needs root for unshare -n; run with sudo.
set -euo pipefail
cd "$(dirname "$0")"

BIN=${BIN:-./geist-bitnet}
CAST=${CAST:-take_offline.cast}

# Pre-warm outside the recording so the take starts instantly.
OMP_WAIT_POLICY=active "$BIN" "warm-up" 8 >/dev/null 2>&1

# Pick prompts the model verifiably gets right (see the PR #214 notes: the
# confidently-wrong radiator answer stayed on the cutting-room floor, and
# docs/PI5_BITNET.md#model-limits exists for exactly that reason).
asciinema rec --overwrite -c "unshare -n bash -c '
  ping -c 1 -W 2 1.1.1.1 || true;
  OMP_WAIT_POLICY=active $BIN \"The difference between baking soda and baking powder is\" 80;
  OMP_WAIT_POLICY=active $BIN \"At high altitude, water boils at a lower temperature because\" 80
'" "$CAST"

echo "wrote $CAST — render: agg $CAST ../../assets/demo-offline-box.gif"
