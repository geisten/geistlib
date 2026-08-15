#!/usr/bin/env bash
# take_writer.sh — record assets/demo-writing.gif: a draft email piped
# through batch mode (`-` reads prompts from stdin), continuation streams
# out, nothing leaves the machine. Reconstruction from PR #214 (issue #218).
set -euo pipefail
cd "$(dirname "$0")"

BIN=${BIN:-./geist-bitnet}
CAST=${CAST:-take_writer.cast}

OMP_WAIT_POLICY=active "$BIN" "warm-up" 8 >/dev/null 2>&1

DRAFT='Hi Dana, thanks for the quick turnaround on the review. I went through your comments and'

asciinema rec --overwrite -c "
  echo \"\$ cat draft.txt\";
  echo \"$DRAFT\";
  echo;
  echo \"\$ cat draft.txt | ./geist-bitnet - 90\";
  echo \"$DRAFT\" | OMP_WAIT_POLICY=active $BIN - 90
" "$CAST"

echo "wrote $CAST — render: agg $CAST ../../assets/demo-writing.gif"
