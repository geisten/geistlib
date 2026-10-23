#!/bin/sh
# #171: prepare the 5-question dry-run bundle (development) and the campaign
# bundle (calibration + 500 stratified test questions). Frozen policy:
# cais/mmlu c30699e8, 5-shot, exclude, template date 2026-10-04.
set -e
W=${W:-/tmp/geistlib-171}
C=/Users/germar/workspace/geistlib-campaigns/171
PY=/opt/homebrew/bin/python3.14
MODEL="/Users/germar/Library/Application Support/geisten/models/Ternary-Bonsai-2-27B-PQ2_0.gguf"
SNAP=/Users/germar/.cache/huggingface/hub/datasets--cais--mmlu/snapshots/c30699e8356da336a370243923dbaf21066bb9fe
for which in "$@"; do
  case $which in
    dryrun) extra="--splits development --per-subject 1 --subject-limit 5";;
    calibration) extra="--splits calibration --per-subject 0";;
    test) extra="--splits test --per-subject 0 --test-sample 500";;
    *) echo "unknown $which"; exit 2;;
  esac
  [ -e "$C/cases-$which/metadata.json" ] && continue
  rm -rf "$C/cases-$which"
  (cd $W && $PY tools/prepare_decision_eval.py --snapshot "$SNAP" \
    --dataset-revision c30699e8356da336a370243923dbaf21066bb9fe \
    --model "$MODEL" --tokenizer-binary $W/bin/mac-omp/release/tools/eval_geist \
    --out-dir "$C/cases-$which" --overlap-policy exclude --shots 5 --rotations 1 \
    --template-date 2026-10-04 $extra) > "$C/prepare-$which.log" 2>&1
done
echo prepared
