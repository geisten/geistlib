#!/bin/sh
# vk504_parity.sh [CHUNKS] — perplexity of GigaChat3 and DeepSeek-V2-Lite in
# both MLA layouts (#504): geist Vulkan (default F16 cache; FP32 cache on the
# absorbed files), geist cpu_x86 (reference) and llama.cpp Vulkan, on the
# same llama-tokenize ids (tools/ppl_vs_llama.py, ctx 512, docs/OFFLOAD.md @
# b98a5c4b as PR #500 used). $REVS: label=rev, the first is the one
# measured on every model; later ones run the decompressed files only
# (main refuses the absorbed ones).
. tools/probes/vk504_common.sh
CHUNKS=${1:-8}
build_revs
llama_tools
TEXT=$OUT/offload.txt
git show b98a5c4b:docs/OFFLOAD.md >"$TEXT"
FIRST=${LABELS%% *}; FIRST=$(echo $LABELS | cut -d' ' -f1)
EV() { echo "$RUNNER_TEMP/wt-$1/bin/linux/release/tools/eval_geist"; }
. "$ROOT/scripts/gpu-lock.sh"
uptime
row() { # model label backend env...
    model=$1; lab=$2; be=$3; shift 3
    name="$(basename "$model" .gguf)_${lab}_${be}$(echo "$*" | tr ' =' '__')"
    env GEIST_BACKEND="$be" "$@" python3 "$ROOT/tools/ppl_vs_llama.py" "$model" "$TEXT" \
        --llama-bin "$LBIN" --eval-geist "$(EV "$lab")" --chunks "$CHUNKS" --skip-llama "$LPPL" \
        >"$OUT/$name.json" 2>"$OUT/$name.err"
    echo "| $(basename "$model") | geist $lab $be $* | $(python3 -c "import json,sys; d=json.load(open('$OUT/$name.json')); print(d['geist_ppl'], '|', f\"{d['rel_diff']*100:+.2f} %\", '|', d['tokenizer_agrees'])" 2>/dev/null || echo "FAILED: $(tail -3 "$OUT/$name.err" | tr '\n' ' ')") |" | tee -a "$OUT/summary.md"
}
echo "| model | engine | ppl | vs llama.cpp Vulkan | tokens agree |" >"$OUT/summary.md"
echo "|---|---|---|---|---|" >>"$OUT/summary.md"
for model in "$GIGA" "$V2_S" "$GIGA_C" "$V2"; do
    case $model in *V2-Lite*) ngl="-ngl 99 --n-cpu-moe 4" ;; *) ngl="-ngl 99" ;; esac
    # shellcheck disable=SC2086
    GGML_VK_VISIBLE_DEVICES=0 "$LBIN/llama-perplexity" -m "$model" -f "$TEXT" -c 512 --chunks "$CHUNKS" $ngl \
        >"$OUT/llama_$(basename "$model" .gguf).log" 2>&1
    LPPL=$(grep -o 'Final estimate: PPL = [0-9.]*' "$OUT/llama_$(basename "$model" .gguf).log" | grep -o '[0-9.]*$')
    echo "| $(basename "$model") | llama.cpp Vulkan ($ngl) | $LPPL | | |" | tee -a "$OUT/summary.md"
    [ -n "$LPPL" ] || LPPL=1
    row "$model" "$FIRST" vulkan
    case $model in "$GIGA"|"$V2_S") row "$model" "$FIRST" vulkan GEIST_KV_F16=0 ;; esac
    row "$model" "$FIRST" cpu_x86
    case $model in "$GIGA_C"|"$V2") for lab in $LABELS; do [ "$lab" = "$FIRST" ] || row "$model" "$lab" vulkan; done ;; esac
done
cat "$OUT/summary.md"
