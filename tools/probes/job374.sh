#!/bin/sh
# job374.sh <out dir> — #374 bench scratch (not for main): modes parity,
# test and bench of moe-cpu-374.yml, builds included, inside the caller's
# one with-desktop-lock hold. Env: PROBE_MODE, BASE, NEW, MODELS, TESTS,
# GPTOSS / DEEPSEEK / QWEN3MOE, LLAMA (bench), RUNNER_TEMP.
set -u
O=$1
rc=0
for v in base new; do
    if [ "$v" = base ]; then R=$BASE; [ "$PROBE_MODE" = parity ] || continue; else R=$NEW; fi
    git worktree add --detach "$RUNNER_TEMP/w-$v" "$R" > /dev/null 2>&1
    git -C "$RUNNER_TEMP/w-$v" log --oneline -1 | tee -a "$O/heads.txt"
    make -C "$RUNNER_TEMP/w-$v" TARGET=linux BACKENDS="cpu_x86 cpu_scalar" GEMM_PROVIDER=native -j16 bin > "$O/build-$v.log" 2>&1 || { tail -60 "$O/build-$v.log"; exit 1; }
done
case $PROBE_MODE in
test)
    B=$RUNNER_TEMP/w-new/bin/linux/release/tests
    grep -o 'avx512_vnni\|avx_vnni' /proc/cpuinfo | sort -u | tee -a "$O/report.txt"
    for t in $(echo "$TESTS" | tr ';' ' '); do
        for isa in native avx2 scalar; do
            echo "== $t GEIST_FORCE_ISA=$isa" | tee -a "$O/report.txt"
            if [ $isa = native ]; then
                (cd "$RUNNER_TEMP/w-new" && "$B/$t") > "$O/t.log" 2>&1 || rc=1
            else
                (cd "$RUNNER_TEMP/w-new" && GEIST_FORCE_ISA=$isa "$B/$t") > "$O/t.log" 2>&1 || rc=1
            fi
            tail -40 "$O/t.log" | tee -a "$O/report.txt"
        done
    done
    ;;
parity)
    FILES=""
    for m in $(echo "$MODELS" | tr ';' ' '); do
        case $m in gptoss) FILES="$FILES $GPTOSS" ;; deepseek) FILES="$FILES $DEEPSEEK" ;; qwen3moe) FILES="$FILES $QWEN3MOE" ;; esac
    done
    sh tools/probes/quiet_wait.sh | tee -a "$O/report.txt"
    # shellcheck disable=SC2086
    sh tools/probes/parity374b.sh "$RUNNER_TEMP/w-base/bin/linux/release" \
        "$RUNNER_TEMP/w-new/bin/linux/release" "$PWD/$O/parity" $FILES 2>&1 | tee -a "$O/report.txt"
    grep -h "backend=\|rror\|not finite" "$O"/parity/log_* | sort | uniq -c | tee -a "$O/report.txt"
    ;;
bench)
    B=$RUNNER_TEMP/w-new/bin/linux/release/tests/bench_perf_sweep
    for m in $(echo "$MODELS" | tr ';' ' '); do
        case $m in gptoss) F=$GPTOSS ;; deepseek) F=$DEEPSEEK ;; qwen3moe) F=$QWEN3MOE ;; *) continue ;; esac
        echo "== $m" | tee -a "$O/report.txt"
        sh tools/probes/quiet_wait.sh | tee -a "$O/report.txt"
        echo "-- geist pp512/tg128 (cpu_x86, 16 threads, OFFLOAD.md, 3 repeats)" | tee -a "$O/report.txt"
        GEIST_BENCH_BACKEND=cpu_x86 $B --gguf "$F" --seq-lens 512 --decode-n 128 --warmup 1 --repeats 3 --prompt-file docs/OFFLOAD.md 2>/dev/null | tee -a "$O/report.txt"
        echo "-- llama-bench" | tee -a "$O/report.txt"
        "$LLAMA" -m "$F" -p 512 -n 128 -r 3 -t 16 -o jsonl 2>/dev/null | grep -o '"n_prompt": [0-9]*, "n_gen": [0-9]*.*"avg_ts": [0-9.]*' | tee -a "$O/report.txt"
    done
    ;;
esac
exit $rc
