#!/bin/bash
# ab2.sh <cycles> <variant>... : interleaved, rotated order; one JSON line per run.
# variants: OFF (main, staging off), MAIN (main, 4 GiB default), ALL (main, every cold expert),
#           SAME (branch, profile cache after one request of the measured prompt),
#           CROSS (branch, profile cache after one request of docs/ARCHITECTURE.md),
#           NOPROF (branch, profile cache on but empty: the fresh start)
W=~/work-mpp; cd $W
M=${M:-Qwen3.6-35B-A3B-UD-Q4_K_XL}
P=$W/main/benchmark/results/raw/2026-10-23_qwen36_expert_usage_run1830.txt
[ "$M" = Qwen3-30B-A3B-Q4_K_M ] && P=$W/main/benchmark/results/raw/2026-10-15_moe_expert_usage_vulkan_run1225.txt
SEQ=${SEQ:-2726}
cyc=$1; shift
vs=("$@"); n=${#vs[@]}
for i in $(seq 90); do awk -v l=$(cut -d" " -f1 /proc/loadavg) 'BEGIN{exit !(l<3)}' && break; sleep 10; done
env GEIST_BENCH_BACKEND=vulkan GEIST_MOE_PROFILE=$P main/bin/linux/release/tests/bench_perf_sweep --gguf ~/geist-models/$M.gguf --seq-lens 512 --decode-n 8 --warmup 0 --repeats 1 --prompt-file prompt.md --m-max 512 >/dev/null 2>&1
echo "# start load $(cut -d' ' -f1-3 /proc/loadavg) $(date +%T) model $M"
for c in $(seq $cyc); do
  for k in $(seq 0 $((n-1))); do
    v=${vs[$(( (k + c) % n ))]}
    l=$(cut -d" " -f1 /proc/loadavg)
    log=$(mktemp); cd=$(mktemp -d)
    bin=main/bin/linux/release/tests/bench_perf_sweep; ex=""
    case $v in
      OFF) ex="GEIST_MOE_STAGE=0";;
      ALL) ex="GEIST_MOE_STAGE_HOST=65536";;
      SAME) bin=branch/bin/linux/release/tests/bench_perf_sweep; cp cache-same-$M/*.txt $cd/; ex="GEIST_MOE_PROFILE_CACHE=$cd";;
      CROSS) bin=branch/bin/linux/release/tests/bench_perf_sweep; cp cache-cross-$M/*.txt $cd/; ex="GEIST_MOE_PROFILE_CACHE=$cd";;
      NOPROF) bin=branch/bin/linux/release/tests/bench_perf_sweep; ex="GEIST_MOE_PROFILE_CACHE=$cd";;
    esac
    r=$(env $ex GEIST_VERBOSE=1 GEIST_MOE_PROFILE=$P GEIST_BENCH_BACKEND=vulkan $bin --gguf ~/geist-models/$M.gguf --seq-lens $SEQ --decode-n 128 --warmup 8 --repeats 1 --prompt-file prompt.md --m-max 512 2>$log | tail -1)
    th=$(grep -oE "threshold [0-9]+" $log | head -1); st=$(grep -oE "[0-9]+ cold experts staged on vulkan for [0-9]+ rows" $log | head -1)
    hot=$(grep -oE "[0-9]+ of [0-9]+ experts \([0-9]+ MiB\) on [a-z]+; profile [^,]*, cover [0-9.]+" $log | head -1 | sed 's|profile [^,]*/|profile |')
    hc=$(grep -oE "staging on: [0-9]+ cold experts held in host memory \([0-9]+ MiB\)[^;]*; chosen by .*" $log | head -1)
    echo "{\"cycle\":$c,\"v\":\"$v\",\"load\":$l,\"th\":\"$th\",\"staged\":\"$st\",\"hot\":\"$hot\",\"host\":\"$hc\",\"r\":$r}"
    rm -rf $log $cd
  done
done
echo "# end load $(cut -d' ' -f1-3 /proc/loadavg) $(date +%T)"
