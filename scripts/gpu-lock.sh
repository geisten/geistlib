# gpu-lock.sh — source at the top of a CI step that runs on the shared GPU
# (#706). The self-hosted machine with the RTX 2080 Ti serves GPU jobs of
# several repositories (geistlib, geist-runtime), each with its own runner;
# GitHub's concurrency groups do not reach across them. This takes the same
# machine-wide lock geist-runtime's scripts/test-gpu.sh takes, so the jobs
# run one after another instead of sharing the card (spurious out-of-device-
# memory failures, distorted timings). The lock is fd 9 of the sourcing
# shell: it is held until that step's shell exits. Builds stay outside it.
if command -v flock >/dev/null; then
    lock=${GEIST_GPU_LOCK:-/tmp/geist-gpu.lock}
    exec 9>"$lock"
    echo "gpu: waiting for $lock"
    flock -w 3600 9 || { echo "gpu: $lock still held after an hour" >&2; exit 1; }
fi
echo "gpu: $(date -u +%FT%TZ) · load $(uptime | sed 's/.*average[s]*: //')"
command -v nvidia-smi >/dev/null &&
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
