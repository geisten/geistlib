# Self-hosted runner (vulkan-gpu leg)

The `vulkan-gpu` job in `.github/workflows/ci.yml` needs a real Vulkan device,
which no hosted GitHub runner has. It runs on one machine: a Linux desktop with
a discrete GPU, registered as a self-hosted runner on this repo.

Everything else in CI stays on hosted runners. If this runner is offline, only
that job queues — the rest of the PR gates are unaffected.

## Setup

Get the current package URL and a registration token from
`https://github.com/geisten/geistlib/settings/actions/runners/new`, then:

```sh
mkdir -p ~/actions-runner && cd ~/actions-runner
curl -o r.tar.gz -L <url from that page>
tar xzf r.tar.gz
./config.sh --url https://github.com/geisten/geistlib --token <TOKEN> \
  --labels self-hosted,linux,x64,geist-vulkan
sudo ./svc.sh install "$USER"   # systemd unit, starts on boot
sudo ./svc.sh start
```

`runs-on: [self-hosted, linux, x64, geist-vulkan]` matches those labels; the
contract is fixed in `docs/CI_COVERAGE.md`. The runner polls GitHub outbound;
no inbound port, no firewall change.

Host prerequisites, installed once by hand (the job has no apt step):
`gcc >= 14`, `make`, the Vulkan loader and headers (`libvulkan-dev`),
`vulkan-tools` for `vulkaninfo`, and a working vendor driver — verify with
`vulkaninfo --summary` before blaming CI.

## Model e2e (GGUF)

The GPU leg runs `test_known_answer_e2e` (five cloze prompts, argmax, floor
4/5) and `test_prefill_determinism_int` on the physical device. One repo
setting drives it, under Settings -> Secrets and variables -> Actions ->
Variables:

| variable | value |
| :-- | :-- |
| `GEIST_GGUF_PATH` | absolute path to the GGUF **on the runner host**, e.g. `/home/<user>/models/gemma/gemma-4-E2B-it-Q4_K_M.gguf` |

The path must lie outside the runner's workspace: `actions/checkout` runs
`git clean -ffdx`, which deletes `gguf_artifacts/` — it is gitignored. A
symlink farm inside the repo does not survive a job. The step sets
`GEIST_STRICT_FIXTURES=gguf`, so an unset or wrong path fails the job instead
of skipping green.

Model provisioning is manual and one-time: put the GGUF anywhere readable by
the runner user and point the variable at it. Nothing is downloaded per job —
3 GB per PR is not worth the bandwidth on a machine that already has the file.

### Why this step is worth its runtime

It is the gate that caught the bug the kernel tests could not see. All three
Vulkan kernel tests passed on the device while every prompt produced nothing:
`scratch_ones_headdim_max` was allocated outside the scratch pool, Vulkan's
fused `attn_qkv_prep` requires q/k/v and the norm weights in one buffer and
declined with `GEIST_E_UNSUPPORTED`, and `layer_attn.c` treats a decline from
a plan-bound fused stage as a hard error — so prefill failed outright. Kernel
parity says nothing about whether the assembled forward pass runs.

Local repro of the whole leg on the runner host:

```sh
make clean   # BACKENDS changes are not tracked by make (mk/target-linux.mk)
make TARGET=linux BACKENDS="vulkan cpu_x86 cpu_scalar" GEMM_PROVIDER=native bin
GEIST_GGUF_PATH=~/models/gemma/gemma-4-E2B-it-Q4_K_M.gguf \
  GEIST_BACKEND=vulkan bin/linux/release/tests/test_known_answer_e2e
```

`GEIST_BACKEND` overrides the `auto` pick for every caller (`src/engine/backend.c`),
which is how a GPU build gets exercised end to end without a code change.

## Operational notes

- The machine must be powered on. GitHub cannot wake it; queued jobs wait, and
  the workflow run is cancelled after 24 h.
- The workspace is not sticky: `actions/checkout` cleans tracked *and* ignored
  files, so `build/` starts empty every job and the BACKENDS-staleness trap
  (`make clean` when switching backends) cannot bite in CI. What persists is
  the host: toolchain, driver, and the GGUF outside the workspace.
- Diagnostics (`vulkaninfo`, `nvidia-smi`, test output) upload as the
  `vulkan-gpu-diagnostics` artifact on every run, pass or fail: a driver update
  landing under us is the usual cause of a sudden red.
- No ccache on this leg, unlike the hosted ones — a full build on the desktop
  CPU is cheap enough. Add it if the 30 min timeout ever gets close.

Status of the hosted legs and what each one covers: `docs/CI_COVERAGE.md`.
