# Hardware CI

The jobs that need real hardware — the Raspberry Pi 5 and the Ryzen 9 9950X /
RTX 2080 Ti desktop — are workflows of this repository on git.geisten.net
(`.gitea/workflows/`). The repository is private there, so no outside pull
request reaches these runners; the GitHub mirror runs pull-request checks on
GitHub's runners only.

Each workflow checks out this repository at the `ref` input (`main` by
default). A pull request imported from GitHub is tested by dispatching its
branch **after reading it**:

```sh
tea actions workflows dispatch --login git.geisten.net --repo geisten/geistlib \
  --ref main -i ref=github-pr/123 vulkan-gpu.yml
tea actions workflows dispatch --login git.geisten.net --repo geisten/geistlib \
  --ref main -i new=origin/<branch> pi5-ab.yml
```

| workflow | runner label | trigger |
| :-- | :-- | :-- |
| `pi5` | `pi5` | nightly + dispatch: build, unit, int, frozen `make bench` vs last green |
| `pi5-ab` | `pi5` | dispatch: A/B of two revisions |
| `pi5-decision` | `pi5` | dispatch: Gemma 4 decision fixtures |
| `native-avx512` | `geist-avx512` | nightly + dispatch |
| `vulkan-gpu` | `geist-vulkan` | nightly + dispatch: Vulkan suite + model e2e on the discrete GPU |
| `vulkan-ab` | `geist-vulkan` | dispatch: A/B on one Vulkan device or the Zen 5 CPU |
| `cross-engine-campaign` | `geist-vulkan` | dispatch: one cell of the cross-engine matrix |

## Runners

One host-mode `gitea-runner` (v5, `capacity: 1`) per machine, registered on
the `geisten` organisation (geistlib and geist-runtime share them), so jobs on
one machine never overlap. The CI runners in docker mode are separate
(`ubuntu-24.04`, `ubuntu-24.04-tsan`, `ubuntu-24.04-arm`, `macos-15`):

| machine | name | labels | install |
| :-- | :-- | :-- | :-- |
| Pi 5 (`rspdevelop`) | `pi5-rspdevelop` | `pi5:host` | `~/gitea-runner`, system unit `gitea-runner.service` |
| desktop | `geisten-desktop-gr` | `geist-vulkan:host`, `geist-avx512:host` | `~/gitea-runner`, user unit `gitea-runner.service` (linger) |

`config.yaml` puts a node24 first on `PATH` (JavaScript actions such as
`upload-artifact` need it) and, on the Pi, sets
`hooks.job_started`/`job_completed` to `~/bin/runner-job-started.sh` /
`runner-job-completed.sh`: they hold `/tmp/pi-heavy.lock` for the whole job,
so these jobs also wait for other runners on the board.
Heavy steps still go through `~/bin/with-pi-lock`.

Gitea's own job timeout (`[actions] ENDLESS_TASK_TIMEOUT`, 3 h by default)
caps every job, whatever `timeout-minutes` says — `cross-engine-campaign`
needs it raised in `app.ini`.

## Desktop host

Host prerequisites, installed once by hand (the job has no apt step):
`gcc >= 14`, `make`, the Vulkan loader and headers (`libvulkan-dev`),
`vulkan-tools` for `vulkaninfo`, and a working vendor driver — verify with
`vulkaninfo --summary` before blaming CI.

### The device the job actually runs on

A host may enumerate several Vulkan devices (discrete GPU, iGPU, llvmpipe).
`vk_pick_device` takes the first discrete one and falls back to device 0 when
there is none, which would let this leg pass on llvmpipe. The environment
step therefore fails the job if no `PHYSICAL_DEVICE_TYPE_DISCRETE_GPU` is
enumerated. `GEIST_VK_DEVICE=<index>`
pins a specific device when a host has more than one discrete GPU.

## Model e2e (GGUF)

`vulkan-gpu` runs `test_known_answer_e2e` (five cloze prompts, argmax, floor
4/5), `test_prefill_determinism_int` and `test_qwen35_vulkan_e2e_int` (qwen35
greedy continuations against `cpu_scalar`, token for token) on the physical
device. Two repo settings drive it, under this repo's Settings -> Actions -> Variables:

| variable | value |
| :-- | :-- |
| `GEIST_GGUF_PATH` | absolute path to the GGUF **on the runner host**, e.g. `/home/<user>/models/gemma/gemma-4-E2B-it-Q4_K_M.gguf` |
| `GEIST_QWEN35_GGUF_PATH` | the same for Qwen3.5 0.8B Q8_0 (`make fetch-qwen35-model`), e.g. `/home/<user>/models/qwen/qwen3.5-0.8b-q8_0.gguf` |

The path must lie outside the runner's workspace: `actions/checkout` runs
`git clean -ffdx`, which deletes `gguf_artifacts/` — it is gitignored. A
symlink farm inside the repo does not survive a job. The step sets
`GEIST_STRICT_FIXTURES=gguf`, so an unset or wrong path fails the job instead
of skipping green.

Model provisioning is manual and one-time: put the GGUF anywhere readable by
the runner user and point the variable at it. Nothing is downloaded per job —
3 GB per run is not worth the bandwidth on a machine that already has the file.

The whole leg (checkout, full build, three kernel tests, both e2e binaries)
takes about 42 s. The e2e step
exists because kernel parity says nothing about whether the assembled forward
pass runs.

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

- The machine must be powered on. Gitea cannot wake it; queued jobs wait, and
  the job is abandoned after Gitea's timeout.
- The workspace is not sticky: the checkout step runs `git clean -ffdx`, so `build/` starts empty every job and the BACKENDS-staleness trap
  (`make clean` when switching backends) cannot bite in CI. What persists is
  the host: toolchain, driver, and the GGUF outside the workspace.
- Diagnostics (`vulkaninfo`, `nvidia-smi`, test output) upload as the
  `vulkan-gpu-diagnostics` artifact on every run, pass or fail: a driver update
  landing under us is the usual cause of a sudden red.
- No ccache on this leg, unlike the hosted ones — a full build on the desktop
  CPU is cheap enough. Add it if the 30 min timeout ever gets close.

Hosted legs and what each covers: geistlib `docs/CI_COVERAGE.md`.
