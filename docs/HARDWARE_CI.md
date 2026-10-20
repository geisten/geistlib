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
| `bench-matrix` | per column (`geist-avx512`, `geist-vulkan`, `macos-15-heavy`, `pi5`) | nightly (Sundays: every model) + dispatch: the cross-engine matrix, published to [geisten.net/bench](https://geisten.net/bench/) |

### Benchmark matrix

`benchmark/bench_matrix.json` lists which protocol model runs on which host
column and how often; `tools/bench_matrix.py plan` turns it into one job per
cell. Each job fetches its model into the host cache (`GEIST_MODEL_CACHE`,
default `~/geist-models`) with `tools/fetch_bench_model.py` — the URL comes
from `benchmark/model_sources.json`, the SHA-256 from the protocol — builds
the pinned llama.cpp (`scripts/ensure-llama-bench.sh`) and runs
`tools/bench_cross_engine.py` inside the host lock: `/tmp/geist-bench.lock`
on the desktop (the docker CI runner's health check pauses it while held),
`with-pi-lock` on the Pi, `~/bin/with-mac-lock` on the Mac (`/tmp/mac-heavy.lock`; the
`mac-m1max` runner (`macos-15-heavy`) holds it for every job and takes no
job while an agent holds it — see "Mac runners" and "Host locks"). The Apple profiles gate on idle CPU (>= 80 %
over the settle window) instead of the load average, which macOS on a shared
desktop never brings near 0.2/core, and measure a run again when its repeats
spread by more than 5 % (`host_gate` in the protocol's host profile).
Runs go to `https://geisten.net/bench/runs/<column>/<model>/` (org secret
`BENCH_SSH_KEY`, rrsync-restricted to that directory); the publish job
rebuilds `index.html`/`history.json` and fails when geist slowed by more than
5 % against the cell's previous run while its ratio to llama.cpp fell too.

## Runners

One host-mode `gitea-runner` (v5, `capacity: 1`) per machine, registered on
the `geisten` organisation (geistlib and geist-runtime share them), so jobs on
one machine never overlap. The CI runners in docker mode are separate
(`ubuntu-24.04`, `ubuntu-24.04-tsan`, `ubuntu-24.04-arm`); the Mac's two
runners are described below:

| machine | name | labels | install |
| :-- | :-- | :-- | :-- |
| Pi 5 (`rspdevelop`) | `pi5-rspdevelop` | `pi5:host` | `~/gitea-runner`, system unit `gitea-runner.service` |
| desktop | `geisten-desktop-gr` | `geist-vulkan:host`, `geist-avx512:host` | `~/gitea-runner`, user unit `gitea-runner.service` (linger) |

`config.yaml` puts a node24 first on `PATH` (JavaScript actions such as
`upload-artifact` need it) and, on the Pi, sets
`hooks.job_started`/`job_completed` to `~/bin/runner-job-started.sh` /
`runner-job-completed.sh`: they hold the board lock for the whole job, so
these jobs also wait for other runners on the board.
Heavy steps still go through `~/bin/with-pi-lock`.

### Host locks: the lease broker

Since 2026-10-20 `with-mac-lock` and `with-pi-lock` queue through a small
broker per host (`hostlease.py`, geisten-infra role `host_lease`; Mac:
LaunchAgent `net.geisten.hostlease`, status on `127.0.0.1:7381`; Pi: system
unit `hostlease`, `127.0.0.1:7382`). Commands, `WAIT=<s>` (exit 1 on
timeout), the `WAIT=0 … true` probe, `status` and the exit codes are the
ones the plain `lockf`/`flock` wrappers had. What it adds:

- one lease at a time, first come first served within a priority:
  `LOCK_PRIO=ci` (runner hooks) > `interactive` (default) > `batch` (long
  research, campaigns); a waiter rises one level per 30 min waited;
- `LOCK_MAX=<minutes>` (default 60, batch 240, cap 240): past it `status`
  marks the lease OVERTIME, 15 min later EXPIRED. Nothing is killed and the
  lease stays exclusive; only `LOCK_KILL=1` lets the wrapper stop its own
  command. `LOCK_NOTE=<text>` shows in `status -v`;
- a lease ends with its client process, so a crash never leaves the host
  locked; a broker restart hands the active lease to its client, waiters
  re-queue.

The broker holds `/tmp/mac-heavy.lock` / `/tmp/pi-heavy.lock` for the active
lease and waits while anything else holds that file, so plain `flock` users
still exclude — notably the `pi5-ci` docker jobs, which flock the
bind-mounted file from inside their container, and its `flock -n` health
check. Without the broker the wrappers fall back to the plain lock
(`with-mac-lock.lockf`, `with-pi-lock.flock`). The broker knows nothing
about temperature or load: the Pi's thermal pre-check (< 60 °C, load < 0.5)
and the Mac's quiet gates stay with the callers.

### Mac runners (M1 Max, host mode)

The Mac is shared with agents and the human, who take `~/bin/with-mac-lock`
(`/tmp/mac-heavy.lock`) for every benchmark. It has two org runners, so a
build never waits behind a benchmark and CI never holds the lock for a
compile:

| name | label | lock | install |
| :-- | :-- | :-- | :-- |
| `mac-light` | `macos-15:host` | none; daemon at `nice 10` | `~/gitea-runner-mac-light`, LaunchAgent `net.geisten.gitea-runner-mac-light` (`Nice` 10) |
| `mac-m1max` | `macos-15-heavy:host` | held for the whole job | `~/gitea-runner-geisten`, LaunchAgent `net.geisten.gitea-runner-geisten` |

`mac-m1max`'s hooks (`~/bin/mac-runner-job-started.sh` /
`mac-runner-job-completed.sh`) hold a `ci` lease while a job runs. Its
health check (`~/bin/mac-runner-free.sh`) fetches a job only when a ci lease
would be granted within 10 min (free, or the holder declared a `LOCK_MAX`
that ends by then); while busy, it and the completed hook report waiting
`macos-15-heavy` jobs, and the broker keeps the next hand-over free for CI
(up to 180 s) instead of giving it to the next agent. `mac-light` has neither. Each has its own `cache.dir` and
`host.workdir_parent` (host mode clones actions there), so the two never
race on one action checkout. Both run with `capacity: 1` and
`fetch_timeout: 30s`.

Which label a job takes:

- `macos-15-heavy` — the job loads a real model or its result is a timing:
  the `macOS arm64` leg of CI (Metal parity, qwen3.5 integration, perf
  gate; `scripts/gitea-ci.py` sets it, GitHub's copy keeps `macos-15`) and
  the Mac columns of `bench-matrix` (`benchmark/bench_matrix.json`).
- `macos-15` — build and model-free tests: CI's `macOS arm64 GCD (no
  OpenMP)` leg and the release build (`release.yml` `macos-arm64`). These
  may run while an agent measures; nice 10 keeps them behind it, not out.

A step that must not overlap a measurement on a `macos-15` job wraps itself
in `~/bin/with-mac-lock`; better, it moves to `macos-15-heavy`.

Gitea's own job timeout (`[actions] ENDLESS_TASK_TIMEOUT`, 3 h by default)
caps every job, whatever `timeout-minutes` says — `cross-engine-campaign`
needs it raised in `app.ini`.

### Cloud arm64 runner

`ubuntu-24.04-arm` has two docker-mode runners: `pi5-ci` on the Pi (capacity
1, under the board lock) and, since 2026-10-20, `cloud-arm64-hel1`, a
Hetzner CAX21 in Helsinki (Neoverse-N1, 4 vCPU, 7.5 GB + 4 GB swap for the
ASan builds, 75 GB disk, Ubuntu 26.04, host name `ci-arm64-hel1`). Gitea
hands a queued arm64 job to whichever is free. Nothing measured runs there:
it is a shared cloud VM, and the Pi 5 stays the arbiter for timings.

| item | value |
| :-- | :-- |
| runner | `gitea-runner` v5.0.0, system unit `gitea-runner-ci`, user `runner` (docker group) |
| config | `/home/runner/gitea-runner-ci/config.yaml` |
| jobs | `capacity: 2`, `OMP_NUM_THREADS=2`, `timeout: 3h`, `fetch_timeout: 30s` |
| workspace | `bind_workdir` under `/home/runner/gitea-runner-ci/work` (jobs `docker run -v "$PWD:/src"`); idle cleanup after 12 h |
| hook | `job_completed` = `hooks/job-completed.sh`, mounted read-only into the job container, chowns the task dir back to `runner` (root-owned files otherwise pile up) |
| health check | no job below 20 GB free disk |
| actions cache | runner's own, `172.17.0.1:8088` (ufw admits only the docker subnets) |
| CI hygiene | system timer `ci-hygiene` (as `runner`), the desktop's script; log `/home/runner/ci-hygiene/ci-hygiene.log` |
| git cache | `git-cache.service` on `172.17.0.1:8417`, installed but off until a read-only token is in `/home/runner/.config/git-cache/token`; the runner's `envs` are not wired to it yet ("CI git cache") |

Access is key-only SSH as `germar` (no root login, ufw: 22 rate-limited
only); unattended-upgrades is on.

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

## CI git cache

Every CI job checks this repository out from git.geisten.net, a 1-vCPU VM.
A run of 19 jobs made it pack and send about 9 MB nineteen times; on busy
days (17 GB in 22 h on 2026-10-18) its network dropped packets and
checkouts hung until Gitea stopped them as zombie tasks. The desktop runner
host therefore keeps a read-only mirror per repository and serves the job
containers from it: `scripts/git-cache.py`, a user service `git-cache` on
`192.168.178.43:8417`. Before it answers a fetch it updates the mirror with
one incremental fetch from Gitea (token in `~/.config/git-cache/token`, from
the org secret `GIT_CACHE_TOKEN`, read-only); concurrent jobs share that
fetch. Only docker networks and localhost may connect, and it refuses
pushes.

The docker runners (`desktop-ci`, `desktop-tsan`) give their job containers

```yaml
envs:
  GIT_CONFIG_COUNT: "2"
  GIT_CONFIG_KEY_0: url.http://192.168.178.43:8417/.insteadOf
  GIT_CONFIG_VALUE_0: https://git.geisten.net/
  GIT_CONFIG_KEY_1: url.https://git.geisten.net/.pushInsteadOf
  GIT_CONFIG_VALUE_1: https://git.geisten.net/
```

so git fetches through the cache and pushes still go to Gitea. Install,
update, roll back or inspect it with the `git-cache` workflow
(`-i action=deploy|rollback|status`); `deploy` proves a job checkout went
through the cache or restores the runners' configs. If the cache is down,
checkouts on those runners fail: roll back, or `systemctl --user restart
git-cache` on the desktop.

Each runner host has its own cache, listening where its jobs connect:

| host | service | listens on | runners wired |
| :-- | :-- | :-- | :-- |
| desktop | user unit `git-cache` | `192.168.178.43:8417` | `desktop-ci`, `desktop-tsan` |
| Pi 5 | system unit `git-cache` (as `germar`) | `172.17.0.1:8417` (docker bridge), `127.0.0.1:8417` | `pi5-ci` → bridge, `pi5-rspdevelop` (host mode) → 127.0.0.1 |
| Mac | LaunchAgent `net.geisten.git-cache` | `127.0.0.1:8417` | `mac-light`, `mac-m1max` (host mode) |
| cloud-arm64-hel1 | system unit `git-cache` | `172.17.0.1:8417` | not yet (no token) |

The Pi takes the same workflow with `-i host=pi5`. Its deploy job runs on
`pi5-rspdevelop` and holds the board lock, so `pi5-ci` (which takes no job
while the lock is held) restarts at once; `pi5-rspdevelop` itself restarts
from a transient unit once the job has ended and the lock is free.
`-i host=pi5 -i action=verify` then runs one checkout on each Pi
runner and looks for both in the cache's log (`ubuntu-24.04-arm` is shared
with the cloud runner: if the container probe lands there, dispatch again).

The Mac is installed by hand, `sh scripts/git-cache-mac.sh install`
(`status` shows its state). Its token is not the org secret: the user
places a read-only token in `~/.config/git-cache/token` (mode 0600), and
launchd starts the cache once that file exists. The host-mode runners do
not point git at the cache directly; their `envs` include a file

```yaml
  GIT_CONFIG_COUNT: "1"
  GIT_CONFIG_KEY_0: include.path
  GIT_CONFIG_VALUE_0: /Users/germar/git-cache/ci.gitconfig
```

that the cache writes once it serves geistlib and removes when it stops. A
missing include file is ignored, so without the token, or with the cache
down, Mac jobs fetch from Gitea as before.

## CI hygiene

A user timer `ci-hygiene` on the desktop (every 10 min,
`scripts/ci-hygiene.sh run`, units `scripts/ci-hygiene.{service,timer}`,
log `~/ci-hygiene/ci-hygiene.log`) keeps two recurring failures away:

- **Stale action-cache lock.** A checkout of an action repository that died
  mid-way leaves `~/.cache/act*/<hash>/.git/index.lock`, and every later job
  needing that action fails within seconds at "Prepare all required
  actions". The timer removes such a lock once it is older than 10 min and
  no `git` process works in that repository (cwd or arguments), and logs
  each removal.
- **Docker disk.** Once a day `docker system prune -f --filter until=72h`
  and `docker image prune -f --filter until=168h`. Without `-a` both remove
  only stopped containers, unused networks, build cache and *dangling*
  images, never a tagged image such as the runner images
  (`docker.gitea.com/runner-images:*`); no `--volumes`. The script compares
  the tagged image list before and after and logs any loss.

Install or update it with the `ci-hygiene` workflow (`-i action=deploy`),
inspect it with `-i action=status` (timer, last log lines, any lock left in
the action caches).

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
