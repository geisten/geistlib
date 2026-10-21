### Changed
- **CI runs in two tiers: seven fast checks before the merge, everything
  heavy after it.** A pull request now waits only for clang-format, the API
  contract, the version check, `changed paths` and build + unit tests on
  Linux x86_64 (musl, the release recipe), Linux arm64 and macOS arm64 —
  seven jobs, about 15 minutes, instead of 25. ASan/UBSan, TSan, coverage, the
  real-model suites, Vulkan on lavapipe, AVX-512 under SDE, the audio legs,
  the remaining build legs, the Metal gate and the perf floors run on every
  push to `main`, nightly at 03:47 UTC and on dispatch; the Metal and perf
  legs moved to dedicated hardware (macOS on `macos-15-heavy`, Linux arm64 on
  the Pi 5), since the shared cloud VM missed the prefill floor by noise
  (19.9 < 20 tok/s, run 1586). A job failing on `main` opens a Gitea issue
  per job name (`CI on main: <job>`, label `ci-failure`,
  `scripts/ci-issues.py`), later failures comment on it and the first green
  run closes it. A pull request labelled `ci:full` runs both tiers.
  `scripts/required_checks.py` derives the required list from the workflow
  (a job is post-merge when its `if:` tests `needs.changes.outputs.full`) and
  checks that the issue job waits for every other job.
