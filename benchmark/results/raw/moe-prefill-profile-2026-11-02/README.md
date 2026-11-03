# Staging host copies by the run-time profile's prefill rows (#239 follow-up)

RTX 2080 Ti + Ryzen 9 9950X, every run under `~/bin/with-desktop-lock` and
`/tmp/geist-gpu.lock`, started at load < 3 after a model warm-up (the load
during a run is the run's own). Request: the first 2,726 tokens of
`docs/OFFLOAD.md` as of main b358d73e (`prompt.md`), 128 decoded,
`--m-max 512`, GPU mode, the resident set from the expert profiles of runs
1830 / 1225 (`GEIST_MOE_PROFILE`). `sum2.py <file>` prints the tables.

Profiles: `mkprof.sh` runs one such request on the branch with
`GEIST_MOE_PROFILE_CACHE` set, which writes the v2 cache file when the
model goes: `same` from the measured prompt, `cross` from the first 2,726
tokens of `docs/ARCHITECTURE.md`. `ab2.sh` copies the file into a fresh
directory for every run, so every run loads the same profile.

| variant | binary | what |
|:--|:--|:--|
| OFF | main b358d73e | `GEIST_MOE_STAGE=0` |
| MAIN | main | the 4 GiB default, copies in decode-profile order |
| ALL | main | `GEIST_MOE_STAGE_HOST=65536`: every cold expert copied |
| SAME / CROSS | branch | 4 GiB, copies by the `same` / `cross` cache's prefill rows |
| NOPROF | branch | the cache on and empty: the fresh start |

| file | branch commit | what |
|:--|:--|:--|
| `ab36.jsonl` | 51ddb132 (first build: rows at the threshold, then decode counts) | Qwen3.6-35B-A3B, 3 cycles |
| `ab36b.jsonl`, `ab30b.jsonl` | 16012893 (rows at a quarter of the threshold break ties) | both models, 3 cycles |
| `ab30c.jsonl` | 16012893 | Qwen3-30B-A3B, 5 cycles (the table in docs/OFFLOAD.md) |
| `ppl2.jsonl` | 16012893 | `bench_nll`, main vs SAME, four windows per model |
| `pplcheck.txt` | same | the +0.91 % window against staging off, every cold expert, SAME twice |
| `pred.out` | - | `pred.py`: offline share of a prefill's staged rows the 4 GiB copies hold, by predictor (needs the cache files, not kept: ~400 KB each, `mkprof.sh` remakes them) |
