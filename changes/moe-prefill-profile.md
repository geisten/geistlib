### Changed
- **The staging host copies of a mixture of experts follow the run-time
  profile's prefill rows (#239 follow-up).** The run-time expert profile
  (#165) also counts each prefill chunk's rows per expert, in power-of-two
  buckets, and keeps them and the measured staging threshold in the
  profile cache (file format v2; a v1 file still loads). With them the
  4 GiB of pinned host copies go to the cold experts with the most rows at
  the threshold per byte, ties by the rows at a quarter of it, instead of
  the decode profile's order; the resident experts do not change. With the
  cache filled by one request (`GEIST_MOE_PROFILE_CACHE`), the 2,726-token
  request on an RTX 2080 Ti + Ryzen 9 9950X prefills Qwen3.6-35B-A3B at
  926 tok/s (916 with a profile of another text) against 850 on main, the
  speed of every cold expert copied (921); Qwen3-30B-A3B at 840 / 834
  against 820 (all: 828). Decode is unchanged, perplexity within the
  spread staging already has (+0.22 / +0.06 % over four windows); without
  prefill rows (no cache, a fresh one) the copies are the same as before.
  docs/OFFLOAD.md, "Host copies by the prefill profile".
