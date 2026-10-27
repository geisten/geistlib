### Removed
- **The Metal subtractive profiler (`GEIST_SKIP_<CAT>`, `GEIST_SKIP_H`) is
  gone.** With `GEIST_METAL_PROFILE=1` these variables dropped a whole op
  category's dispatches so the wait-time delta showed that category's GPU
  time. Every dispatch paid a flag test for it, single skips were known to
  mislead (overlapping GEMM and DeltaNet chains, repeats > 1), and the
  per-stage dispatch counts and command-buffer GPU times of
  `GEIST_METAL_PROFILE=1` remain. Use Xcode's GPU capture
  (`GEIST_METAL_CAPTURE`) for per-kernel time.
