### Fixed
- **The same prompt prefilled twice no longer gets other logits the second
  time (#591).** The tail-norm fusion (#17) records for which layer,
  buffer, position and row count a layer's residual step already wrote the
  next layer's normalised input, and cleared the layer to 0 after use, which
  is also layer 0's index. When a session's next pass had the rows and the
  position of the one before it, layer 0 skipped its attention norm and
  read the earlier pass's last normalised rows: a prompt of one chunk
  prefilled again after a reset, or a token fed at a position a decode step
  was truncated from. Gemma 4 E2B on the M1 Max, main: reset and prefill of
  the same 5 tokens twice gave two different logit vectors on Metal and on
  `cpu_neon`, as did 1 token and a whole chunk (256 on Metal, 64 on
  `cpu_neon`), and a 512-token prefill after a 256-token one differed from
  the same prefill on an empty session. Layer 0 now never takes its input
  as pre-normalised; the second pass has the first one's bits in each case.
  A first pass on a session, and any pass whose row count or position
  differs from the one before, computed what it computes now.
