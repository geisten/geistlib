### Changed
- **Greedy decode is the argmax of the full output head again: the
  speculative i8-sketch head is opt-in (`GEIST_SPEC_HEAD=1`) (#384).** The
  sketch head scores only the 1024 rows an int8 sketch of a quarter of the
  hidden dimensions ranks highest, and nothing guarantees the true argmax is
  among them. On Llama 3.2 3B (Q6_K tied head) it emitted " The" where the
  full head and llama.cpp pick " Casual" (0.6-0.9 logits apart); over 1024
  greedy steps the true argmax needed 2 finalists at the median, 375 at p99,
  and 1070 and 2108 at the two steps it missed. Rows left out reached logit
  20.1 next to a winner of 21.2, so no check on the finalists tells a safe
  step from a miss, and a bound that proves one declines on nearly every
  token (6a005cf6). Off by default, the sketch table (~98 MB on Llama 3.2 3B,
  ~82 MB on BitNet 2B-4T) is no longer built at load either. Decode on
  models with an eligible tied head (BitNet 2B-4T F16, Llama 3.2 3B Q6_K,
  Q8_0 heads of width >= 2560) runs the dense head and is slower by what the
  sketch saved; `GEIST_SPEC_HEAD=1` restores the old behaviour.
  `test_spec_head_greedy_unit` checks every greedy token against the full
  logits' argmax on a fixture the spec head accepts.
