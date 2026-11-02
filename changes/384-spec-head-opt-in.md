### Changed
- **Greedy decode is the argmax of the full output head again, except on
  BitNet: the model family decides whether the speculative i8-sketch head
  is on by default (#384).** The sketch head computes exact logits only for
  the 1024 rows that an int8 sketch of a quarter of the hidden dimensions
  ranks highest, and nothing guarantees that the true argmax is among
  them. On Llama 3.2 3B (Q6_K tied head) it emitted " The" where the full
  head and llama.cpp pick " Casual". Over 1024 greedy steps, 2 tokens were
  not the argmax: the true argmax had sketch rank 1070 and 2108. Rows left
  out reached logit 20.1 next to a winner of 21.2, so no check on the
  finalists can tell a safe step from a miss. A bound that proves one
  declines on nearly every token (6a005cf6).
  `transformer_family.spec_head_default` now holds the default. It is on
  for the BitNet families, whose tied F16 head is ~50 % of a decode step
  (Pi 5: +82 % decode). There, 3 of 1024 greedy steps on BitNet 2B-4T
  (~0.3 %) emit a token that is not the full head's argmax. It is off for
  every other family. That costs Llama 3.2 3B ~8 % decode on Apple CPU
  (Mac mini M6: +8.7 % / +8.3 % ms at 128 / 512 tokens) and Gemma 4 E4B
  ~0.4 %, and these models no longer build the ~98 MB sketch table at
  load. `GEIST_SPEC_HEAD=0/1` overrides the default in either direction.
  `test_spec_head_greedy_unit` checks every greedy token against the full
  logits' argmax on a llama fixture the spec head accepts, and checks that
  BitNet keeps the default.
