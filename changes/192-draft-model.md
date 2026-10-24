### Added
- **Speculative decoding with a small draft model (#192, EXPERIMENTAL).**
  `geist_session_set_draft(target, draft)` makes
  `geist_session_decode_speculative` take its guesses from a session of a
  small model with the same vocabulary, e.g. Qwen3.5 0.8B for Bonsai 2 27B.
  - The engine keeps the draft session in step with the `history` it
    receives and checkpoints after every draft step. After the verify it
    returns the draft to the accepted tokens.
  - Two building blocks are public as well: `geist_session_verify_drafts`
    verifies caller-supplied drafts, and `geist_session_checkpoint` lets
    `geist_session_truncate` return to one of the 8 newest checkpoints, on
    DeltaNet layers too.
  - Bonsai 2 27B + Qwen3.5 0.8B, greedy, 3 thinking prompts × 128 tokens,
    `k_max` 7, token-identical to plain decode:

    | | plain decode | speculative |
    |---|--:|--:|
    | M1 Max (Metal) | 18.7 tok/s | 24.8 tok/s |
    | RTX 2080 Ti (Vulkan) | 38.6 tok/s | 40.7 tok/s |
