### Changed
- **Batched decode runs every session's DeltaNet step in one Metal call (#26,
  batch 3).**
  - A new optional fused slot `deltanet_mix_states` steps n sessions' decode
    rows. It uses two dispatches (`dn_dec_qk_b`, `dn_dec_v_b`) in place of 2n,
    and reads and advances each session's own conv and recurrent state.
  - The math and its order are the same as one `deltanet_mix` call per
    session, and the outputs are bit-identical (tested).
  - Bonsai 2 27B, M1 Max, 8 sessions: `decode_batch` steps are 4.3 % faster
    (median of 12 interleaved cycles, faster in 11, 95 % interval
    [-7.1 %, -0.6 %]).
  - `GEIST_DECODE_BATCH_PER_SESSION=1` at load keeps one call per session.
    Vulkan still uses one call per session.
