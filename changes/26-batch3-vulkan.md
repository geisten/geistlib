### Changed
- **Vulkan: batched decode runs every session's DeltaNet step in four
  dispatches (#26, batch 3).**
  - Previously it took four dispatches per session. The fused slot is
    `deltanet_mix_states`.
  - `dn_conv_b` and `dn_scan_b` bind each session's own conv and recurrent
    state. The q/k l2norm and the gated RMSNorm take all n rows in one
    dispatch.
  - The outputs are bit-identical to one `deltanet_mix` call per session
    (tested on an RTX 2080 Ti).
  - Bonsai 2 27B, RTX 2080 Ti, 8 sessions: `decode_batch` steps are 4.1 %
    faster (median of 12 interleaved cycles, faster in all 12, 95 % interval
    [-5.0 %, -3.8 %]).
  - A dispatch can now bind up to 14 storage buffers, where the device
    allows; on a device that allows fewer, the per-session path stays.
