### Changed
- **Batched decode runs the attention layers' per-session steps on the
  batch's own rows (#26, batch 3).**
  - Each session's RoPE, KV append and attention now read and write its row
    of the batch in place. The four row copies into and out of its scratch
    per layer are gone, and q is scaled once for all rows.
  - The ops and values are the ones the copying path used, so the outputs
    are bit-identical (tested on Metal and Vulkan).
  - Bonsai 2 27B, 8 sessions:
    - RTX 2080 Ti (Vulkan): -1.7 % per `decode_batch` step (6 cycles, all
      faster, 95 % interval [-2.3 %, -1.6 %]).
    - M1 Max (Metal, F16 KV): -3.0 %, on a busy host (interval
      [-11.6 %, -0.6 %]).
  - `GEIST_DECODE_BATCH_ATTN_COPY=1` at load keeps the copying path (A/B).
