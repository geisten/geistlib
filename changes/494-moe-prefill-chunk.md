### Changed
- **MoE prefill on Metal runs 512-row chunks and a faster Q6_K dequant
  (#494).** A mixture-of-experts model on a GPU that runs the experts'
  grouped GEMMs on the device (Metal's `fused->moe_ffn_ids`) now prefills
  512 rows per chunk, as on discrete GPUs (#163): 256 rows filled an
  expert's 32-row tiles to ~65 % and read every expert's weights twice as
  often. The Q6_K dequant of the Metal GEMMs reads its quants as 16-bit
  words (bit-identical results), 6-9 % faster per Q6_K grouped GEMM on the
  Apple M6. Prefill of OLMoE, DeepSeek-V2-Lite, GigaChat3 and gpt-oss-20b
  at 512 and 2048 tokens gets 12-20 % faster on the M6. `GEIST_M_MAX`
  still overrides the chunk. New `tests/bench_metal_moe_mm` times the
  grouped GEMM per dtype on llama.cpp's MUL_MAT_ID shapes.
