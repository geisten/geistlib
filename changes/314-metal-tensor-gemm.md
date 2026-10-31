### Changed
- **Metal: prefill GEMMs on the GPU's matrix units on M5/M6-class Macs,
  2.0-2.4x faster prefill there (#314), greedy output unchanged.** On Apple
  GPU family 10+ the quantized prefill GEMMs (every Metal quant format)
  run through the Metal 4 tensor API (MetalPerformancePrimitives
  `matmul2d`), as llama.cpp does on those chips; that was the whole
  55-63 % prefill gap to llama.cpp on the M6. Apple M6: Gemma 4 E2B Q4_K_M
  pp512 1034 -> 2385 tok/s, Bonsai 2 27B PQ2_0 96 -> 201 tok/s.
  `GEIST_METAL_TENSOR=0` keeps the simdgroup GEMMs, `=1` forces the tensor
  path on older GPUs (correct, but slower there). M1-M4 are unchanged.
