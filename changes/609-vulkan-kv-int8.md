### Changed
- **Vulkan: the INT8 and INT4 KV caches are read and written on the device
  (#609).** Vulkan had no kernel for them: every layer mapped its K and V
  rows to quantize them on the host and mapped the query to attend there, a
  submit and a wait each. One compute shader is now
  `fused->attention_kv_int8` and `attention_kv_int4` (the host kernel's
  arithmetic: the query head quantized at amax / 127, exact integer dots,
  V weighted by weight times scale), a second one the quantizing append,
  rotation included; the rotation of the query and of the output runs
  through the backend's Hadamard kernel. Gemma 4 E2B Q4_K_M on an RTX 2080
  Ti, two runs each, the first token the same in every row:

  | KV cache | prefill, 512 tokens | decode at 512 | prefill, 1536 | decode at 1536 |
  |---|---|---|---|---|
  | F16 | 37-49 ms | 6.1-6.7 ms/step | 134-135 ms | 6.5 ms/step |
  | INT8, before | 1.4-2.1 s | 196-335 ms/step | | |
  | INT8 | 70 ms | 8.2-8.4 ms/step | 271 ms | 9.5 ms/step |
  | INT4 | 70 ms | 8.2-8.4 ms/step | 272 ms | 9.4-9.6 ms/step |

  The quantized caches are one kernel for prefill and decode; F16 keeps its
  lead through its tensor-core prefill and split decode kernels. Against
  the host loop the kernel's output is within 1.2e-6 of its scale on that
  card and 9.4e-7 on lavapipe. A GPU's division need not be correctly
  rounded, so an element of a query head that sits on a rounding boundary
  can land one quantization step from the host's; the test allows that for
  a head in fifty on Vulkan. `GEIST_KV_INT8_FUSED=0` keeps the host loops.
