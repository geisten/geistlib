### Fixed
- **A CPU MoE expert runs the dense FFN's SwiGLU kernel (#479).** The
  experts' `silu(gate) * up` ran on a host loop over libm's `expf` while a
  dense FFN runs the backend's `silu_mul` (on cpu_x86 a Cephes exp in AVX2),
  so an expert and the same dense FFN differed in the last bit; where an
  int8 activation quantization followed (cpu_x86's W8A8 F32 weights) that
  could become a whole grid step (`test_qwen35moe_unit`'s dense twin, 8.3e-3
  in the coverage build after #477). The experts now call the backend's
  `silu_mul` (or `silu` then `mul`, as the layer plan has it) on views of
  the session's scratch; the twin is bit-identical on cpu_x86 and cpu_scalar
  in every build mode.
