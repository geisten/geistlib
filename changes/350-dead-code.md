### Removed
- **Five internal functions and one macro that nothing called (#350).**
  `conv2d_fp32` (audio; `conv2d_fp32_from` with `start_h = 0` is the same),
  `linear_w_scaled_input_or_legacy`, `transformer_advance_audio_token`,
  `vk_seq_dispatch`, `gguf_reader.c`'s `read_u8` and `VISION_MAX_PATCHES`.
  Found with `nm` over the macOS (cpu_neon + Metal), Linux arm64 gcc-14
  (cpu_neon + Vulkan) and x86-64 (cpu_x86 + Vulkan) builds of the library,
  the tools and every test, cross-checked against the coverage report
  (zero calls). No public symbol changes.
