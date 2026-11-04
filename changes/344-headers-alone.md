### Fixed
- **Every internal header compiles on its own (#344).**
  `backends/common/selected_rows.h` includes `host_buffer.h` for the CPU
  backends' `struct geist_buffer`, and `metal/metal_objc.h` includes
  `metal_internal.h`, which includes it back after `struct metal_state`.
  The objects are byte-identical (macOS clang, Linux gcc-14); the two
  headers leave `tools/layers_baseline.txt`.
