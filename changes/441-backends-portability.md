### Changed
- **`check-layers` counts portability `#if`s outside the backends only, and
  checks the Vulkan sources in a required job (#441).** A backend is the
  boundary for its own hardware, like `src/base/` for the platform, so its
  `#if`s on `__ARM_NEON`, `__AVX512F__` and the like no longer count; the
  ceiling stays for the archs, formats, quant, io and engine (235 → 67
  counted). The musl x86_64 job runs `check-layers` with `vulkan` in
  `BACKENDS`.
