### Changed
- **An external function defined without a prototype in scope fails the
  build (#343).** `-Wmissing-prototypes` is on. It found twelve: two in
  `weight_load/tensor_views.c`, whose header was not included, and ten
  AVX-512 VNNI kernels declared by hand in their dispatchers. Those
  declarations moved into `kernel_w8a8.h`, `kernel_w4a8.h` and
  `kernel_i2s.h`, which the kernels include too.
