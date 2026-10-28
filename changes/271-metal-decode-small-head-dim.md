### Fixed
- **Metal decode attention at head_dim below 32 is correct again (#271).**
  From 32 cached keys on, a one-row attention took the split-KV decode
  kernels, which give every lane of a simdgroup one head_dim element
  unguarded: at head_dim 16 lanes 16-31 read unset threadgroup memory and
  overwrote the neighbouring simdgroup's partial sums, off by O(1) and
  intermittently NaN. That made `test_decision_prefix_unit` ("prefix + one
  suffix token at exact capacity", Metal on the qwen35 fixture) fail in
  about half of all `make test-decision` runs. head_dim < 32 now keeps the
  scalar kernel; real models (head_dim 64 and up) take the same kernels as
  before. New Metal parity cases at head_dim 16 and 8 hold decode to
  attention in double (1.4e-6), and the exact-capacity check now compares
  logits with a fresh prefill instead of only checking success.
