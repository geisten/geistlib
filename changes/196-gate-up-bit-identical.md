### Fixed

- Metal: the fused Q4_K gate/up decode kernels (`ffn_gate_up`, `ffn_gate_up_silu`) are bit-identical to two `linear_t` and the activation multiply at every K again. Above K = 1024 the SwiGLU kernel's running sums rounded differently from the matvec's (1e-5 relative at K = 2048 and 3072, Llama 3.2 3B's d_model), and at K = 1536 with n_out >= 1024 it ran the n4 layout where `linear_t` takes the half-simdgroup kernel. The parity test now covers K = 1536, 2048, 2560 and 3072 (#196).
