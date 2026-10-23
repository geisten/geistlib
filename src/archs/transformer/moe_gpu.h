/*
 * src/archs/transformer/moe_gpu.h - hot MoE experts on a GPU backend (#78
 * phase 3, docs/OFFLOAD.md).
 *
 * A mixture-of-experts model loaded on a CPU backend with a split backend
 * keeps every expert on the CPU and gives the split backend a copy of the
 * experts the router picks most often, as many as its free memory holds.
 * Per layer the GPU runs the routed rows of its experts as one batch while
 * the CPU workers run the rest; the weighted sum is the CPU's as before.
 */
#ifndef GEIST_TRANSFORMER_MOE_GPU_H
#define GEIST_TRANSFORMER_MOE_GPU_H

#include "arch_state.h"

struct gguf_ctx;

/* True when `gguf` is a mixture-of-experts model ("<arch>.expert_count"). */
[[nodiscard]] bool moe_gpu_is_moe(const struct gguf_ctx *gguf);

/* Places the hottest experts of `st` (loaded on its CPU backend) on `gbe`.
 * The order is GEIST_MOE_PROFILE's (USAGE lines of tools/moe_usage, the
 * "profile" set), else the same share of every layer; GEIST_MOE_HOT=<n>
 * sets the count past that budget (0: none). Experts that do not fit stay on the CPU, so
 * GEIST_OK does not mean any expert moved. */
[[nodiscard]] enum geist_status moe_gpu_attach(struct transformer_arch_state *st,
                                               struct geist_backend          *gbe);

void moe_gpu_destroy(struct transformer_moe_gpu *mg);

/* The model on a GPU (#164): resolves on st->moe_cpu every expert that got
 * no copy on the GPU (moe_gpu_attach ran first) and releases the file pages
 * of those that did. A hot expert then has no CPU copy, so a session that
 * cannot get its GPU buffers is refused (moe_gpu_session_alloc). */
[[nodiscard]] enum geist_status moe_resolve_cold(struct transformer_arch_state *st);

/* The session's buffers on the GPU backend. On failure the session runs its
 * experts on the CPU (one line on stderr) and this returns GEIST_OK, unless
 * the hot experts have no CPU copy (moe_resolve_cold): then the error. */
[[nodiscard]] enum geist_status moe_gpu_session_alloc(struct transformer_arch_session *sess);
void                            moe_gpu_session_free(struct transformer_arch_session *sess);

/* Expert e of layer `layer` has a GPU copy and the session can run it. */
[[nodiscard]] bool moe_gpu_hot(const struct transformer_arch_session *sess, size_t layer, size_t e);

/* Runs experts hot[0, n_hot) of layer `layer` over their bucketed rows
 * (sess->moe_xg / moe_row_start) on the GPU and writes their outputs into
 * sess->moe_y, as the CPU path would. Blocks until done. */
[[nodiscard]] enum geist_status moe_gpu_run(struct transformer_arch_session *sess,
                                            size_t                           layer,
                                            size_t                           n_hot,
                                            const uint32_t                   hot[static n_hot]);

/* One wait per layer (#161, #163), the model on the GPU: launchable says
 * whether experts hot[0, n_hot) can go this way; launch gathers their rows
 * from x on the device, records their batch (grouped matvecs at one row
 * each, else a GEMM per expert) and starts it without waiting; finish
 * writes out[t] = cold[t] + the hot outputs of token t weighted by the
 * router, on the GPU, cold being the host's sum of the other experts —
 * no wait either. */
[[nodiscard]] bool              moe_gpu_launchable(const struct transformer_arch_session *sess,
                                                   size_t                                 layer,
                                                   size_t                                 n_hot,
                                                   const uint32_t                         hot[static n_hot]);
[[nodiscard]] enum geist_status moe_gpu_launch(struct transformer_arch_session *sess,
                                               size_t                           layer,
                                               size_t                           seq,
                                               size_t                           n_hot,
                                               const uint32_t                   hot[static n_hot]);
[[nodiscard]] enum geist_status moe_gpu_finish(struct transformer_arch_session *sess,
                                               size_t                           seq,
                                               size_t                           n_hot,
                                               const uint32_t                   hot[static n_hot],
                                               const float cold[static seq * sess->model->d_model],
                                               struct geist_buffer *out);

#endif
