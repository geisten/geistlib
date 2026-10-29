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
 * "profile" set), else the counts of earlier runs from the profile cache
 * (moe_profile.h, #165), else the same share of every layer; GEIST_MOE_HOT=<n>
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

/* One wait per layer (#161, #163), the model on the GPU (mg->one_wait, the
 * session's moe_x_io, 1 <= n_hot <= GEIST_MOE_MAX_EXPERTS): launch gathers
 * the rows of experts hot[0, n_hot) from x on the device, records their
 * batch (grouped matvecs at one row each, else a GEMM per expert) and
 * starts it without waiting; finish
 * writes out[t] = cold[t] + the hot outputs of token t weighted by the
 * router, on the GPU, cold being the host's sum of the other experts —
 * no wait either. */
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

/* #162: a decode step (seq == 1) of layer `layer` routes on the device
 * with the experts split. route_launch starts the work so far (the router)
 * and returns its mark (fused->map_marked: the host reads the logits and
 * moe_x_io and routes too), then records the device's routing -- the same
 * ranking -- into sess->moe_dev_ids and the hot experts' slots by it, and
 * starts them. route_finish, with the host's ids and weights in
 * sess->moe_ids/moe_wts: out = cold + the hot slots' outputs, weighted, on
 * the GPU. */
[[nodiscard]] bool
moe_gpu_routed(const struct transformer_arch_session *sess, size_t layer, size_t seq);
[[nodiscard]] uint64_t          moe_gpu_route_launch(struct transformer_arch_session *sess,
                                                     size_t                           layer,
                                                     enum geist_status               *status);
[[nodiscard]] enum geist_status moe_gpu_route_finish(struct transformer_arch_session *sess,
                                                     size_t                           layer,
                                                     const float cold[static sess->model->d_model],
                                                     struct geist_buffer *out);

/* #201: a decode step (seq == 1) of layer `layer` routes and runs its
 * experts on the device: mg->device[layer] and the session's buffers. */
[[nodiscard]] bool
moe_gpu_device(const struct transformer_arch_session *sess, size_t layer, size_t seq);

/* That step: routing of sess->moe_logits, gate, up, silu_mul, down by the
 * routed ids, and out = add + their weighted sum, recorded on the device
 * without a wait. x is the normed input [seq, d_model]. */
[[nodiscard]] enum geist_status moe_gpu_device_ffn(struct transformer_arch_session *sess,
                                                   size_t                           layer,
                                                   size_t                           seq,
                                                   struct geist_buffer             *x,
                                                   struct geist_buffer             *add,
                                                   struct geist_buffer             *out);

/* After a step's logits are read (the device idle): the device route's
 * non-finite flag as an error, cleared. GEIST_OK without device routing. */
[[nodiscard]] enum geist_status moe_gpu_device_check(struct transformer_arch_session *sess);

#endif
