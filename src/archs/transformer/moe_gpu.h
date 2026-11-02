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
 * batch (grouped matvecs at one row each, else a GEMM per expert; with
 * decode_rows, a batched decode's rows (#245), one grouped matvec per
 * selection where the layer has grouped linears) and starts it without
 * waiting; finish, with launch's layer and decode_rows,
 * writes out[t] = cold[t] + the hot outputs of token t weighted by the
 * router, on the GPU, cold being the host's sum of the other experts —
 * no wait either. hot[0, n_res) have a copy on the GPU; hot[n_res, n_hot)
 * are a prefill step's staged experts (#239, moe_gpu_stage ran for them),
 * recorded after the others. */
[[nodiscard]] enum geist_status moe_gpu_launch(struct transformer_arch_session *sess,
                                               size_t                           layer,
                                               size_t                           seq,
                                               bool                             decode_rows,
                                               size_t                           n_res,
                                               size_t                           n_hot,
                                               const uint32_t                   hot[static n_hot]);
[[nodiscard]] enum geist_status moe_gpu_finish(struct transformer_arch_session *sess,
                                               size_t                           seq,
                                               size_t                           layer,
                                               bool                             decode_rows,
                                               size_t                           n_hot,
                                               const uint32_t                   hot[static n_hot],
                                               const float cold[static seq * sess->model->d_model],
                                               struct geist_buffer *out);

/* #239: a prefill step may copy expert e of layer `layer`, which has no
 * copy on the GPU, there for its m rows (m >= mg->stage_m_min, a host twin,
 * the session's buffers). */
[[nodiscard]] bool
moe_gpu_stageable(const struct transformer_arch_session *sess, size_t layer, size_t e, size_t m);

/* Copies experts e[0, n) of layer `layer` (each moe_gpu_stageable) into the
 * GPU's staging slots, without waiting; launch then runs them after the
 * resident experts. Not GEIST_OK: none copied, they stay on the CPU. */
[[nodiscard]] enum geist_status moe_gpu_stage(struct transformer_arch_session *sess,
                                              size_t                           layer,
                                              size_t                           n,
                                              const uint32_t                   e[static n]);

/* #239: the staging threshold from a cost model measured at load (after
 * moe_resolve_cold): copy rate, GPU and CPU time per expert row.
 * GEIST_MOE_STAGE_M_MIN=<rows> sets it (0: off). No-op without staging. */
void moe_gpu_stage_calibrate(struct transformer_arch_state *st);

/* Its rule: the rows from which a staged expert of `bytes`, copied at bw
 * bytes/s, pays against t_cpu s per row on the CPU: bytes / (bw * t_cpu),
 * the power of two at or below. SIZE_MAX: never (no copy rate). */
[[nodiscard]] size_t moe_stage_m_min(double bytes, double bw, double t_cpu);

/* #239: a cold expert's claim on a host copy for staging. */
struct moe_stage_cand {
    uint64_t rows;  /* prefill rows staging took, by the profile (moe_profile_prefill_rows) */
    uint64_t bytes; /* gate + up + down */
    uint64_t count; /* selections: the resident experts' ranking key */
    uint32_t idx;   /* e * n_layers + layer */
};

/* The order the host copies are made in, until the budget is used: the
 * most prefill rows per byte first, then the most selections, then the
 * lower idx. With no prefill rows (every rows 0) that is the resident
 * ranking's order. */
void moe_stage_order(size_t n, struct moe_stage_cand c[static n]);

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
 * experts on the device: mg->device[layer] and the session's buffers.
 * decode_rows: the seq rows are a batched decode's (#245), one token of
 * each of seq sessions; up to mg->device_rows of them take the route. */
[[nodiscard]] bool moe_gpu_device(const struct transformer_arch_session *sess,
                                  size_t                                 layer,
                                  size_t                                 seq,
                                  bool                                   decode_rows);

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
