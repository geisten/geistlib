/*
 * src/archs/transformer/moe_adapt.h - the adaptive expert cache (#166):
 * a mixture of experts on a discrete GPU swaps hot and cold experts between
 * decode windows, and only where that shortens the layer.
 *
 * A decode layer with the model on the GPU runs two branches at once: the
 * GPU's hot experts and the CPU's cold ones; the layer takes the longer.
 * Over a window of W decode steps the cache counts each layer's selections
 * per expert and times both branches: the CPU branch on every step, the
 * GPU branch on the window's probe steps (the host waits for the hot batch
 * right after its launch, two steps per window). At the window's end, in a
 * layer whose CPU branch is the longer one by a margin, it plans swaps:
 * the most selected cold expert for the least selected hot one, while the
 * predicted CPU branch stays above the predicted GPU branch, i.e. up to
 * the balance point. A layer whose GPU branch is already the longer one is
 * left alone. The plan's swaps run one per decode step at the start of the
 * next steps (moe_gpu_swap: one expert's device copy for another's, so
 * device memory stays as placed). Only with one session on the model.
 *
 * GEIST_MOE_ADAPT=1 turns it on (off by default); GEIST_MOE_ADAPT_WINDOW
 * (default 64 tokens), GEIST_MOE_ADAPT_SWAPS (most swaps per window,
 * default 16). A cold expert replaces a hot one only with at least twice its
 * selections (+2), and an expert that moved stays for two windows. With
 * the cache on, every expert keeps its CPU copy (#164 does not apply), so a
 * swap moves one device copy and nothing is repacked. Placement then follows the text, so which
 * expert runs on which backend, and the low bits of the result, depend on timing.
 */
#ifndef GEIST_TRANSFORMER_MOE_ADAPT_H
#define GEIST_TRANSFORMER_MOE_ADAPT_H

#include <geist.h>

#include <stddef.h>
#include <stdint.h>

struct transformer_arch_state;
struct moe_adapt;

/* The cache for st->moe_gpu when GEIST_MOE_ADAPT=1, else nullptr. */
[[nodiscard]] struct moe_adapt *moe_adapt_create(const struct transformer_arch_state *st);

void moe_adapt_destroy(struct moe_adapt *a);

/* At the first MoE layer of a decode step: runs one planned swap, closes a
 * window (plans the next swaps) and says whether this step probes the GPU
 * branch. Returns the swap's status (GEIST_OK without one). */
[[nodiscard]] enum geist_status moe_adapt_step(struct transformer_arch_state *st, bool *probe);

/* A decode step's routed selections of layer `layer`: ids[0, n) (expert
 * ids, the routed ones), n_cold of them on the CPU, and the CPU branch's
 * time; gpu_ns the hot batch's time on a probe step (0 otherwise). */
void moe_adapt_record(struct moe_adapt *a,
                      size_t            layer,
                      size_t            n,
                      const uint32_t    ids[static n],
                      size_t            n_cold,
                      uint64_t          cpu_ns,
                      uint64_t          gpu_ns);

#endif
