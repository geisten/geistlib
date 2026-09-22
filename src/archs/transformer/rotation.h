/*
 * src/archs/transformer/rotation.h — activation transform for checkpoints
 * whose weights were folded into a blockwise Walsh-Hadamard basis
 * (prism.hadamard.* GGUF keys; Ternary-Bonsai-2).
 *
 * A folded weight W' = W (H S P)^T computes the original layer only if its
 * input is rotated first: W' (H S P x) = W x. The rotated rows of the
 * embedding table are restored after the lookup with the inverse,
 * h = S H z. Which weights are folded is fixed by the file; this module
 * checks that the set matches the call sites the forward pass rotates,
 * and runs the transform through fused->hadamard_rotate.
 *
 * The forward pass rotates, per layer:
 *   deltanet: the normed input of attn_qkv / attn_gate (not of the BF16
 *             ssm_alpha / ssm_beta, which read it unrotated first) and the
 *             mixer output into ssm_out (+ grouped-value permutation);
 *   attention: the normed input of attn_q/k/v and the input of attn_output;
 *   FFN: the normed input of ffn_gate/up and the input of ffn_down;
 * plus the final-norm output into output.weight.
 */
#pragma once

#include "arch_state.h"

#include <geist.h>

#include <stdbool.h>
#include <stddef.h>

/* Read and validate prism.hadamard.* from the state's GGUF after every
 * layer is loaded; on success st->rotation is filled (inactive when the
 * file has no prism.hadamard.version). Unknown versions / transforms /
 * axes / sign modes, a family or backend that cannot run the transform,
 * MTP layers and a tied lm_head are GEIST_E_UNSUPPORTED; a malformed or
 * inconsistent description (bad block, partially rotated layer, unknown
 * weight name, bad sign table) is GEIST_E_FORMAT. The backend error slot
 * names the reason. */
[[nodiscard]] enum geist_status transformer_rotation_load(struct transformer_arch_state *st);

/* Release the sign buffers. Safe on a state that never loaded any. */
void transformer_rotation_release(struct transformer_arch_state *st);

/* Rotate rows x [rows, width] into y. Forward: y = H(S P x) with P the
 * grouped-value permutation when grouped_v (ssm_out input only). Inverse:
 * y = S H x. No-op success when the model has no rotation. y may be x
 * unless grouped_v. */
[[nodiscard]] enum geist_status transformer_rotate(const struct transformer_arch_state *st,
                                                   size_t                               rows,
                                                   size_t                               width,
                                                   bool                                 grouped_v,
                                                   bool                                 inverse,
                                                   struct geist_buffer                 *x,
                                                   struct geist_buffer                 *y);
