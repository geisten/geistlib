/*
 * vision_encoder — Gemma 4 vision tower (SigLIP-derived ViT).
 *
 * P3: weight loading + per-block forward.
 *
 * Reference (transformers/models/gemma4/modeling_gemma4.py):
 *   Gemma4VisionPatchEmbedder, Gemma4VisionEncoder,
 *   Gemma4VisionEncoderLayer, Gemma4VisionAttention, Gemma4VisionMLP.
 *
 * Block layout (sandwich norm, no final norm at top level):
 *   residual = h
 *   h = input_layernorm(h)
 *   h = self_attn(h, position_embeddings, position_ids)
 *   h = post_attention_layernorm(h)
 *   h = residual + h
 *   residual = h
 *   h = pre_feedforward_layernorm(h)
 *   h = down_proj(gelu_tanh(gate_proj(h)) * up_proj(h))
 *   h = post_feedforward_layernorm(h)
 *   h = residual + h
 *
 * Attention internals:
 *   q/k/v_proj → reshape (n, n_heads, head_dim) → q_norm / k_norm /
 *   v_norm-with_scale=False → 2D-split RoPE on q/k → bidirectional
 *   scaled softmax → o_proj.
 *
 * Weights are bf16 in safetensors; loaded once at create-time and held
 * as fp32 fields. Clipped-linear input/output_min/max scalars are
 * captured but unused for now — they're +/- inf in shipped checkpoints
 * (no actual clipping) and adding clamp ops would only slow inference
 * without changing values.
 */
#include "vision_encoder.h"

#include "checked.h"
#include "heap.h"
#include "gemma4_kernels.h"
#include "image_pipeline.h"
#include "safetensors_reader.h"
#include "vision_kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* Tower constants for Gemma 4 vision (config.json vision_config). */
#define VTH VISION_TOWER_HIDDEN /* 768  */
#define V_INTER 3072            /* intermediate_size */
#define V_HEADS 12              /* num_attention_heads */
#define V_HEAD_D 64             /* head_dim */
#define V_LAYERS 16             /* num_hidden_layers */
#define V_EPS 1e-6f             /* rms_norm_eps */
#define V_THETA 100.0f          /* rope_theta */
#define V_POS_GRID 10240        /* position_embedding_size */

/* Per-Linear activation clip (Gemma4ClippableLinear): input is clamped
 * to [in_min, in_max] before the matmul, output is clamped to
 * [out_min, out_max] after. Ranges are stored as 0-d bf16 scalars in
 * the safetensors (4 per linear × 7 linears × 16 layers = 448 scalars).
 * Without this clamp the residual stream explodes by layer 1. */
struct clip {
    float in_min, in_max, out_min, out_max;
};

struct vision_layer {
    /* Norm weights (768) */
    float *input_layernorm;
    float *post_attention_layernorm;
    float *pre_feedforward_layernorm;
    float *post_feedforward_layernorm;

    /* Attention projections (out, in) = (768, 768) for each.
     * Q/K/V share the same input clip (they consume the same `normed`
     * buffer), but each has its own output clip. q/k/v_proj are
     * stacked into a single (3*VTH, VTH) qkv_proj weight at load time
     * so one sgemm covers all three (matching the gate+up fusion). */
    float      *qkv_proj; /* (3*VTH, VTH) stacked Q;K;V */
    struct clip q_clip;
    struct clip k_clip;
    struct clip v_clip;
    float      *o_proj;
    struct clip o_clip;

    /* Per-head norms on Q, K (64) — RMSNorm over the head_dim. */
    float *q_norm;
    float *k_norm;
    /* v_norm: with_scale=False, no weight. */

    /* MLP projections. gate/up share input clip; down has its own.
     * gate_up_proj is the stacked (2*INTER, VTH) weight matrix built at
     * load time so one sgemm covers both gate_proj and up_proj — the
     * resulting (n, 2*INTER) buffer is split into gate/up halves for
     * the GeGLU activation. Lets Accelerate amortize tile setup and
     * cache reuse across the doubled output dim. */
    float      *gate_up_proj; /* (2*INTER, VTH) */
    struct clip gate_clip;
    struct clip up_clip;
    float      *down_proj;
    struct clip down_clip; /* (768, 3072) */
};

struct VisionEncoder {
    struct st_ctx *sf;

    /* Patch embed */
    float *input_proj;         /* (768, 768) — for (3*16*16=768 → 768) */
    float *position_embedding; /* (2, 10240, 768) — flat layout */

    struct vision_layer layers[V_LAYERS];

    /* Multimodal projector: pool-output × root_hidden → RMSNorm
     * (no scale) → Linear (768 → soft_dim).
     * Weight is model.embed_vision.embedding_projection.weight. */
    float *projector_weight; /* (soft_dim, VTH) */
    /* Soft-token width == the TEXT model's residual stream (projector
     * rows): 1536 E2B, 2560 E4B. Read from the safetensors shape (#258). */
    size_t soft_dim;
};

/* Load a 0-d bf16 scalar by name. Returns false if missing or not bf16. */
static bool load_bf16_scalar(struct st_ctx *sf, const char *name, float *out) {
    const struct st_tensor_t *t = st_get(sf, name);
    if (t == nullptr || t->dtype != ST_DTYPE_BF16 || t->nbytes != 2) {
        fprintf(stderr, "vision_encoder: missing/bad scalar '%s'\n", name);
        return false;
    }
    uint16_t raw;
    memcpy(&raw, t->data, 2);
    uint32_t bits = ((uint32_t) raw) << 16;
    memcpy(out, &bits, 4);
    return true;
}

static bool load_clip(struct st_ctx *sf, const char *prefix, struct clip *c) {
    char b[256];
    snprintf(b, sizeof b, "%s.input_min", prefix);
    if (!load_bf16_scalar(sf, b, &c->in_min))
        return false;
    snprintf(b, sizeof b, "%s.input_max", prefix);
    if (!load_bf16_scalar(sf, b, &c->in_max))
        return false;
    snprintf(b, sizeof b, "%s.output_min", prefix);
    if (!load_bf16_scalar(sf, b, &c->out_min))
        return false;
    snprintf(b, sizeof b, "%s.output_max", prefix);
    if (!load_bf16_scalar(sf, b, &c->out_max))
        return false;
    return true;
}

/* Load a BF16 tensor by name, dequant to a freshly-allocated fp32 buffer.
 * Returns nullptr on failure. */
static float *load_bf16(struct st_ctx *sf, const char *name, size_t n) {
    const struct st_tensor_t *t = st_get(sf, name);
    if (t == nullptr) {
        fprintf(stderr, "vision_encoder: missing tensor '%s'\n", name);
        return nullptr;
    }
    if (t->dtype != ST_DTYPE_BF16) {
        fprintf(stderr,
                "vision_encoder: '%s' dtype=%s, want BF16\n",
                name,
                st_dtype_name(t->dtype));
        return nullptr;
    }
    if (t->nbytes != n * 2) {
        fprintf(stderr,
                "vision_encoder: '%s' size mismatch: got %zu bytes, want %zu\n",
                name,
                t->nbytes,
                n * 2);
        return nullptr;
    }
    return bf16_alloc_fp32(t->data, n);
}

static bool load_layer(struct st_ctx *sf, int idx, struct vision_layer *L) {
    char        buf[256];
    const char *P = "model.vision_tower.encoder.layers";

#define LOAD_(field, suffix, count)                            \
    do {                                                       \
        snprintf(buf, sizeof buf, "%s.%d.%s", P, idx, suffix); \
        L->field = load_bf16(sf, buf, (count));                \
        if (L->field == nullptr)                               \
            return false;                                      \
    } while (0)

    LOAD_(input_layernorm, "input_layernorm.weight", VTH);
    LOAD_(post_attention_layernorm, "post_attention_layernorm.weight", VTH);
    LOAD_(pre_feedforward_layernorm, "pre_feedforward_layernorm.weight", VTH);
    LOAD_(post_feedforward_layernorm, "post_feedforward_layernorm.weight", VTH);

    /* Q/K/V stacked into a single (3*VTH, VTH) weight at load time. */
    {
        float *q_tmp = nullptr, *k_tmp = nullptr, *v_tmp = nullptr;
        snprintf(buf, sizeof buf, "%s.%d.%s", P, idx, "self_attn.q_proj.linear.weight");
        q_tmp = load_bf16(sf, buf, (size_t) VTH * VTH);
        if (q_tmp == nullptr)
            return false;
        snprintf(buf, sizeof buf, "%s.%d.%s", P, idx, "self_attn.k_proj.linear.weight");
        k_tmp = load_bf16(sf, buf, (size_t) VTH * VTH);
        if (k_tmp == nullptr) {
            safe_free((void **) &q_tmp);
            return false;
        }
        snprintf(buf, sizeof buf, "%s.%d.%s", P, idx, "self_attn.v_proj.linear.weight");
        v_tmp = load_bf16(sf, buf, (size_t) VTH * VTH);
        if (v_tmp == nullptr) {
            safe_free((void **) &q_tmp);
            safe_free((void **) &k_tmp);
            return false;
        }
        L->qkv_proj = heap_alloc_array_aligned(float, 3 * (size_t) VTH * VTH);
        if (L->qkv_proj == nullptr) {
            safe_free((void **) &q_tmp);
            safe_free((void **) &k_tmp);
            safe_free((void **) &v_tmp);
            return false;
        }
        const size_t one = (size_t) VTH * VTH;
        memcpy(L->qkv_proj, q_tmp, one * sizeof(float));
        memcpy(L->qkv_proj + one, k_tmp, one * sizeof(float));
        memcpy(L->qkv_proj + 2 * one, v_tmp, one * sizeof(float));
        safe_free((void **) &q_tmp);
        safe_free((void **) &k_tmp);
        safe_free((void **) &v_tmp);
    }
    LOAD_(o_proj, "self_attn.o_proj.linear.weight", VTH * VTH);

    LOAD_(q_norm, "self_attn.q_norm.weight", V_HEAD_D);
    LOAD_(k_norm, "self_attn.k_norm.weight", V_HEAD_D);

    /* Load gate+up as separate buffers temporarily, then stack into
     * gate_up_proj (rows 0..INTER-1 = gate, rows INTER..2*INTER-1 = up). */
    {
        float *gate_tmp = nullptr, *up_tmp = nullptr;
        snprintf(buf, sizeof buf, "%s.%d.%s", P, idx, "mlp.gate_proj.linear.weight");
        gate_tmp = load_bf16(sf, buf, (size_t) V_INTER * VTH);
        if (gate_tmp == nullptr)
            return false;
        snprintf(buf, sizeof buf, "%s.%d.%s", P, idx, "mlp.up_proj.linear.weight");
        up_tmp = load_bf16(sf, buf, (size_t) V_INTER * VTH);
        if (up_tmp == nullptr) {
            safe_free((void **) &gate_tmp);
            return false;
        }
        L->gate_up_proj = heap_alloc_array_aligned(float, 2 * (size_t) V_INTER * VTH);
        if (L->gate_up_proj == nullptr) {
            safe_free((void **) &gate_tmp);
            safe_free((void **) &up_tmp);
            return false;
        }
        memcpy(L->gate_up_proj, gate_tmp, (size_t) V_INTER * VTH * sizeof(float));
        memcpy(L->gate_up_proj + (size_t) V_INTER * VTH,
               up_tmp,
               (size_t) V_INTER * VTH * sizeof(float));
        safe_free((void **) &gate_tmp);
        safe_free((void **) &up_tmp);
    }
    LOAD_(down_proj, "mlp.down_proj.linear.weight", (size_t) VTH * V_INTER);

#undef LOAD_

    /* Per-Linear clipping scalars. */
    char b[256];
#define CLIP(field, suffix)                                \
    do {                                                   \
        snprintf(b, sizeof b, "%s.%d.%s", P, idx, suffix); \
        if (!load_clip(sf, b, &L->field))                  \
            return false;                                  \
    } while (0)
    CLIP(q_clip, "self_attn.q_proj");
    CLIP(k_clip, "self_attn.k_proj");
    CLIP(v_clip, "self_attn.v_proj");
    CLIP(o_clip, "self_attn.o_proj");
    CLIP(gate_clip, "mlp.gate_proj");
    CLIP(up_clip, "mlp.up_proj");
    CLIP(down_clip, "mlp.down_proj");
#undef CLIP
    return true;
}

static void free_layer(struct vision_layer *L) {
    void **fields[] = {
            (void **) &L->input_layernorm,
            (void **) &L->post_attention_layernorm,
            (void **) &L->pre_feedforward_layernorm,
            (void **) &L->post_feedforward_layernorm,
            (void **) &L->qkv_proj,
            (void **) &L->o_proj,
            (void **) &L->q_norm,
            (void **) &L->k_norm,
            (void **) &L->gate_up_proj,
            (void **) &L->down_proj,
    };
    for (size_t i = 0; i < sizeof fields / sizeof fields[0]; i++) {
        if (*fields[i] != nullptr)
            safe_free(fields[i]);
    }
}

struct VisionEncoder *vision_encoder_create(const char *safetensors_path) {
    if (safetensors_path == nullptr)
        return nullptr;
    const char    *err = nullptr;
    struct st_ctx *sf  = st_open(safetensors_path, &err);
    if (sf == nullptr) {
        fprintf(stderr, "vision_encoder: %s\n", err ? err : "open failed");
        return nullptr;
    }
    struct VisionEncoder *v = heap_calloc_array_aligned(struct VisionEncoder, 1);
    if (v == nullptr) {
        st_close(sf);
        return nullptr;
    }
    v->sf = sf;

    v->input_proj = load_bf16(
            sf, "model.vision_tower.patch_embedder.input_proj.weight", (size_t) VTH * VTH);
    if (v->input_proj == nullptr)
        goto fail;

    v->position_embedding = load_bf16(sf,
                                      "model.vision_tower.patch_embedder.position_embedding_table",
                                      (size_t) 2 * V_POS_GRID * VTH);
    if (v->position_embedding == nullptr)
        goto fail;

    fprintf(stderr, "vision_encoder: loading %d ViT layers...\n", V_LAYERS);
    for (int i = 0; i < V_LAYERS; i++) {
        if (!load_layer(sf, i, &v->layers[i]))
            goto fail;
    }
    fprintf(stderr, "vision_encoder: layers loaded.\n");

    /* Projector rows ARE the soft-token width (text residual stream:
     * 1536 E2B, 2560 E4B) — read from the checkpoint, not pinned (#258). */
    {
        const struct st_tensor_t *ep = st_get(sf, "model.embed_vision.embedding_projection.weight");
        if (ep == nullptr || ep->rank != 2 || ep->shape[1] != VTH) {
            fprintf(stderr,
                    "vision_encoder: model.embed_vision.embedding_projection.weight missing "
                    "or not (text_hidden, %d)\n",
                    VTH);
            goto fail;
        }
        v->soft_dim = ep->shape[0];
    }
    v->projector_weight = load_bf16(
            sf, "model.embed_vision.embedding_projection.weight", v->soft_dim * (size_t) VTH);
    if (v->projector_weight == nullptr)
        goto fail;
    fprintf(stderr,
            "vision_encoder: soft-token dim %zu (from embedding_projection)\n",
            v->soft_dim);
    return v;

fail:
    vision_encoder_destroy(v);
    return nullptr;
}

size_t vision_encoder_soft_dim(const struct VisionEncoder *v) {
    /* Pre-load callers get the E2B default; after create it is the
     * checkpoint's embedding_projection row count. */
    return (v != nullptr && v->soft_dim > 0) ? v->soft_dim : VISION_SOFT_TOKEN_DIM;
}

void vision_encoder_destroy(struct VisionEncoder *v) {
    if (v == nullptr)
        return;
    if (v->input_proj)
        safe_free((void **) &v->input_proj);
    if (v->position_embedding)
        safe_free((void **) &v->position_embedding);
    if (v->projector_weight)
        safe_free((void **) &v->projector_weight);
    for (int i = 0; i < V_LAYERS; i++)
        free_layer(&v->layers[i]);
    if (v->sf != nullptr)
        st_close(v->sf);
    v->sf   = nullptr;
    void *p = v;
    safe_free(&p);
}

/* Optional debug dump: writes a buffer to $GEIST_VISION_DUMP_DIR/<name>.bin
 * when the env var is set. */
static void dbg_dump(const char *name, const float *buf, size_t n) {
    const char *root = getenv("GEIST_VISION_DUMP_DIR");
    if (root == nullptr || root[0] == '\0')
        return;
    char path[512];
    snprintf(path, sizeof path, "%s/%s.bin", root, name);
    FILE *f = fopen(path, "wb");
    if (f == nullptr)
        return;
    fwrite(buf, sizeof(float), n, f);
    fclose(f);
}

/* In-place per-head RMSNorm: applies rmsnorm_fp32 to each (head_dim,)
 * slice within (n_tokens, n_heads, head_dim). */
static void rmsnorm_per_head(float       *x,
                             const float *weight,
                             size_t       n_tokens,
                             size_t       n_heads,
                             size_t       head_dim,
                             float        eps) {
    rmsnorm_fp32(x, weight, n_tokens * n_heads, head_dim, eps, x);
}

/* In-place clamp x to [lo, hi] (Gemma4ClippableLinear input/output gates).
 * NEON-vectorized 16-wide on ARM; bit-exact vs scalar for finite inputs
 * (NaN behavior is irrelevant since activations are guaranteed finite
 * by the upstream clamp chain). */
static void clamp_fp32(float *x, size_t n, float lo, float hi) {
    size_t i = 0;
#if defined(__ARM_NEON)
    const float32x4_t vlo = vdupq_n_f32(lo);
    const float32x4_t vhi = vdupq_n_f32(hi);
    for (; i + 16 <= n; i += 16) {
        float32x4_t v0 = vld1q_f32(x + i + 0);
        float32x4_t v1 = vld1q_f32(x + i + 4);
        float32x4_t v2 = vld1q_f32(x + i + 8);
        float32x4_t v3 = vld1q_f32(x + i + 12);
        v0             = vminq_f32(vmaxq_f32(v0, vlo), vhi);
        v1             = vminq_f32(vmaxq_f32(v1, vlo), vhi);
        v2             = vminq_f32(vmaxq_f32(v2, vlo), vhi);
        v3             = vminq_f32(vmaxq_f32(v3, vlo), vhi);
        vst1q_f32(x + i + 0, v0);
        vst1q_f32(x + i + 4, v1);
        vst1q_f32(x + i + 8, v2);
        vst1q_f32(x + i + 12, v3);
    }
#endif
    for (; i < n; i++) {
        if (x[i] < lo)
            x[i] = lo;
        else if (x[i] > hi)
            x[i] = hi;
    }
}

bool vision_encoder_run_tower(const struct VisionEncoder *v,
                              const float                *patches_in,
                              const int32_t              *positions,
                              size_t                      n_patches,
                              float                      *hidden_out) {
    if (v == nullptr || patches_in == nullptr || positions == nullptr || hidden_out == nullptr ||
        n_patches == 0) {
        return false;
    }

    /* Scratch buffers. Largest: gate/up at (n, V_INTER=3072) fp32. */
    float *scaled           = heap_alloc_array_aligned(float, n_patches *VTH);
    float *resid            = heap_alloc_array_aligned(float, n_patches *VTH);
    float *normed           = heap_alloc_array_aligned(float, n_patches *VTH);
    float *q                = heap_alloc_array_aligned(float, n_patches *VTH);
    float *k                = heap_alloc_array_aligned(float, n_patches *VTH);
    float *vbuf             = heap_alloc_array_aligned(float, n_patches *VTH);
    float *attn             = heap_alloc_array_aligned(float, n_patches *VTH);
    float *proj             = heap_alloc_array_aligned(float, n_patches *VTH);
    float *gate             = heap_alloc_array_aligned(float, n_patches *V_INTER);
    float *up               = heap_alloc_array_aligned(float, n_patches *V_INTER);
    float *gate_up_combined = heap_alloc_array_aligned(float, n_patches * 2 * V_INTER);
    float *qkv_combined     = heap_alloc_array_aligned(float, n_patches * 3 * VTH);

    if (!scaled || !resid || !normed || !q || !k || !vbuf || !attn || !proj || !gate || !up ||
        !gate_up_combined || !qkv_combined) {
        goto fail;
    }

    /* ---- Patch embedder ---------------------------------------------- */
    /* scaled = 2 * (patches_in - 0.5) */
    {
        const size_t N = n_patches * VTH;
        for (size_t i = 0; i < N; i++)
            scaled[i] = 2.0f * (patches_in[i] - 0.5f);
    }
    /* hidden = scaled @ input_proj^T  → (n, 768) */
    linear_fp32(scaled, v->input_proj, nullptr, n_patches, VTH, VTH, hidden_out);

    /* pos_embed += table[0, x] + table[1, y]; HF clamps -1 → 0.
     * Padding rows (px==py==-1) end up with table[0, 0] + table[1, 0],
     * then HF's patch_embedder zeros them via `padding_positions`. We
     * don't carry a padding mask here; callers using non-square inputs
     * rely on the planner producing exact-fit grids (no padding). */
    const float *tbl0 = v->position_embedding; /* (10240, 768) */
    const float *tbl1 = v->position_embedding + (size_t) V_POS_GRID * VTH;
    for (size_t t = 0; t < n_patches; t++) {
        int32_t px = positions[t * 2 + 0];
        if (px < 0)
            px = 0;
        int32_t py = positions[t * 2 + 1];
        if (py < 0)
            py = 0;
        const float *rx = tbl0 + (size_t) px * VTH;
        const float *ry = tbl1 + (size_t) py * VTH;
        float       *h  = hidden_out + t * VTH;
        for (size_t j = 0; j < VTH; j++)
            h[j] += rx[j] + ry[j];
    }
    dbg_dump("patch_embed_out", hidden_out, n_patches * VTH);

    /* ---- Encoder layers ---------------------------------------------- */
    for (int li = 0; li < V_LAYERS; li++) {
        const struct vision_layer *L = &v->layers[li];

        /* residual = h */
        memcpy(resid, hidden_out, n_patches * VTH * sizeof(float));
        rmsnorm_fp32(hidden_out, L->input_layernorm, n_patches, VTH, V_EPS, normed);

        /* Q/K/V share `normed` input — clamp once with their common range,
         * then run a fused (3*VTH, VTH) sgemm and split into q/k/vbuf. */
        clamp_fp32(normed, n_patches * VTH, L->q_clip.in_min, L->q_clip.in_max);
        linear_fp32(normed, L->qkv_proj, nullptr, n_patches, VTH, 3 * VTH, qkv_combined);
        {
            const float qlo = L->q_clip.out_min, qhi = L->q_clip.out_max;
            const float klo = L->k_clip.out_min, khi = L->k_clip.out_max;
            const float vlo = L->v_clip.out_min, vhi = L->v_clip.out_max;
#if defined(__ARM_NEON)
            const float32x4_t vqlo = vdupq_n_f32(qlo), vqhi = vdupq_n_f32(qhi);
            const float32x4_t vklo = vdupq_n_f32(klo), vkhi = vdupq_n_f32(khi);
            const float32x4_t vvlo = vdupq_n_f32(vlo), vvhi = vdupq_n_f32(vhi);
#endif
            for (size_t t = 0; t < n_patches; t++) {
                const float *src_q = qkv_combined + t * 3 * VTH;
                const float *src_k = src_q + VTH;
                const float *src_v = src_q + 2 * VTH;
                float       *dst_q = q + t * VTH;
                float       *dst_k = k + t * VTH;
                float       *dst_v = vbuf + t * VTH;
                size_t       j     = 0;
#if defined(__ARM_NEON)
                for (; j + 16 <= VTH; j += 16) {
                    float32x4_t q0 = vminq_f32(vmaxq_f32(vld1q_f32(src_q + j + 0), vqlo), vqhi);
                    float32x4_t q1 = vminq_f32(vmaxq_f32(vld1q_f32(src_q + j + 4), vqlo), vqhi);
                    float32x4_t q2 = vminq_f32(vmaxq_f32(vld1q_f32(src_q + j + 8), vqlo), vqhi);
                    float32x4_t q3 = vminq_f32(vmaxq_f32(vld1q_f32(src_q + j + 12), vqlo), vqhi);
                    float32x4_t k0 = vminq_f32(vmaxq_f32(vld1q_f32(src_k + j + 0), vklo), vkhi);
                    float32x4_t k1 = vminq_f32(vmaxq_f32(vld1q_f32(src_k + j + 4), vklo), vkhi);
                    float32x4_t k2 = vminq_f32(vmaxq_f32(vld1q_f32(src_k + j + 8), vklo), vkhi);
                    float32x4_t k3 = vminq_f32(vmaxq_f32(vld1q_f32(src_k + j + 12), vklo), vkhi);
                    float32x4_t v0 = vminq_f32(vmaxq_f32(vld1q_f32(src_v + j + 0), vvlo), vvhi);
                    float32x4_t v1 = vminq_f32(vmaxq_f32(vld1q_f32(src_v + j + 4), vvlo), vvhi);
                    float32x4_t v2 = vminq_f32(vmaxq_f32(vld1q_f32(src_v + j + 8), vvlo), vvhi);
                    float32x4_t v3 = vminq_f32(vmaxq_f32(vld1q_f32(src_v + j + 12), vvlo), vvhi);
                    vst1q_f32(dst_q + j + 0, q0);
                    vst1q_f32(dst_q + j + 4, q1);
                    vst1q_f32(dst_q + j + 8, q2);
                    vst1q_f32(dst_q + j + 12, q3);
                    vst1q_f32(dst_k + j + 0, k0);
                    vst1q_f32(dst_k + j + 4, k1);
                    vst1q_f32(dst_k + j + 8, k2);
                    vst1q_f32(dst_k + j + 12, k3);
                    vst1q_f32(dst_v + j + 0, v0);
                    vst1q_f32(dst_v + j + 4, v1);
                    vst1q_f32(dst_v + j + 8, v2);
                    vst1q_f32(dst_v + j + 12, v3);
                }
#endif
                for (; j < VTH; j++) {
                    float qv = src_q[j];
                    if (qv < qlo)
                        qv = qlo;
                    else if (qv > qhi)
                        qv = qhi;
                    float kv = src_k[j];
                    if (kv < klo)
                        kv = klo;
                    else if (kv > khi)
                        kv = khi;
                    float vv = src_v[j];
                    if (vv < vlo)
                        vv = vlo;
                    else if (vv > vhi)
                        vv = vhi;
                    dst_q[j] = qv;
                    dst_k[j] = kv;
                    dst_v[j] = vv;
                }
            }
        }

        rmsnorm_per_head(q, L->q_norm, n_patches, V_HEADS, V_HEAD_D, V_EPS);
        rmsnorm_per_head(k, L->k_norm, n_patches, V_HEADS, V_HEAD_D, V_EPS);
        rmsnorm_per_head(vbuf, nullptr, n_patches, V_HEADS, V_HEAD_D, V_EPS);

        rope_2d_split_fp32(q, positions, n_patches, V_HEADS, V_HEAD_D, V_THETA);
        rope_2d_split_fp32(k, positions, n_patches, V_HEADS, V_HEAD_D, V_THETA);

        vision_attention_bidir_fp32(q, k, vbuf, n_patches, V_HEADS, V_HEAD_D, attn);

        clamp_fp32(attn, n_patches * VTH, L->o_clip.in_min, L->o_clip.in_max);
        linear_fp32(attn, L->o_proj, nullptr, n_patches, VTH, VTH, proj);
        clamp_fp32(proj, n_patches * VTH, L->o_clip.out_min, L->o_clip.out_max);

        rmsnorm_fp32(proj, L->post_attention_layernorm, n_patches, VTH, V_EPS, proj);
        add_fp32(resid, proj, n_patches * VTH, hidden_out);

        memcpy(resid, hidden_out, n_patches * VTH * sizeof(float));
        rmsnorm_fp32(hidden_out, L->pre_feedforward_layernorm, n_patches, VTH, V_EPS, normed);

        clamp_fp32(normed, n_patches * VTH, L->gate_clip.in_min, L->gate_clip.in_max);

        /* Fused gate+up sgemm: one (n, 2*INTER) matmul, then split-and-
         * clamp in one fused pass into separate gate/up buffers. */
        linear_fp32(
                normed, L->gate_up_proj, nullptr, n_patches, VTH, 2 * V_INTER, gate_up_combined);
        {
            const float glo = L->gate_clip.out_min, ghi = L->gate_clip.out_max;
            const float ulo = L->up_clip.out_min, uhi = L->up_clip.out_max;
#if defined(__ARM_NEON)
            const float32x4_t vglo = vdupq_n_f32(glo), vghi = vdupq_n_f32(ghi);
            const float32x4_t vulo = vdupq_n_f32(ulo), vuhi = vdupq_n_f32(uhi);
#endif
            for (size_t t = 0; t < n_patches; t++) {
                const float *src_g = gate_up_combined + t * 2 * V_INTER;
                const float *src_u = src_g + V_INTER;
                float       *dst_g = gate + t * V_INTER;
                float       *dst_u = up + t * V_INTER;
                size_t       j     = 0;
#if defined(__ARM_NEON)
                for (; j + 16 <= V_INTER; j += 16) {
                    float32x4_t g0 = vminq_f32(vmaxq_f32(vld1q_f32(src_g + j + 0), vglo), vghi);
                    float32x4_t g1 = vminq_f32(vmaxq_f32(vld1q_f32(src_g + j + 4), vglo), vghi);
                    float32x4_t g2 = vminq_f32(vmaxq_f32(vld1q_f32(src_g + j + 8), vglo), vghi);
                    float32x4_t g3 = vminq_f32(vmaxq_f32(vld1q_f32(src_g + j + 12), vglo), vghi);
                    float32x4_t u0 = vminq_f32(vmaxq_f32(vld1q_f32(src_u + j + 0), vulo), vuhi);
                    float32x4_t u1 = vminq_f32(vmaxq_f32(vld1q_f32(src_u + j + 4), vulo), vuhi);
                    float32x4_t u2 = vminq_f32(vmaxq_f32(vld1q_f32(src_u + j + 8), vulo), vuhi);
                    float32x4_t u3 = vminq_f32(vmaxq_f32(vld1q_f32(src_u + j + 12), vulo), vuhi);
                    vst1q_f32(dst_g + j + 0, g0);
                    vst1q_f32(dst_g + j + 4, g1);
                    vst1q_f32(dst_g + j + 8, g2);
                    vst1q_f32(dst_g + j + 12, g3);
                    vst1q_f32(dst_u + j + 0, u0);
                    vst1q_f32(dst_u + j + 4, u1);
                    vst1q_f32(dst_u + j + 8, u2);
                    vst1q_f32(dst_u + j + 12, u3);
                }
#endif
                for (; j < V_INTER; j++) {
                    float g = src_g[j];
                    if (g < glo)
                        g = glo;
                    else if (g > ghi)
                        g = ghi;
                    float u = src_u[j];
                    if (u < ulo)
                        u = ulo;
                    else if (u > uhi)
                        u = uhi;
                    dst_g[j] = g;
                    dst_u[j] = u;
                }
            }
        }

        gelu_tanh_mul_fp32(gate, up, n_patches * V_INTER, gate);

        clamp_fp32(gate, n_patches * V_INTER, L->down_clip.in_min, L->down_clip.in_max);
        linear_fp32(gate, L->down_proj, nullptr, n_patches, V_INTER, VTH, proj);
        clamp_fp32(proj, n_patches * VTH, L->down_clip.out_min, L->down_clip.out_max);

        rmsnorm_fp32(proj, L->post_feedforward_layernorm, n_patches, VTH, V_EPS, proj);
        add_fp32(resid, proj, n_patches * VTH, hidden_out);

        /* Optional per-layer debug dump. */
        char nm[32];
        snprintf(nm, sizeof nm, "layer%02d", li);
        dbg_dump(nm, hidden_out, n_patches * VTH);
    }

    safe_free((void **) &scaled);
    safe_free((void **) &resid);
    safe_free((void **) &normed);
    safe_free((void **) &q);
    safe_free((void **) &k);
    safe_free((void **) &vbuf);
    safe_free((void **) &attn);
    safe_free((void **) &proj);
    safe_free((void **) &gate);
    safe_free((void **) &up);
    safe_free((void **) &gate_up_combined);
    safe_free((void **) &qkv_combined);
    return true;

fail:
    if (scaled)
        safe_free((void **) &scaled);
    if (resid)
        safe_free((void **) &resid);
    if (normed)
        safe_free((void **) &normed);
    if (q)
        safe_free((void **) &q);
    if (k)
        safe_free((void **) &k);
    if (vbuf)
        safe_free((void **) &vbuf);
    if (attn)
        safe_free((void **) &attn);
    if (proj)
        safe_free((void **) &proj);
    if (gate)
        safe_free((void **) &gate);
    if (up)
        safe_free((void **) &up);
    if (gate_up_combined)
        safe_free((void **) &gate_up_combined);
    if (qkv_combined)
        safe_free((void **) &qkv_combined);
    return false;
}

/* Shared core: image plan → preprocess → tower → pool → projector for
 * ONE image. Used by both run_image (single-shot) and run_video (called
 * once per frame). max_soft caps the soft-token output for this single
 * image (280 for stills, 70 for video frames). */
static size_t run_image_internal(const struct VisionEncoder *v,
                                 const uint8_t              *rgb,
                                 size_t                      height,
                                 size_t                      width,
                                 size_t                      max_soft,
                                 float                      *out) {
    struct image_plan plan;
    if (!image_pipeline_plan(height, width, max_soft, &plan))
        return 0;
    const size_t n_patches = plan.grid_h * plan.grid_w;
    const size_t n_soft    = plan.soft_tokens;
    const size_t patch_px  = VISION_PATCH_SIZE * VISION_PATCH_SIZE * 3;

    float   *patches   = heap_alloc_array_aligned(float, n_patches *patch_px);
    int32_t *positions = heap_alloc_array_aligned(int32_t, n_patches * 2);
    float   *tower     = heap_alloc_array_aligned(float, n_patches *VTH);
    float   *pooled    = heap_alloc_array_aligned(float, n_soft *VTH);
    float   *normed    = heap_alloc_array_aligned(float, n_soft *VTH);

    if (!patches || !positions || !tower || !pooled || !normed)
        goto fail;

    if (!image_pipeline_preprocess(rgb, &plan, patches))
        goto fail;
    image_pipeline_position_ids(&plan, positions);
    if (!vision_encoder_run_tower(v, patches, positions, n_patches, tower)) {
        goto fail;
    }
    avgpool2d_k3_fp32(tower, pooled, plan.grid_h, plan.grid_w, VTH);
    {
        const float  scale = sqrtf((float) VTH);
        const size_t N     = n_soft * VTH;
        for (size_t i = 0; i < N; i++)
            pooled[i] *= scale;
    }
    dbg_dump("pool_out", pooled, n_soft * VTH);
    rmsnorm_fp32(pooled, nullptr, n_soft, VTH, V_EPS, normed);
    linear_fp32(normed, v->projector_weight, nullptr, n_soft, VTH, v->soft_dim, out);
    dbg_dump("soft_tokens", out, n_soft * v->soft_dim);

    safe_free((void **) &patches);
    safe_free((void **) &positions);
    safe_free((void **) &tower);
    safe_free((void **) &pooled);
    safe_free((void **) &normed);
    return n_soft;

fail:
    if (patches)
        safe_free((void **) &patches);
    if (positions)
        safe_free((void **) &positions);
    if (tower)
        safe_free((void **) &tower);
    if (pooled)
        safe_free((void **) &pooled);
    if (normed)
        safe_free((void **) &normed);
    return 0;
}

size_t vision_encoder_run_image(const struct VisionEncoder *v,
                                const uint8_t              *rgb,
                                size_t                      height,
                                size_t                      width,
                                float                      *out) {
    if (v == nullptr || rgb == nullptr || out == nullptr || height == 0 || width == 0 ||
        height > IMAGE_PIPELINE_MAX_DIM || width > IMAGE_PIPELINE_MAX_DIM) {
        return 0;
    }
    return run_image_internal(v, rgb, height, width, VISION_SOFT_TOKENS_PER_IMAGE, out);
}

/* Per HF (modeling_gemma4.py:2437), each video frame is an independent
 * batch element of the same vision tower. Sequential in C is semantically
 * identical — same per-frame plan / preprocess / tower / pool / projector
 * chain as a still image, just with max_soft=70 per frame so 32 frames
 * fit in a reasonable LM context. */
size_t vision_encoder_run_video(const struct VisionEncoder *v,
                                const uint8_t              *frames,
                                size_t                      n_frames,
                                size_t                      height,
                                size_t                      width,
                                float                      *out) {
    if (v == nullptr || frames == nullptr || out == nullptr || n_frames == 0 || height == 0 ||
        width == 0 || height > IMAGE_PIPELINE_MAX_DIM || width > IMAGE_PIPELINE_MAX_DIM) {
        return 0;
    }
    const size_t per_frame_soft = VISION_SOFT_TOKENS_PER_VIDEO_FRAME;
    /* frame_bytes and the whole-clip extent, checked: the loop below walks
     * `frames` in frame_bytes strides, and both the per-frame product and
     * n_frames * frame_bytes come from the caller. */
    size_t frame_px    = 0;
    size_t frame_bytes = 0;
    size_t clip_bytes  = 0;
    if (ckd_mul(&frame_px, height, width) || ckd_mul(&frame_bytes, frame_px, 3u) ||
        ckd_mul(&clip_bytes, frame_bytes, n_frames)) {
        return 0;
    }
    size_t total_soft = 0;
    for (size_t f = 0; f < n_frames; f++) {
        size_t got = run_image_internal(v,
                                        frames + f * frame_bytes,
                                        height,
                                        width,
                                        per_frame_soft,
                                        out + total_soft * v->soft_dim);
        if (got == 0)
            return 0;
        total_soft += got;
    }
    return total_soft;
}
