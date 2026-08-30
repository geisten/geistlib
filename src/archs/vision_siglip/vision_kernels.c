/*
 * vision_kernels — 2D split-axis RoPE, kernel-3 avg-pool, bidirectional
 * attention. All FP32.
 *
 * Reference for the 2D RoPE and attention semantics:
 * transformers/models/gemma4/modeling_gemma4.py
 *   - Gemma4VisionRotaryEmbedding.compute_default_rope_parameters
 *   - apply_multidimensional_rope (ndim=2)
 *   - eager_attention_forward (no causal mask in the vision path)
 */
#include "vision_kernels.h"
#include <geist_types.h>

#include "heap.h"

#include <math.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

/* Dense fp32 matmul via the geist_gemm facade (Accelerate / OpenBLAS / native). */
#include "geist_gemm.h"

void avgpool2d_k3_fp32(const float *in, float *out, size_t grid_h, size_t grid_w, size_t hidden) {
    const size_t pool_h = grid_h / 3;
    const size_t pool_w = grid_w / 3;
    const float  inv9   = 1.0f / 9.0f;
    for (size_t py = 0; py < pool_h; py++) {
        for (size_t px = 0; px < pool_w; px++) {
            float *o = out + (py * pool_w + px) * hidden;
            memset(o, 0, hidden * sizeof(float));
            for (size_t dy = 0; dy < 3; dy++) {
                for (size_t dx = 0; dx < 3; dx++) {
                    const float *p = in + ((py * 3 + dy) * grid_w + (px * 3 + dx)) * hidden;
                    for (size_t k = 0; k < hidden; k++) {
                        o[k] += p[k];
                    }
                }
            }
            for (size_t k = 0; k < hidden; k++)
                o[k] *= inv9;
        }
    }
}

void rope_2d_split_fp32(float         *x,
                        const int32_t *positions,
                        size_t         n_tokens,
                        size_t         n_heads,
                        size_t         head_dim,
                        float          theta) {
    /* Gemma 4 vision: ndim=2, half = head_dim/2, pair = half/2.
     * compute_default_rope_parameters:
     *   spatial_dim = head_dim / 2
     *   inv_freq[k] = 1 / (theta^(2k / spatial_dim))  for k in 0..spatial_dim/2-1
     * In forward: freqs = inv_freq[k] * pos for each (k, pos); emb =
     * concat([freqs, freqs]) so emb has length spatial_dim = half.
     * apply_multidimensional_rope splits x into two halves, applies the
     * standard concat-style RoPE within each half:
     *   y[k]      = x[k]      * cos[k] - x[k+pair] * sin[k]
     *   y[k+pair] = x[k+pair] * cos[k] + x[k]      * sin[k]
     *
     * Position-cache optimization: For an image patch grid, x_pos and
     * y_pos take only grid_w / grid_h unique values respectively.
     * Precomputing cos/sin tables per unique position eliminates ~75x
     * of the cosf/sinf calls vs the naive per-token-per-head loop.
     */
    const size_t half = head_dim / 2;
    const size_t pair = half / 2;

    /* Find unique-position bounds (handle padding -1 as 0). */
    int32_t max_px = 0, max_py = 0;
    for (size_t t = 0; t < n_tokens; t++) {
        int32_t px = positions[t * 2 + 0];
        if (px < 0)
            px = 0;
        int32_t py = positions[t * 2 + 1];
        if (py < 0)
            py = 0;
        if (px > max_px)
            max_px = px;
        if (py > max_py)
            max_py = py;
    }
    const size_t nx = (size_t) max_px + 1;
    const size_t ny = (size_t) max_py + 1;

    float *inv_freq = heap_alloc_array_aligned(float, pair);
    float *cos_x    = heap_alloc_array_aligned(float, nx *pair);
    float *sin_x    = heap_alloc_array_aligned(float, nx *pair);
    float *cos_y    = heap_alloc_array_aligned(float, ny *pair);
    float *sin_y    = heap_alloc_array_aligned(float, ny *pair);
    if (!inv_freq || !cos_x || !sin_x || !cos_y || !sin_y) {
        if (inv_freq)
            safe_free((void **) &inv_freq);
        if (cos_x)
            safe_free((void **) &cos_x);
        if (sin_x)
            safe_free((void **) &sin_x);
        if (cos_y)
            safe_free((void **) &cos_y);
        if (sin_y)
            safe_free((void **) &sin_y);
        return;
    }
    for (size_t k = 0; k < pair; k++) {
        inv_freq[k] = 1.0f / powf(theta, (float) (2 * k) / (float) half);
    }

    /* Build (cos, sin) tables for all unique x/y positions. */
    for (size_t p = 0; p < nx; p++) {
        const float fp = (float) p;
        for (size_t k = 0; k < pair; k++) {
            float a             = inv_freq[k] * fp;
            cos_x[p * pair + k] = cosf(a);
            sin_x[p * pair + k] = sinf(a);
        }
    }
    for (size_t p = 0; p < ny; p++) {
        const float fp = (float) p;
        for (size_t k = 0; k < pair; k++) {
            float a             = inv_freq[k] * fp;
            cos_y[p * pair + k] = cosf(a);
            sin_y[p * pair + k] = sinf(a);
        }
    }

    /* Apply: each (token, head) gets a 2D-split rotation looked up from
     * the per-axis tables. */
    for (size_t t = 0; t < n_tokens; t++) {
        const int32_t px_raw = positions[t * 2 + 0];
        const int32_t py_raw = positions[t * 2 + 1];
        const size_t  px     = (size_t) (px_raw < 0 ? 0 : px_raw);
        const size_t  py     = (size_t) (py_raw < 0 ? 0 : py_raw);
        const float  *cx     = cos_x + px * pair;
        const float  *sx     = sin_x + px * pair;
        const float  *cy     = cos_y + py * pair;
        const float  *sy     = sin_y + py * pair;

        for (size_t h = 0; h < n_heads; h++) {
            float *xh = x + (t * n_heads + h) * head_dim;
            /* X-axis half: indices [0, half). Pair (k, k+pair). */
            for (size_t k = 0; k < pair; k++) {
                const float c = cx[k];
                const float s = sx[k];
                const float a = xh[k];
                const float b = xh[k + pair];
                xh[k]         = a * c - b * s;
                xh[k + pair]  = b * c + a * s;
            }
            /* Y-axis half: indices [half, head_dim). */
            for (size_t k = 0; k < pair; k++) {
                const float c       = cy[k];
                const float s       = sy[k];
                const float a       = xh[half + k];
                const float b       = xh[half + k + pair];
                xh[half + k]        = a * c - b * s;
                xh[half + k + pair] = b * c + a * s;
            }
        }
    }

    safe_free((void **) &inv_freq);
    safe_free((void **) &cos_x);
    safe_free((void **) &sin_x);
    safe_free((void **) &cos_y);
    safe_free((void **) &sin_y);
}

void vision_attention_bidir_fp32(const float *q,
                                 const float *k,
                                 const float *v,
                                 size_t       n_tokens,
                                 size_t       n_heads,
                                 size_t       head_dim,
                                 float       *out) {
    /* Per-head attention with OpenMP parallelism over heads.
     *
     * Q/K/V/O live in interleaved (n, n_heads, head_dim) layout — we
     * use BLAS strides (lda = n_heads * head_dim) to read per-head
     * tiles without copying. Each head's QK^T + softmax + AV runs
     * concurrently in a thread; per-thread scores scratch is
     * allocated on the heap once and indexed by omp_get_thread_num()
     * to avoid per-iteration malloc.
     *
     * Threading model: OpenMP outer parallel over n_heads=12; we
     * disable Accelerate's internal threading per-call via
     * BLAS_THREADING=SINGLE-THREADED if possible — on macOS we rely
     * on Accelerate's small-problem auto-detect to stay single-thread
     * for K=64. With 8 cores and 12 heads, expect 2-4x speedup over
     * the sequential per-head loop.
     */
    const float scale     = 1.0f;
    const int   hd_stride = (int) (n_heads * head_dim);

#if defined(_OPENMP)
    const int n_threads = omp_get_max_threads();
#else
    const int n_threads = 1;
#endif
    const size_t per_thread_scores = n_tokens * n_tokens;
    float *scores_pool = heap_alloc_array_aligned(float, (size_t) n_threads *per_thread_scores);
    if (scores_pool == nullptr)
        return;

    /* GEIST_FAST_TANH=1 enables vForce vvexpf for softmax — ~10x faster
     * than scalar expf loop, but introduces 1-2 ULP per element drift.
     * Reuses the GEIST_FAST_TANH env flag (same precision/perf trade). */
    /* Racy-benign first-use cache: several sessions can reach it at once.
     * Both would compute the same answer, but a plain int read/written
     * concurrently is still a data race (and TSan says so). Relaxed
     * atomics make it defined at no cost — the load is a plain load on
     * every target. */
    static _Atomic int fast_expf = -1;
    int                fe        = atomic_load_explicit(&fast_expf, memory_order_relaxed);
    if (fe < 0) {
        const char *s = getenv("GEIST_FAST_TANH");
        fe            = (s != nullptr && s[0] == '1') ? 1 : 0;
        atomic_store_explicit(&fast_expf, fe, memory_order_relaxed);
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t h = 0; h < n_heads; h++) {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        float *scores = scores_pool + (size_t) tid * per_thread_scores;

        const float *qh = q + h * head_dim;
        const float *kh = k + h * head_dim;
        const float *vh = v + h * head_dim;
        float       *oh = out + h * head_dim;

        /* scores = qh @ kh^T  → (n, n). Strided lda/ldb to read from
         * the interleaved layout in-place. */
        geist_sgemm(GEIST_OP_N,
                    GEIST_OP_T,
                    (int) n_tokens,
                    (int) n_tokens,
                    (int) head_dim,
                    scale,
                    qh,
                    hd_stride,
                    kh,
                    hd_stride,
                    0.0f,
                    scores,
                    (int) n_tokens);

        /* Row-wise softmax (fp32). Dominates attention cost (~70%);
         * vvexpf when fast-path is enabled brings it from ~1100 ms
         * wall down to ~150 ms wall. Branch hoisted out of the inner
         * loop so the default path stays bit-for-bit identical to the
         * pre-vvexpf code. */
#if defined(__APPLE__)
        if (fe) {
            extern void vvexpf(float *y, const float *x, const int *n_int);
            int         n_int = (int) n_tokens;
            for (size_t i = 0; i < n_tokens; i++) {
                float *row  = scores + i * n_tokens;
                float  maxv = row[0];
                for (size_t j = 1; j < n_tokens; j++) {
                    if (row[j] > maxv)
                        maxv = row[j];
                }
                for (size_t j = 0; j < n_tokens; j++)
                    row[j] -= maxv;
                vvexpf(row, row, &n_int);
                float sum = 0.0f;
                for (size_t j = 0; j < n_tokens; j++)
                    sum += row[j];
                float inv = 1.0f / sum;
                for (size_t j = 0; j < n_tokens; j++)
                    row[j] *= inv;
            }
        } else
#endif
        {
            for (size_t i = 0; i < n_tokens; i++) {
                float *row  = scores + i * n_tokens;
                float  maxv = row[0];
                for (size_t j = 1; j < n_tokens; j++) {
                    if (row[j] > maxv)
                        maxv = row[j];
                }
                float sum = 0.0f;
                for (size_t j = 0; j < n_tokens; j++) {
                    row[j] = expf(row[j] - maxv);
                    sum += row[j];
                }
                float inv = 1.0f / sum;
                for (size_t j = 0; j < n_tokens; j++)
                    row[j] *= inv;
            }
        }

        /* oh = scores @ vh  → (n, head_dim) interleaved. */
        geist_sgemm(GEIST_OP_N,
                    GEIST_OP_N,
                    (int) n_tokens,
                    (int) head_dim,
                    (int) n_tokens,
                    1.0f,
                    scores,
                    (int) n_tokens,
                    vh,
                    hd_stride,
                    0.0f,
                    oh,
                    hd_stride);
    }

    safe_free((void **) &scores_pool);
}
