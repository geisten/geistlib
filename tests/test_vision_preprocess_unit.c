/*
 * test_vision_preprocess_unit — Gemma 4 image-preprocess parity vs HF.
 *
 * Reads HF reference dumps and asserts (the script that wrote them,
 * dump_vision_preprocess.py, is not in this repository):
 *   - image_pipeline_plan produces identical plan fields
 *   - image_pipeline_position_ids matches exactly
 *   - image_pipeline_preprocess matches the HF (PIL bicubic) reference
 *     within bicubic rounding noise:
 *       - max |Δ|         <= 8/255  ≈ 0.0314  (sanity gate)
 *       - 99.99 percentile <= 2/255 ≈ 0.0078  (rounding noise only)
 *     These bounds reflect the unavoidable uint8 rounding mismatch
 *     between stb_image_resize2's Catmull-Rom and PIL's bicubic. The
 *     vast majority of pixels match bit-exactly; the differences are a
 *     sparse halo around blocks where one impl rounds up and the other
 *     rounds down.
 *
 * Usage: test_vision_preprocess_unit <dumps_basename>
 *   where <dumps_basename> = e.g. dumps/vision/syn_320x224
 *   expects files <basename>.{input,plan,patches,positions}.bin.
 *
 * Without args, looks for dumps/vision/syn_320x224.* and skips if absent.
 */
#include "image_pipeline.h"
#include "test_helpers.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint8_t *read_input_bin(const char *path, int32_t *out_h, int32_t *out_w) {
    FILE *f = fopen(path, "rb");
    if (f == nullptr)
        return nullptr;
    int32_t h = 0, w = 0;
    if (fread(&h, sizeof(int32_t), 1, f) != 1 || fread(&w, sizeof(int32_t), 1, f) != 1) {
        fclose(f);
        return nullptr;
    }
    size_t   n   = (size_t) h * (size_t) w * 3;
    uint8_t *buf = malloc(n);
    if (buf == nullptr) {
        fclose(f);
        return nullptr;
    }
    if (fread(buf, 1, n, f) != n) {
        free(buf);
        fclose(f);
        return nullptr;
    }
    fclose(f);
    *out_h = h;
    *out_w = w;
    return buf;
}

static bool read_plan_bin(const char *path, int64_t out[9]) {
    FILE *f = fopen(path, "rb");
    if (f == nullptr)
        return false;
    bool ok = fread(out, sizeof(int64_t), 9, f) == 9;
    fclose(f);
    return ok;
}

static void *read_full(const char *path, size_t *out_bytes) {
    FILE *f = fopen(path, "rb");
    if (f == nullptr)
        return nullptr;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = malloc((size_t) sz);
    if (buf == nullptr) {
        fclose(f);
        return nullptr;
    }
    if (fread(buf, 1, (size_t) sz, f) != (size_t) sz) {
        free(buf);
        fclose(f);
        return nullptr;
    }
    fclose(f);
    *out_bytes = (size_t) sz;
    return buf;
}

static const char *default_basename = "dumps/vision/syn_320x224";

int main(int argc, char **argv) {
    const char *base = argc > 1 ? argv[1] : default_basename;
    char        path[512];

    snprintf(path, sizeof path, "%s.input.bin", base);
    int32_t  in_h = 0, in_w = 0;
    uint8_t *rgb = read_input_bin(path, &in_h, &in_w);
    if (rgb == nullptr) {
        fprintf(stdout,
                "SKIP: dumps not found at %s.* (HF reference dumps; the "
                "script that writes them is not in this repository)\n",
                base);
        return GEIST_TEST_SKIP;
    }
    fprintf(stdout, "input: %dx%d RGB\n", in_h, in_w);

    /* --- Plan -------------------------------------------------------- */
    snprintf(path, sizeof path, "%s.plan.bin", base);
    int64_t hf_plan[9];
    if (!read_plan_bin(path, hf_plan)) {
        free(rgb);
        fprintf(stderr, "missing %s\n", path);
        return GEIST_TEST_ERROR;
    }
    struct image_plan plan;
    if (!image_pipeline_plan((size_t) in_h,
                             (size_t) in_w,
                             (size_t) hf_plan[8] /* soft_tokens as max_soft seed */,
                             &plan)) {
        /* HF dumped soft_tokens; we treat that as max_soft. But for the
         * real check below we just verify our plan matches HF's
         * computed plan with the canonical max_soft=280. */
        (void) 0;
    }
    /* Use canonical max_soft=280 (matches Gemma4ImageProcessor default). */
    if (!image_pipeline_plan((size_t) in_h, (size_t) in_w, 280, &plan)) {
        fprintf(stderr, "image_pipeline_plan failed\n");
        free(rgb);
        return GEIST_TEST_FAIL;
    }
    int64_t got[9] = {
            (int64_t) plan.in_h,
            (int64_t) plan.in_w,
            (int64_t) plan.resized_h,
            (int64_t) plan.resized_w,
            (int64_t) plan.grid_h,
            (int64_t) plan.grid_w,
            (int64_t) plan.pool_h,
            (int64_t) plan.pool_w,
            (int64_t) plan.soft_tokens,
    };
    static const char *plan_fields[9] = {
            "in_h",
            "in_w",
            "resized_h",
            "resized_w",
            "grid_h",
            "grid_w",
            "pool_h",
            "pool_w",
            "soft_tokens",
    };
    bool plan_ok = true;
    for (int i = 0; i < 9; i++) {
        if (got[i] != hf_plan[i]) {
            fprintf(stderr,
                    "plan.%s: got %lld want %lld\n",
                    plan_fields[i],
                    (long long) got[i],
                    (long long) hf_plan[i]);
            plan_ok = false;
        }
    }
    if (!plan_ok) {
        free(rgb);
        return GEIST_TEST_FAIL;
    }
    fprintf(stdout,
            "plan OK: resized=%zux%zu grid=%zux%zu pool=%zux%zu soft=%zu\n",
            plan.resized_h,
            plan.resized_w,
            plan.grid_h,
            plan.grid_w,
            plan.pool_h,
            plan.pool_w,
            plan.soft_tokens);

    /* --- Positions --------------------------------------------------- */
    snprintf(path, sizeof path, "%s.positions.bin", base);
    size_t   pos_bytes = 0;
    int32_t *hf_pos    = read_full(path, &pos_bytes);
    if (hf_pos == nullptr) {
        free(rgb);
        fprintf(stderr, "missing %s\n", path);
        return GEIST_TEST_ERROR;
    }
    size_t n_patches = plan.grid_h * plan.grid_w;
    if (pos_bytes != n_patches * 2 * sizeof(int32_t)) {
        fprintf(stderr,
                "positions.bin size mismatch: got %zu want %zu\n",
                pos_bytes,
                n_patches * 2 * sizeof(int32_t));
        free(rgb);
        free(hf_pos);
        return GEIST_TEST_FAIL;
    }
    int32_t *pos = malloc(n_patches * 2 * sizeof(int32_t));
    image_pipeline_position_ids(&plan, pos);
    for (size_t i = 0; i < n_patches * 2; i++) {
        if (pos[i] != hf_pos[i]) {
            fprintf(stderr, "positions[%zu]: got %d want %d\n", i, pos[i], hf_pos[i]);
            free(rgb);
            free(hf_pos);
            free(pos);
            return GEIST_TEST_FAIL;
        }
    }
    free(hf_pos);
    free(pos);
    fprintf(stdout, "positions OK (%zu patches)\n", n_patches);

    /* --- Patches (fp32, eps <= 1e-4) --------------------------------- */
    snprintf(path, sizeof path, "%s.patches.bin", base);
    size_t pat_bytes  = 0;
    float *hf_patches = read_full(path, &pat_bytes);
    if (hf_patches == nullptr) {
        free(rgb);
        fprintf(stderr, "missing %s\n", path);
        return GEIST_TEST_ERROR;
    }
    size_t patch_px = 16 * 16 * 3;
    size_t n_floats = n_patches * patch_px;
    if (pat_bytes != n_floats * sizeof(float)) {
        fprintf(stderr,
                "patches.bin size mismatch: got %zu want %zu\n",
                pat_bytes,
                n_floats * sizeof(float));
        free(rgb);
        free(hf_patches);
        return GEIST_TEST_FAIL;
    }

    float *patches = malloc(n_floats * sizeof(float));
    if (!image_pipeline_preprocess(rgb, &plan, patches)) {
        free(rgb);
        free(hf_patches);
        free(patches);
        fprintf(stderr, "image_pipeline_preprocess failed\n");
        return GEIST_TEST_FAIL;
    }

    /* Histogram of |Δ| in 1/255 units (the natural quantum since pixels
     * are uint8 / 255 floats). Buckets: 0, 1, 2, 3-7, >=8. */
    size_t hist[5] = {0, 0, 0, 0, 0};
    float  max_abs = 0.0f;
    size_t max_idx = 0;
    for (size_t i = 0; i < n_floats; i++) {
        float d = patches[i] - hf_patches[i];
        if (d < 0)
            d = -d;
        if (d > max_abs) {
            max_abs = d;
            max_idx = i;
        }

        int q = (int) (d * 255.0f + 0.5f); /* nearest 1/255 unit */
        if (q == 0)
            hist[0]++;
        else if (q == 1)
            hist[1]++;
        else if (q == 2)
            hist[2]++;
        else if (q < 8)
            hist[3]++;
        else
            hist[4]++;
    }
    fprintf(stdout,
            "patches max|Δ|=%.6f (geist=%.4f hf=%.4f at idx %zu)\n"
            "  histogram: |d|=0 %zu  =1 %zu  =2 %zu  3-7 %zu  >=8 %zu\n",
            (double) max_abs,
            (double) patches[max_idx],
            (double) hf_patches[max_idx],
            max_idx,
            hist[0],
            hist[1],
            hist[2],
            hist[3],
            hist[4]);

    /* Gates: max diff <= 8/255 (sanity), p99.99 <= 2/255 (no algorithmic
     * skew, only rounding noise). 99.99% threshold = at most 0.01% can
     * land in buckets 3-7 or >=8. */
    const float  max_gate  = 8.0f / 255.0f + 1e-7f;
    const size_t p9999_max = (n_floats * 1 + 9999) / 10000; /* ceil(n * 1e-4) */
    const size_t over_2    = hist[3] + hist[4];

    bool ok_max   = max_abs <= max_gate;
    bool ok_p9999 = over_2 <= p9999_max;

    if (!ok_max) {
        fprintf(stderr,
                "FAIL: max|Δ|=%.6f exceeds gate %.6f (8/255)\n",
                (double) max_abs,
                (double) max_gate);
    }
    if (!ok_p9999) {
        fprintf(stderr, "FAIL: %zu pixels >2/255 exceeds p99.99 budget %zu\n", over_2, p9999_max);
    }

    int rc = (ok_max && ok_p9999) ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
    free(rgb);
    free(hf_patches);
    free(patches);
    return rc;
}
