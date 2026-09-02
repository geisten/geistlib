#define _POSIX_C_SOURCE 200809L

/*
 * test_vision_tower_blocks_unit — per-layer parity test for the
 * Gemma 4 vision tower against HF dumps from dump_vision_tower_blocks.py.
 *
 * Sets GEIST_VISION_DUMP_DIR to a scratch directory so vision_encoder
 * writes per-block buffers; then compares each layer's output (plus
 * patch_embed_out) against the HF reference at max |Δ| <= 2e-2.
 *
 * The 2e-2 budget reflects fp32 accumulation noise over 16 ViT blocks
 * with OpenMP-parallelized attention — different sgemm thread schedules
 * give 1-2 ULP differences per attention call that compound across
 * blocks (mean diff stays ~2e-4, layer-15 max ~2e-2 on magnitudes
 * ~1000 = ~2e-5 relative). Anything significantly larger indicates
 * an algorithmic divergence (op ordering, RoPE sign, etc.).
 *
 * Usage: test_vision_tower_blocks_unit <vision_tower.safetensors>
 *                                       <dumps_basename>
 *   defaults: vision_bench/vision_tower.safetensors, dumps/vision/syn_320x224
 */
#define GEIST_INTERNAL_ARCH_LAYER
#include "vision_encoder.h"
#undef GEIST_INTERNAL_ARCH_LAYER

#include "test_helpers.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define N_PATCHES_SYN 2520
#define VTH 768

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

static int compare_layer(const char *name,
                         const char *geist_dir,
                         const char *hf_basename,
                         size_t      n_floats,
                         float       eps) {
    char hf_path[512], geist_path[512];
    snprintf(hf_path, sizeof hf_path, "%s.%s.bin", hf_basename, name);
    snprintf(geist_path, sizeof geist_path, "%s/%s.bin", geist_dir, name);

    size_t hf_b = 0, gi_b = 0;
    float *hf = read_full(hf_path, &hf_b);
    float *gi = read_full(geist_path, &gi_b);
    if (hf == nullptr) {
        fprintf(stdout, "  %-15s SKIP (missing HF dump %s)\n", name, hf_path);
        if (gi)
            free(gi);
        return 0; /* not a fail */
    }
    if (gi == nullptr) {
        fprintf(stderr, "  %-15s FAIL: missing geist dump %s\n", name, geist_path);
        free(hf);
        return 1;
    }
    if (hf_b != n_floats * sizeof(float) || gi_b != n_floats * sizeof(float)) {
        fprintf(stderr,
                "  %-15s FAIL: size mismatch hf=%zu geist=%zu want=%zu\n",
                name,
                hf_b,
                gi_b,
                n_floats * sizeof(float));
        free(hf);
        free(gi);
        return 1;
    }
    float  max_abs = 0.0f;
    size_t max_idx = 0;
    double sum_abs = 0.0;
    for (size_t i = 0; i < n_floats; i++) {
        float d = gi[i] - hf[i];
        if (d < 0)
            d = -d;
        sum_abs += d;
        if (d > max_abs) {
            max_abs = d;
            max_idx = i;
        }
    }
    double mean = sum_abs / (double) n_floats;
    fprintf(stdout,
            "  %-15s max|Δ|=%.6f  mean=%.6f  (geist=%.4f hf=%.4f at %zu)\n",
            name,
            (double) max_abs,
            mean,
            (double) gi[max_idx],
            (double) hf[max_idx],
            max_idx);
    free(hf);
    free(gi);
    return max_abs <= eps ? 0 : 1;
}

int main(int argc, char **argv) {
    const char *weights_path = argc > 1 ? argv[1] : "vision_bench/vision_tower.safetensors";
    const char *hf_basename  = argc > 2 ? argv[2] : "dumps/vision/syn_320x224";

    /* Skip if weights or dumps missing. */
    {
        FILE *f = fopen(weights_path, "rb");
        if (f == nullptr) {
            fprintf(stdout,
                    "SKIP: %s not found. Run tools/dump_vision_tower.py "
                    "to extract.\n",
                    weights_path);
            return GEIST_TEST_SKIP;
        }
        fclose(f);
    }
    {
        char p[512];
        snprintf(p, sizeof p, "%s.patches.bin", hf_basename);
        FILE *f = fopen(p, "rb");
        if (f == nullptr) {
            fprintf(stdout,
                    "SKIP: %s not found. Run "
                    "tools/dump_vision_preprocess.py first.\n",
                    p);
            return GEIST_TEST_SKIP;
        }
        fclose(f);
        /* HF block dumps are optional — if missing, the per-layer
         * comparisons skip individually and the test passes after the
         * tower runs without crashing. This lets us validate the load +
         * forward path locally before running tools/dump_vision_tower_blocks.py
         * (which requires torch). */
    }

    /* Read patches + positions. */
    char   path[512];
    size_t bytes = 0;
    snprintf(path, sizeof path, "%s.patches.bin", hf_basename);
    float *patches = read_full(path, &bytes);
    if (patches == nullptr || bytes != N_PATCHES_SYN * VTH * sizeof(float)) {
        fprintf(stderr, "bad patches file\n");
        return GEIST_TEST_ERROR;
    }
    snprintf(path, sizeof path, "%s.positions.bin", hf_basename);
    int32_t *positions = read_full(path, &bytes);
    if (positions == nullptr || bytes != N_PATCHES_SYN * 2 * sizeof(int32_t)) {
        fprintf(stderr, "bad positions file\n");
        return GEIST_TEST_ERROR;
    }

    /* Scratch dump dir. */
    const char *dump_dir = "build/test-tmp/vision_dump";
    mkdir("build/test-tmp", 0755);
    mkdir(dump_dir, 0755);
    setenv("GEIST_VISION_DUMP_DIR", dump_dir, 1);

    /* Load weights + run tower. */
    fprintf(stdout, "loading %s...\n", weights_path);
    struct VisionEncoder *enc = vision_encoder_create(weights_path);
    if (enc == nullptr) {
        free(patches);
        free(positions);
        return GEIST_TEST_FAIL;
    }

    float *hidden_out = malloc(N_PATCHES_SYN * VTH * sizeof(float));
    fprintf(stdout, "running tower forward (n=%d, %d layers)...\n", N_PATCHES_SYN, 16);
    bool ok = vision_encoder_run_tower(enc, N_PATCHES_SYN, patches, positions, hidden_out);
    vision_encoder_destroy(enc);
    free(patches);
    free(positions);
    free(hidden_out);
    if (!ok) {
        fprintf(stderr, "vision_encoder_run_tower failed\n");
        return GEIST_TEST_FAIL;
    }

    /* Per-stage diff. */
    const float eps   = 2e-2f;
    int         fails = 0;
    fprintf(stdout, "parity gate: max|Δ| <= %g\n", (double) eps);
    fails += compare_layer("patch_embed_out", dump_dir, hf_basename, N_PATCHES_SYN * VTH, eps);
    for (int li = 0; li < 16; li++) {
        char nm[32];
        snprintf(nm, sizeof nm, "layer%02d", li);
        fails += compare_layer(nm, dump_dir, hf_basename, N_PATCHES_SYN * VTH, eps);
    }
    /* `encoder_out` is identical to `layer15` (no final norm in the
     * encoder per modeling_gemma4.py:1024) — skip the duplicate check. */

    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
