#define _POSIX_C_SOURCE 200809L

/*
 * test_vision_e2e_unit — Gemma 4 vision end-to-end parity vs HF.
 *
 * Decodes the synthetic test image, runs the full geist pipeline
 *   image_pipeline_preprocess → vision_encoder_run_tower → pool → ×√768
 *   → RMSNorm-no-scale → projector
 * via geist_session_attach_image's underlying entry point
 * (vision_encoder_run_image), and compares the soft tokens + the
 * intermediate pool output against the HF dumps from
 * dump_vision_tower_blocks.py.
 *
 * Gates are relative because this test runs the WHOLE preprocessing
 * pipeline from the PNG (stb bicubic) vs HF's pre-resized dump (PIL
 * bicubic). That 1/255 per-pixel rounding noise compounds through 16
 * residual-stream layers and gets amplified by √768=27.7 in the pool
 * scale. Per-token cosine similarity stays > 0.999, so the relative
 * gates below catch real bugs while tolerating preprocessing rounding.
 *
 *   pool_out    : max|Δ| / max|hf| ≤ 0.08  (post √768 scale)
 *   soft_tokens : max|Δ| / max|hf| ≤ 0.15  (one more linear over 1536)
 *
 * Usage: test_vision_e2e_unit <weights> <dumps_basename> [<png_path>]
 *   defaults: vision_bench/vision_tower.safetensors,
 *             dumps/vision/syn_320x224, vision_bench/syn_320x224.png
 *
 * Without HF dumps, the per-stage diffs skip individually; the test
 * still validates that the full encoder runs without crashing and
 * produces the expected number of soft tokens.
 */
#define GEIST_INTERNAL_ARCH_LAYER
#include "vision_encoder.h"
#include <geist_util.h>
#undef GEIST_INTERNAL_ARCH_LAYER

#include "test_helpers.h"

#include "stb_image.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define VTH 768
#define V_SOFT 1536

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

static int compare_stage(const char *name,
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
        fprintf(stdout, "  %-12s SKIP (missing HF dump %s)\n", name, hf_path);
        if (gi)
            free(gi);
        return 0;
    }
    if (gi == nullptr) {
        fprintf(stderr, "  %-12s FAIL: geist dump missing %s\n", name, geist_path);
        free(hf);
        return 1;
    }
    if (hf_b != n_floats * sizeof(float) || gi_b != n_floats * sizeof(float)) {
        fprintf(stderr,
                "  %-12s FAIL: size mismatch hf=%zu geist=%zu want=%zu\n",
                name,
                hf_b,
                gi_b,
                n_floats * sizeof(float));
        free(hf);
        free(gi);
        return 1;
    }
    float  max_abs = 0.0f, max_hf = 0.0f;
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
        float ah = hf[i] < 0 ? -hf[i] : hf[i];
        if (ah > max_hf)
            max_hf = ah;
    }
    double mean = sum_abs / (double) n_floats;
    float  rel  = max_hf > 0 ? max_abs / max_hf : max_abs;
    fprintf(stdout,
            "  %-12s max|Δ|=%.4f  rel=%.4f  mean=%.4f  "
            "(geist=%.2f hf=%.2f at %zu, |hf|_max=%.1f)\n",
            name,
            (double) max_abs,
            (double) rel,
            mean,
            (double) gi[max_idx],
            (double) hf[max_idx],
            max_idx,
            (double) max_hf);
    free(hf);
    free(gi);
    return rel <= eps ? 0 : 1;
}

int main(int argc, char **argv) {
    const char *weights_path = argc > 1 ? argv[1] : "vision_bench/vision_tower.safetensors";
    const char *hf_basename  = argc > 2 ? argv[2] : "dumps/vision/syn_320x224";
    const char *png_path     = argc > 3 ? argv[3] : "vision_bench/syn_320x224.png";

    {
        FILE *f = fopen(weights_path, "rb");
        if (f == nullptr) {
            fprintf(stdout, "SKIP: %s not found. Run extract_vision_tower.\n", weights_path);
            return GEIST_TEST_SKIP;
        }
        fclose(f);
        f = fopen(png_path, "rb");
        if (f == nullptr) {
            fprintf(stdout, "SKIP: %s not found. Run dump_vision_preprocess.py.\n", png_path);
            return GEIST_TEST_SKIP;
        }
        fclose(f);
    }

    /* Decode PNG via stb_image. */
    int      w = 0, h = 0, c = 0;
    uint8_t *rgb = stbi_load(png_path, &w, &h, &c, 3);
    if (rgb == nullptr) {
        fprintf(stderr, "stb_image: failed to load %s\n", png_path);
        return GEIST_TEST_ERROR;
    }
    fprintf(stdout, "loaded %s: %dx%d %dch\n", png_path, h, w, c);

    /* Scratch dump dir. */
    const char *dump_dir = "build/test-tmp/vision_e2e";
    mkdir("build/test-tmp", 0755);
    mkdir(dump_dir, 0755);
    setenv("GEIST_VISION_DUMP_DIR", dump_dir, 1);

    /* Load weights. */
    fprintf(stdout, "loading %s...\n", weights_path);
    struct VisionEncoder *enc = vision_encoder_create(weights_path);
    if (enc == nullptr) {
        stbi_image_free(rgb);
        return GEIST_TEST_FAIL;
    }

    /* Run end-to-end. */
    float *soft = malloc(VISION_SOFT_TOKENS_PER_IMAGE * V_SOFT * sizeof(float));
    fprintf(stdout, "running vision_encoder_run_image (h=%d w=%d)...\n", h, w);
    size_t n_soft = vision_encoder_run_image(enc, (size_t) h, (size_t) w, rgb, soft);
    vision_encoder_destroy(enc);
    stbi_image_free(rgb);

    if (n_soft == 0) {
        free(soft);
        fprintf(stderr, "vision_encoder_run_image returned 0\n");
        return GEIST_TEST_FAIL;
    }
    fprintf(stdout,
            "produced %zu soft tokens (expected %d)\n",
            n_soft,
            VISION_SOFT_TOKENS_PER_IMAGE);
    if (n_soft != VISION_SOFT_TOKENS_PER_IMAGE) {
        free(soft);
        fprintf(stderr, "wrong soft-token count\n");
        return GEIST_TEST_FAIL;
    }

    /* Compare against HF dumps with relative gates. */
    int fails = 0;
    fprintf(stdout, "parity gates: pool rel=0.08, soft_tokens rel=0.15\n");
    fails += compare_stage("pool_out", dump_dir, hf_basename, n_soft * VTH, 0.08f);
    fails += compare_stage("soft_tokens", dump_dir, hf_basename, n_soft * V_SOFT, 0.15f);

    free(soft);
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
