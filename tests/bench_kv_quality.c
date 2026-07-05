/*
 * bench_kv_quality — KV-quant quality A/B via single-pass top-1 prediction.
 *
 * Tokenizes a fixed paragraph, then calls verify_forward on the WHOLE
 * sequence in one batched pass. verify_forward produces the model's
 * argmax at each position simultaneously, so we get N-1 predictions for
 * the cost of one forward pass.
 *
 * Reports top-1 accuracy: fraction of positions where the model's argmax
 * matches the actual next token in the text. A monotonic scalar quality
 * signal sensitive to KV-cache quantization noise.
 *
 * Runs all four KV modes in one pass (FP32 / INT8 / INT8+ROT / KIVI) and
 * prints a comparison table plus the fraction of the INT8→KIVI quality gap
 * that the issue-#61 Hadamard rotation recovers. The KV mode is resolved
 * from GEIST_KV_* env vars at model-load time, so each mode reloads the
 * model with those vars set (weights re-mmap, cheap). Override a single
 * mode the old way by exporting GEIST_KV_* before the run — it is honored
 * as the process default but the sweep sets its own per iteration.
 *
 * Uses arch_ops->verify_forward via geist_model_internal_arch_meta — a
 * test-only accessor that bypasses the public session API to keep the
 * harness independent of the live spec_step plumbing. SKIPs if no GGUF
 * or tokenizer is reachable.
 */
#define GEIST_INTERNAL_ENGINE_LAYER
#define GEIST_INTERNAL_ARCH_LAYER

#include "test_helpers.h"

#include "src/engine/model.h"
#include "src/engine/sp_bpe_tokenizer.h"
#include "src/engine/gguf_tokenizer.h"
#include "src/archs/transformer/arch.h" /* geist_arch_transformer descriptor */

#include <geist.h>
#include <geist_backend.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ~400 tokens so the KIVI drained 2-bit path is actually exercised
 * (KIVI_K_GROUP_SIZE=128: a drain triggers around token 128 and again
 * around 256, giving us ≥256 tokens of *drained 2-bit* attention reads
 * mixed with the residual). */
static const char *DEFAULT_TEXT =
        "Raspberry Pi is a series of small single-board computers developed "
        "in the United Kingdom by the Raspberry Pi Foundation. The original "
        "model became more popular than anticipated, selling outside its "
        "target market for uses such as robotics. It is widely used in many "
        "areas, such as for weather monitoring, because of its low cost, "
        "modularity, and open design. After the release of the second board "
        "type, the Raspberry Pi Foundation set up a new entity, Raspberry "
        "Pi Trading, to handle commercial operations while the original "
        "Foundation focused on educational outreach. The boards run a "
        "Debian-based operating system called Raspberry Pi OS as the "
        "official supported distribution. The Pi 5 was released in 2023 "
        "with a Broadcom BCM2712 system-on-chip featuring a quad-core "
        "Arm Cortex-A76 processor clocked at 2.4 gigahertz, paired with "
        "an 800 megahertz VideoCore VII graphics processor. The board "
        "includes options for 4 or 8 gigabytes of LPDDR4X memory, two "
        "micro-HDMI ports supporting dual 4K displays, two USB 3.0 ports, "
        "two USB 2.0 ports, gigabit Ethernet, and Wi-Fi 5 plus Bluetooth "
        "5.0 wireless connectivity. A new dedicated power button replaces "
        "the previous practice of pulling the power cable to shut down. "
        "An internal real-time clock with battery backup keeps the system "
        "time accurate across reboots and unpowered intervals. The board "
        "also exposes a PCI Express interface via a flat ribbon connector "
        "for high-speed peripherals such as NVMe storage.";

/* One KV mode: which GEIST_KV_* vars to set before the model load that
 * resolves it. A nullptr field means unset that var. */
struct kv_mode_cfg {
    const char *label;
    const char *int8; /* GEIST_KV_INT8 */
    const char *kivi; /* GEIST_KV_KIVI */
    const char *rot;  /* GEIST_KV_ROT  */
};

static const struct kv_mode_cfg MODES[] = {
        {"FP32", "0", nullptr, nullptr},
        {"INT8", "1", nullptr, "0"},
        {"INT8+ROT", "1", nullptr, "1"},
        {"KIVI", nullptr, "1", nullptr},
};

static void set_or_unset(const char *name, const char *val) {
    if (val != nullptr)
        setenv(name, val, 1);
    else
        unsetenv(name);
}

/* Load the model under the current GEIST_KV_* env, run top-1 next-token
 * prediction over `ids`, and return accuracy. Returns -1.0 on error. The
 * caller owns `ids` (tokenized once and reused across modes). */
static double run_one(const char           *model_path,
                      struct geist_backend *be,
                      const uint32_t       *ids,
                      size_t                n_ids,
                      size_t               *n_eval_out) {
    struct geist_model *model = nullptr;
    enum geist_status   s     = geist_model_load(model_path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr,
                "model_load: %s — %s\n",
                geist_status_to_string(s),
                geist_last_create_error());
        return -1.0;
    }

    void                                *arch_meta = geist_model_internal_arch_meta(model);
    const struct geist_arch_ops_decoder *ops       = &geist_arch_transformer;
    if (ops->state_reset == nullptr || ops->verify_forward == nullptr ||
        ops->kv_truncate == nullptr) {
        fprintf(stderr, "arch lacks state_reset / verify_forward / kv_truncate\n");
        geist_model_destroy(model);
        return -1.0;
    }

    ops->state_reset(arch_meta);
    ops->kv_truncate(arch_meta, 0);

    const size_t   k     = n_ids - 1;
    geist_token_t *preds = (geist_token_t *) calloc(k, sizeof *preds);
    if (preds == nullptr) {
        geist_model_destroy(model);
        return -1.0;
    }

    /* Chunk because verify_forward caps at m_max (64). kv_truncate after
     * each chunk forces a KIVI residual drain so the 2-bit path is
     * exercised; harmless for the other modes. */
    const size_t M_MAX      = 64;
    size_t       kv_len_acc = 0;
    for (size_t off = 0; off < k; off += M_MAX) {
        const size_t      chunk = (k - off > M_MAX) ? M_MAX : (k - off);
        enum geist_status vs    = ops->verify_forward(
                arch_meta, chunk, (const geist_token_t *) ids + off, preds + off);
        if (vs != GEIST_OK) {
            fprintf(stderr, "verify_forward(@%zu): %s\n", off, geist_status_to_string(vs));
            free(preds);
            geist_model_destroy(model);
            return -1.0;
        }
        kv_len_acc += chunk;
        ops->kv_truncate(arch_meta, kv_len_acc);
    }

    size_t n_correct = 0;
    for (size_t i = 0; i < k - 1; i++) {
        if ((uint32_t) preds[i] == ids[i + 1])
            n_correct++;
    }
    const size_t n_eval = k - 1;
    free(preds);
    geist_model_destroy(model);
    if (n_eval_out != nullptr)
        *n_eval_out = n_eval;
    return n_eval > 0 ? (double) n_correct / (double) n_eval : 0.0;
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : geist_test_find_gguf();
    GEIST_SKIP_IF(model_path == nullptr, "no GGUF model found — pass path or set GEIST_GGUF_PATH");

    const char *text = (argc > 2) ? argv[2] : DEFAULT_TEXT;

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    /* Tokenize once (mode-independent) via a throwaway load. */
    struct geist_model *model = nullptr;
    s                         = geist_model_load(model_path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr, "model_load(%s): %s\n", model_path, geist_status_to_string(s));
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    /* Try the sp_bpe tokenizer (external tokenizer.bin models) first, then
     * the GGUF-embedded tokenizer (Gemma/BitNet unigram, Qwen bpe). */
    uint32_t                *ids           = nullptr;
    size_t                   n_ids         = 0;
    bool                     enc_ok        = false;
    bool                     had_tokenizer = false;
    struct sp_bpe_tokenizer *tok           = geist_model_internal_tokenizer(model);
    if (tok != nullptr) {
        had_tokenizer = true;
        enc_ok        = sp_bpe_tokenizer_encode(tok, text, &ids, &n_ids);
    } else {
        struct gguf_tokenizer *gtok = geist_model_internal_gguf_tokenizer(model);
        if (gtok != nullptr) {
            had_tokenizer    = true;
            const size_t cap = strlen(text) + 16;
            int32_t     *tmp = (int32_t *) malloc(cap * sizeof(int32_t));
            if (tmp != nullptr && gguf_tokenizer_encode(gtok, text, tmp, cap, &n_ids)) {
                ids = (uint32_t *) malloc(n_ids * sizeof(uint32_t));
                if (ids != nullptr) {
                    for (size_t i = 0; i < n_ids; i++)
                        ids[i] = (uint32_t) tmp[i];
                    enc_ok = true;
                }
            }
            free(tmp);
        }
    }
    geist_model_destroy(model); /* done with this load; sweep reloads per mode */
    if (!had_tokenizer) {
        free(ids);
        geist_backend_destroy(be);
        GEIST_SKIP_IF(true, "model carries no usable tokenizer");
    }
    if (!enc_ok || n_ids < 2) {
        fprintf(stderr, "tokenizer encode failed or text too short (n_ids=%zu)\n", n_ids);
        free(ids);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    printf("model:    %s\n", model_path);
    printf("backend:  %s\n", geist_backend_name(be));
    printf("n_tokens: %zu (n_eval=%zu)\n\n", n_ids, n_ids - 2);

    const size_t n_modes = sizeof(MODES) / sizeof(MODES[0]);
    double       acc[sizeof(MODES) / sizeof(MODES[0])];
    printf("%-10s  %s\n", "kv_mode", "top-1 acc");
    printf("%-10s  %s\n", "-------", "---------");
    for (size_t m = 0; m < n_modes; m++) {
        set_or_unset("GEIST_KV_INT8", MODES[m].int8);
        set_or_unset("GEIST_KV_KIVI", MODES[m].kivi);
        set_or_unset("GEIST_KV_ROT", MODES[m].rot);
        acc[m] = run_one(model_path, be, ids, n_ids, nullptr);
        if (acc[m] < 0.0) {
            free(ids);
            geist_backend_destroy(be);
            return GEIST_TEST_FAIL;
        }
        printf("%-10s  %.4f\n", MODES[m].label, acc[m]);
    }

    /* Issue #61 headline: fraction of the INT8→KIVI gap that ROT recovers.
     * MODES order is FP32, INT8, INT8+ROT, KIVI. */
    const double gap = acc[3] - acc[1]; /* KIVI - INT8 */
    if (gap > 1e-6) {
        const double recovered = (acc[2] - acc[1]) / gap;
        printf("\nINT8→KIVI gap recovered by rotation: %.0f%%  (INT8 %.4f → ROT %.4f → KIVI "
               "%.4f)\n",
               100.0 * recovered,
               acc[1],
               acc[2],
               acc[3]);
    } else {
        printf("\nINT8 already within noise of KIVI (gap=%.4f) — rotation headroom is negligible "
               "here.\n",
               gap);
    }

    free(ids);
    geist_backend_destroy(be);
    return GEIST_TEST_PASS;
}
