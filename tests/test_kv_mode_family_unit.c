/*
 * test_kv_mode_family_unit - which KV cache a session gets (#48).
 *
 * The CPU backends read an F16 cache now (caps.kv_f16_attention) but do not
 * switch to it on their own (caps.kv_f16_by_default unset): a model family
 * or a session asks for it. On every CPU backend in the build:
 *   - a dense Llama under AUTO keeps the backend's default (no F16);
 *   - Qwen3-MoE under AUTO gets F16 (its family's kv_auto);
 *   - an explicit kv_mode beats the family: FP32 is FP32, and F16 on Llama
 *     is F16;
 *   - the environment beats the family: GEIST_KV_INT8=1 gives Qwen3-MoE
 *     INT8, GEIST_KV_F16=0 gives it FP32;
 *   - GEIST_KV_F16=1 gives F16 to a dense model whose AUTO is FP32, though
 *     the backend does not default to F16;
 *   - kv_mode INT4 is INT4 on Llama but F16 on Qwen3-MoE, where INT4 misses
 *     the quality bar (#53, transformer_family.kv_int4_instead);
 *   - an INT8 or INT4 cache is Hadamard-rotated unless GEIST_KV_ROT=0.
 */
#define _POSIX_C_SOURCE 200809L /* setenv, unsetenv */
#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_ENGINE_LAYER

#include "../src/archs/transformer/arch_state.h"
#include "../src/engine/model.h"
#include "model_fixtures.h"
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>

/* The KV layout a fresh session gets: 'F' F16, 'I' INT8, 'Q' INT4 (K INT8,
 * V 4-bit), 'P' FP32 (plain); lowercase 'i' and 'q' when rotated. */
static char kv_of(struct geist_model *m, struct geist_backend *be, enum geist_kv_mode mode) {
    struct geist_session           *s = nullptr;
    const struct geist_session_opts o = {
            .top_p = 1.0f, .m_max = 8, .max_seq_len = 64, .kv_mode = mode};
    if (geist_session_create(m, be, &o, &s) != GEIST_OK)
        return '?';
    const struct transformer_arch_session *a = geist_session_internal_arch_session(s);
    const char c = a->kv_f16_enabled           ? 'F'
                   : a->kv_int4_packed_enabled ? (a->kv_rot_enabled ? 'q' : 'Q')
                   : a->kv_int8_enabled        ? (a->kv_rot_enabled ? 'i' : 'I')
                   : a->kv_kivi_enabled        ? 'I'
                                               : 'P';
    geist_session_destroy(s);
    return c;
}

int main(void) {
    /* No profile cache in the user's home: placement stays the tests' (#165). */
    setenv("GEIST_MOE_PROFILE_CACHE", "0", 1);
    const char *const backends[] = {"cpu_scalar", "cpu_neon", "cpu_x86"};
    int               fails      = 0;
    unsetenv("GEIST_KV_INT8");
    unsetenv("GEIST_KV_F16");
    unsetenv("GEIST_KV_INT4");
    unsetenv("GEIST_KV_KIVI");
    for (size_t b = 0; b < sizeof backends / sizeof backends[0]; b++) {
        struct geist_backend *be = nullptr;
        if (geist_backend_create(backends[b], nullptr, nullptr, &be) != GEIST_OK) {
            printf("skip %s: not in this build\n", backends[b]);
            continue;
        }
        struct tf_buf       dense = mf_llama_gguf(&(struct mf_llama) {.layers   = 1,
                                                                      .d_model  = 64,
                                                                      .heads    = 4,
                                                                      .kv_heads = 2,
                                                                      .ffn      = 128,
                                                                      .vocab    = 64,
                                                                      .context  = 64,
                                                                      .seed     = 3});
        struct tf_buf       moe   = mf_qwen3moe_gguf(&(struct mf_qwen3moe) {.layers    = 1,
                                                                            .d_model   = 64,
                                                                            .heads     = 4,
                                                                            .kv_heads  = 2,
                                                                            .head_dim  = 16,
                                                                            .experts   = 4,
                                                                            .used      = 2,
                                                                            .expert_ff = 32,
                                                                            .vocab     = 64,
                                                                            .seed      = 3});
        struct geist_model *md = nullptr, *mm = nullptr;
        if (geist_model_load_from_memory(dense.b, dense.n, be, &md) != GEIST_OK ||
            geist_model_load_from_memory(moe.b, moe.n, be, &mm) != GEIST_OK) {
            fails += geist_expect(false, "fixtures load");
        } else {
            const char dense_auto = kv_of(md, be, GEIST_KV_AUTO);
            printf("%s: llama AUTO=%c, qwen3moe AUTO=%c FP32=%c, llama F16=%c\n",
                   backends[b],
                   dense_auto,
                   kv_of(mm, be, GEIST_KV_AUTO),
                   kv_of(mm, be, GEIST_KV_FP32),
                   kv_of(md, be, GEIST_KV_F16));
            fails += geist_expect(dense_auto != 'F', "a dense model keeps the backend default");
            fails += geist_expect(kv_of(mm, be, GEIST_KV_AUTO) == 'F',
                                  "qwen3moe gets F16 under AUTO");
            fails += geist_expect(kv_of(mm, be, GEIST_KV_FP32) == 'P',
                                  "kv_mode FP32 beats the family");
            fails += geist_expect(kv_of(md, be, GEIST_KV_F16) == 'F', "kv_mode F16 is honoured");
            setenv("GEIST_KV_INT8", "1", 1);
            fails += geist_expect(kv_of(mm, be, GEIST_KV_AUTO) == 'i',
                                  "GEIST_KV_INT8=1 beats the family, rotated");
            unsetenv("GEIST_KV_INT8");
            setenv("GEIST_KV_F16", "0", 1);
            fails += geist_expect(kv_of(mm, be, GEIST_KV_AUTO) == 'P', "GEIST_KV_F16=0 gives FP32");
            unsetenv("GEIST_KV_F16");
            fails += geist_expect(kv_of(md, be, GEIST_KV_INT4) == 'q', "INT4 on llama, rotated");
            fails += geist_expect(kv_of(mm, be, GEIST_KV_INT4) == 'F',
                                  "INT4 on qwen3moe falls back to its F16");
            setenv("GEIST_KV_ROT", "0", 1);
            fails += geist_expect(kv_of(md, be, GEIST_KV_INT8) == 'I', "GEIST_KV_ROT=0 opts out");
            unsetenv("GEIST_KV_ROT");
            if (dense_auto == 'P') {
                setenv("GEIST_KV_F16", "1", 1);
                fails += geist_expect(kv_of(md, be, GEIST_KV_AUTO) == 'F',
                                      "GEIST_KV_F16=1 requests F16 under AUTO");
                unsetenv("GEIST_KV_F16");
            }
        }
        geist_model_destroy(md);
        geist_model_destroy(mm);
        geist_backend_destroy(be);
        free(dense.b);
        free(moe.b);
    }
    if (fails == 0)
        printf("PASS: KV mode by family, session and environment; dense models unchanged\n");
    return fails == 0 ? 0 : 1;
}
