#!/usr/bin/env python3
"""Bind the calibrated SiLU arithmetic in a scratch copy of the frozen engine."""
import json
from pathlib import Path
import sys


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"expected one source anchor {old!r}")
    return text.replace(old, new, 1)


def update(file, *edits):
    text = file.read_text()
    for old, new in edits:
        text = once(text, old, new)
    file.write_text(text)


def patch(engine, probe):
    metal = engine / "src/backends/metal"
    for basename in ("qwen-m1-rounding.h", "qwen-m1-rounding-io.h", "qwen47-m1-rounding-metal.h"):
        (metal / basename).write_bytes((probe / basename).read_bytes())

    update(
        metal / "metal_internal.h",
        ("    size_t                   dn_scratch_bytes;",
         "    size_t                   dn_scratch_bytes;\n"
         "    struct metal_buffer_h *q1_exp;\n"
         "    struct metal_buffer_h *q1_rcp;"),
    )
    update(
        metal / "lifecycle.c",
        ('#include "parse.h"', '#include "parse.h"\n#include "qwen47-m1-rounding-metal.h"'),
        ('        metal_msg_send_void0(st, st->dn_scratch, "release");',
         '        metal_msg_send_void0(st, st->dn_scratch, "release");\n'
         '        metal_msg_send_void0(st, st->q1_exp, "release");\n'
         '        metal_msg_send_void0(st, st->q1_rcp, "release");'),
        ('    metal_msg_send_void0(st, st->device, "retain");',
         '    metal_msg_send_void0(st, st->device, "retain");\n'
         '    if (!q1_load_buffer(Q1_EXP_FIRST, Q1_EXP_LAST, getenv("QWEN47_M1_EXP"), &st->q1_exp, st) ||\n'
         '        !q1_load_buffer(Q1_RCP_FIRST, Q1_RCP_LAST, getenv("QWEN47_M1_RCP"), &st->q1_rcp, st)) {\n'
         '        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: missing or invalid M1 calibration");\n'
         '        metal_destroy_state(be, st);\n'
         '        return GEIST_E_BACKEND;\n'
         '    }'),
    )
    ops = metal / "ops.c"
    original = ops.read_text()
    start = original.index("metal_deltanet_mix(struct geist_backend *be,")
    end = original.index("/* deltanet_mix_states", start)
    mixer = once(
        original[start:end],
        "    const size_t key_dim   = args->n_k_heads * args->head_k;",
         "    if (args->seq != 1) {\n"
         "        return GEIST_E_BACKEND;\n"
         "    }\n\n"
         "    const size_t key_dim   = args->n_k_heads * args->head_k;",
    )
    mixer = once(
        mixer,
        "        const struct metal_size gk = {args->n_k_heads, 1, 1};\n"
         "        const struct metal_size tk = {256, 1, 1};\n"
         "        metal_msg_send_set_pipeline(st, enc, st->dn_dec_qk_pipeline);",
         "        metal_msg_send_set_buffer(st, enc, st->q1_exp, 0, 14);\n"
         "        metal_msg_send_set_buffer(st, enc, st->q1_rcp, 0, 15);\n"
         "        const struct metal_size gk = {args->n_k_heads, 1, 1};\n"
         "        const struct metal_size tk = {256, 1, 1};\n"
         "        metal_msg_send_set_pipeline(st, enc, st->dn_dec_qk_pipeline);",
    )
    mixer = once(
        mixer,
        "    struct metal_deltanet_params params = ",
        "    if (args->head_k > 128u || st->dn_dec_qk_pipeline == nullptr ||\n"
        "        st->dn_dec_v_pipeline == nullptr ||\n"
        "        !metal_pipeline_fits(st, st->dn_dec_qk_pipeline, 256u) ||\n"
        "        !metal_pipeline_fits(st, st->dn_dec_v_pipeline, 4u * args->head_v)) {\n"
        '        geist_backend_set_error(be, GEIST_E_BACKEND, "metal: calibrated SiLU kernels unavailable");\n'
        "        return GEIST_E_BACKEND;\n"
        "    }\n"
        "    struct metal_deltanet_params params = ",
    )
    ops.write_text(original[:start] + mixer + original[end:])

    shader = metal / "metal_shaders.h"
    source = shader.read_text()
    start = source.index("static const char metal_dn_decode_source[] =")
    end = source.index("static const char metal_dn_decode_states_source[] =", start)
    decode = source[start:end]
    marker = 'constant P&p[[buffer(10)]],"'
    if decode.count(marker) != 2:
        raise ValueError("unexpected single-token DeltaNet signatures")
    decode = decode.replace(
        marker,
        marker + '\n        "device const uint*exp_m1[[buffer(14)]],'
                 'device const uint*rcp_m1[[buffer(15)]],"',
    )
    for x in ("qc", "kc", "vc", "gate"):
        decode = once(decode, f"silu1({x})", f"q1_silu({x},exp_m1,rcp_m1)")

    helper = (probe / "qwen-m1-rounding.h").read_text() + """
inline float q1_silu(float x, device const uint *exp_m1, device const uint *rcp_m1) {
    #pragma clang fp reassociate(off)
    #pragma clang fp contract(off)
    // Index the archived FP32 encodings after the denominator rounds once.
    const float e = as_type<float>(q1_exp_bits(as_type<uint>(x) & 0x7fffffffu, exp_m1));
    const float r = as_type<float>(q1_lookup(as_type<uint>(1.0f + e), rcp_m1));
    const float n = x >= 0.0f ? x : x * e;
    return n * r;
}
"""
    prefix = "static const char metal_dn_decode_source[] =\n"
    literal = "".join(f"        {json.dumps(line)}\n" for line in helper.splitlines(keepends=True))
    helper_source = "static const char metal_dn_m1_rounding_source[] =\n" + literal.rstrip() + ";\n\n"
    if not decode.startswith(prefix):
        raise ValueError("unexpected single-token DeltaNet source")
    split = '        "kernel void dn_dec_v('
    decode = once(decode, split, '        "";\n\n'
                  'static const char metal_dn_decode_v_source[] =\n' + split)
    shader.write_text(source[:start] + helper_source + decode + source[end:])
    update(
        metal / "pipelines.c",
        ("                  metal_dn_decode_source,",
         "                  metal_dn_m1_rounding_source,\n"
         "                  metal_dn_decode_source,\n"
         "                  metal_dn_decode_v_source,"),
    )


if __name__ == "__main__":
    patch(Path(sys.argv[1]), Path(__file__).resolve().parent)
