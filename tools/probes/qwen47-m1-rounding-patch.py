#!/usr/bin/env python3
"""Bind the calibrated DeltaNet arithmetic in a scratch copy of the frozen engine."""
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


def patch(engine, probe, experiment="m1-silu"):
    beta_k = experiment != "m1-silu"
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
    if beta_k:
        additional_buffers(metal)
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
    if beta_k:
        marker = "        metal_msg_send_set_buffer(st, enc, st->q1_rcp, 0, 15);"
        mixer = once(mixer, marker, marker + "\n"
            "        metal_msg_send_set_buffer(st, enc, st->q1_beta_neg, 0, 16);\n"
            "        metal_msg_send_set_buffer(st, enc, st->q1_rsqrt, 0, 17);")
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
    sources = ["metal_dn_m1_rounding_source"]
    if beta_k:
        decode = beta_k_decode(decode)
        additional = (probe / "qwen-m1-deltanet.h").read_text()
        additional = once(additional, '#include "qwen-m1-rounding.h"', "")
        additional = once(additional, '#include "qwen-m1-rsqrt-source.h"',
                          (probe / "qwen-m1-rsqrt-source.h").read_text())
        additional += """
inline float q1_beta(float x, device const uint *exp_m1, device const uint *rcp_m1,
                     device const uint *beta_neg_m1) {
    const uint key = as_type<uint>(x);
    const uint magnitude = key & 0x7fffffffu;
    if (key >> 31)
        return as_type<float>(q1_beta_neg_bits(magnitude, beta_neg_m1));
    const float e = as_type<float>(q1_exp_bits(magnitude, exp_m1));
    return as_type<float>(q1_lookup(as_type<uint>(1.0f + e), rcp_m1));
}
"""
        literal = "".join(f"        {json.dumps(line)}\n" for line in additional.splitlines(keepends=True))
        helper_source += "static const char metal_dn_m1_beta_k_source[] =\n" + literal.rstrip() + ";\n\n"
        sources.append("metal_dn_m1_beta_k_source")
    if experiment in ("m1-state", "m1-output"):
        decode = once(decode, '"float d=(vv[j]-mem)*be;float op=0.0f;"',
                      '"float residual=vv[j]-mem;float op=0.0f;"')
        decode = once(decode, "float sv=S[si]*decay+tk[i]*d;",
                      "float sv=q1_state_update(tk[i],residual,be,S[si],decay);")
        update_source = (probe / "qwen-m1-state-update.h").read_text()
        literal = "".join(f"        {json.dumps(line)}\n"
                          for line in update_source.splitlines(keepends=True))
        helper_source += "static const char metal_dn_m1_state_source[] =\n" + literal.rstrip() + ";\n\n"
        sources.append("metal_dn_m1_state_source")
    if experiment == "m1-output":
        v_start = decode.index("static const char metal_dn_decode_v_source[]")
        v = decode[v_start:]
        beta_binding = 'device const uint*beta_neg_m1[[buffer(16)]],"'
        v = once(v, beta_binding, beta_binding +
                 '\n        "device const uint*rsqrt_m1[[buffer(17)]],"')
        v = once(v, "float inv=rsqrt(ss/float(p.dv)+p.eps);",
                 "float inv=as_type<float>(q1_rsqrt_bits(as_type<uint>(ss/float(p.dv)+p.eps),rsqrt_m1));")
        v = once(v, "z[p.zo+h*p.dv+j]=out*inv*nw[p.no+j]*q1_silu(gate,exp_m1,rcp_m1);",
                 "float e=as_type<float>(q1_exp_bits(as_type<uint>(gate)&0x7fffffffu,exp_m1));"
                 "float r=as_type<float>(q1_lookup(as_type<uint>(1.0f+e),rcp_m1));"
                 "z[p.zo+h*p.dv+j]=q1_deltanet_output(out,inv,nw[p.no+j],gate,e,r);")
        decode = decode[:v_start] + v
        output_source = (probe / "qwen-m1-deltanet-output.h").read_text()
        literal = "".join(f"        {json.dumps(line)}\n"
                          for line in output_source.splitlines(keepends=True))
        helper_source += "static const char metal_dn_m1_output_source[] =\n" + literal.rstrip() + ";\n\n"
        sources.append("metal_dn_m1_output_source")
    shader.write_text(source[:start] + helper_source + decode + source[end:])
    update(
        metal / "pipelines.c",
        ("                  metal_dn_decode_source,",
         "".join(f"                  {name},\n" for name in sources)
         + "                  metal_dn_decode_source,\n"
         + "                  metal_dn_decode_v_source,"),
    )


def additional_buffers(metal):
    update(
        metal / "metal_internal.h",
        ("    struct metal_buffer_h *q1_rcp;",
         "    struct metal_buffer_h *q1_rcp;\n"
         "    struct metal_buffer_h *q1_beta_neg;\n"
         "    struct metal_buffer_h *q1_rsqrt;"),
    )
    update(
        metal / "lifecycle.c",
        ('        metal_msg_send_void0(st, st->q1_rcp, "release");',
         '        metal_msg_send_void0(st, st->q1_rcp, "release");\n'
         '        metal_msg_send_void0(st, st->q1_beta_neg, "release");\n'
         '        metal_msg_send_void0(st, st->q1_rsqrt, "release");'),
        ('getenv("QWEN47_M1_RCP"), &st->q1_rcp, st)) {',
         'getenv("QWEN47_M1_RCP"), &st->q1_rcp, st) ||\n'
         '        !q1_load_buffer(Q1_EXP_FIRST, Q1_EXP_LAST, getenv("QWEN47_M1_BETA_NEG"), &st->q1_beta_neg, st) ||\n'
         '        !q1_load_buffer(0x3f800000u, 0x407fffffu, getenv("QWEN47_M1_RSQRT"), &st->q1_rsqrt, st)) {'),
    )


def beta_k_decode(decode):
    start = decode.index("static const char metal_dn_decode_v_source[]")
    qk = decode[:start]
    v = decode[start:]
    marker = 'device const uint*rcp_m1[[buffer(15)]],"'
    qk = once(qk, marker, marker + '\n        "device const uint*rsqrt_m1[[buffer(17)]],"')
    v = once(v, marker, marker + '\n        "device const uint*beta_neg_m1[[buffer(16)]],"')
    qk = once(qk, "float ki=rsqrt(red[0]+p.eps);",
              "float ki=as_type<float>(q1_rsqrt_bits(as_type<uint>(red[0]+p.eps),rsqrt_m1));")
    v = once(v, "float be=1.0f/(1.0f+exp(-beta[p.bo+h]));",
             "float be=q1_beta(beta[p.bo+h],exp_m1,rcp_m1,beta_neg_m1);")
    return qk + v


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] not in ("m1-beta-k", "m1-state", "m1-output")):
        sys.exit("usage: qwen47-m1-rounding-patch.py ENGINE [m1-beta-k|m1-state|m1-output]")
    patch(Path(sys.argv[1]), Path(__file__).resolve().parent, *sys.argv[2:])
