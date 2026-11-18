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


OUTPUT_MODES = ("m1-output", "m1-ffn")


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
    if experiment in ("m1-state", *OUTPUT_MODES):
        decode = once(decode, '"float d=(vv[j]-mem)*be;float op=0.0f;"',
                      '"float residual=vv[j]-mem;float op=0.0f;"')
        decode = once(decode, "float sv=S[si]*decay+tk[i]*d;",
                      "float sv=q1_state_update(tk[i],residual,be,S[si],decay);")
        update_source = (probe / "qwen-m1-state-update.h").read_text()
        literal = "".join(f"        {json.dumps(line)}\n"
                          for line in update_source.splitlines(keepends=True))
        helper_source += "static const char metal_dn_m1_state_source[] =\n" + literal.rstrip() + ";\n\n"
        sources.append("metal_dn_m1_state_source")
    if experiment in OUTPUT_MODES:
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

    if experiment == "m1-ffn":
        patch_swiglu(metal, probe)
        patch_normalization(metal)
        patch_decay(metal, probe)
        patch_attention(metal, probe)


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


def patch_swiglu(metal, probe):
    shader = metal / "metal_shaders.h"
    source = shader.read_text()
    helper = (probe / "qwen-m1-swiglu.h").read_text()
    literal = "".join(f"        {json.dumps(line)}\n" for line in helper.splitlines(keepends=True))
    marker = "static const char metal_dn_m1_rounding_source[] ="
    source = once(source, marker, "static const char metal_m1_swiglu_source[] =\n" + literal.rstrip()
                  + ";\n\n" + marker)
    declaration = ('constant BinS&p[[buffer(3)]],uint "')
    source = once(source, declaration, 'constant BinS&p[[buffer(3)]],"\n'
                  '        "device const uint*exp_m1[[buffer(14)]],device const uint*rcp_m1[[buffer(15)]],"\n'
                  '        "device const uint*beta_neg_m1[[buffer(16)]],uint "')
    source = once(source, '"bv=b[p.b_offset+r*p.b_row_stride+c];y[p.y_offset+r*p.y_row_stride+c]=(av/(1.0f+exp(-av)))"\n'
                  '        "*bv;}\\n";',
                  '"bv=b[p.b_offset+r*p.b_row_stride+c];"\n'
                  '        "y[p.y_offset+r*p.y_row_stride+c]=q1_swiglu(av,bv,q1_beta(av,exp_m1,rcp_m1,beta_neg_m1));}\\n";')
    shader.write_text(source)
    update(metal / "pipelines.c",
           ('METAL_LIB(silu, "SiLU", metal_silu_source),',
            'METAL_LIB(silu, "SiLU", metal_m1_math_prelude, metal_dn_m1_rounding_source,\n'
            '                  metal_dn_m1_beta_k_source, metal_m1_swiglu_source, metal_silu_source),'))
    update(metal / "metal_shaders.h",
           (marker, 'static const char metal_m1_math_prelude[] = "#include <metal_stdlib>\\nusing namespace metal;\\n";\n'
            + marker))
    path = metal / "ops.c"
    source = path.read_text()
    start = source.index("static void metal_encode_binary_rows(")
    end = source.index("\n}", start)
    fn = source[start:end]
    marker = "    metal_msg_send_set_bytes(st, enc, params, sizeof(*params), 3);"
    fn = once(fn, marker, marker + "\n"
              "    if (pipeline == st->silu_mul_rows_pipeline) {\n"
              "        metal_msg_send_set_buffer(st, enc, st->q1_exp, 0, 14);\n"
              "        metal_msg_send_set_buffer(st, enc, st->q1_rcp, 0, 15);\n"
              "        metal_msg_send_set_buffer(st, enc, st->q1_beta_neg, 0, 16);\n"
              "    }")
    path.write_text(source[:start] + fn + source[end:])

def patch_normalization(metal):
    shader = metal / "metal_shaders.h"
    source = shader.read_text()
    start = source.index("static const char metal_elem_simd_source[]")
    end = source.index("static const char metal_embed_source[]", start)
    rows = source[start:end]
    binding = 'device const uint*rsqrt_m1[[buffer(17)]],'
    for params in ("Rows&p[[buffer(3)]],uint ", "Post&p[[buffer(4)]],uint ",
                   "Post2&p[[buffer(6)]],uint "):
        old = params + ('row' if params.startswith("Post2") else '"')
        rows = once(rows, old, params[:-5] + binding + "uint " + old[len(params):])
    rows = once(rows, "float inv=rsqrt(total/float(p.cols)+p.eps);",
                "float inv=q1_rsqrt(total/float(p.cols)+p.eps,rsqrt_m1);")
    rows = once(rows, "nt,uint cols,float eps){float sum=simd_sum(ss);",
                "nt,uint cols,float eps,device const uint*rsqrt_m1){float sum=simd_sum(ss);")
    rows = once(rows, "return rsqrt(total/float(cols)+eps);",
                "return q1_rsqrt(total/float(cols)+eps,rsqrt_m1);")
    if rows.count("sg,sl,nt,p.cols,p.eps)") != 4:
        raise ValueError("unexpected fused RMS reduction calls")
    rows = rows.replace("sg,sl,nt,p.cols,p.eps)", "sg,sl,nt,p.cols,p.eps,rsqrt_m1)")
    source = source[:start] + rows + source[end:]
    start = source.index("static const char metal_dn_decode_source[]")
    end = source.index("static const char metal_dn_decode_v_source[]", start)
    qk = source[start:end]
    qk = once(qk, "float qs=rsqrt(float(p.dk));", "float qs=q1_rsqrt(float(p.dk),rsqrt_m1);")
    qk = once(qk, "float qi=rsqrt(red[0]+p.eps);", "float qi=q1_rsqrt(red[0]+p.eps,rsqrt_m1);")
    source = source[:start] + qk + source[end:]
    source = once(source, '    "RA&p[[buffer(9)]],uint2',
                  '    "RA&p[[buffer(9)]],device const uint*rsqrt_m1[[buffer(17)]],uint2')
    source = once(source, "float inv=rsqrt(total/float(p.hd)+p.eps);",
                  "float inv=q1_rsqrt(total/float(p.hd)+p.eps,rsqrt_m1);")
    marker = '        "inline float q1_beta(float x,'
    source = once(source, marker,
                  '        "inline float q1_rsqrt(float x, device const uint*table) {\\n"\n'
                  '        "    return as_type<float>(q1_rsqrt_bits(as_type<uint>(x),table));\\n"\n'
                  '        "}\\n"\n' + marker)
    shader.write_text(source)
    update(
        metal / "pipelines.c",
        ('                  "SIMD elementwise",\n                  metal_elem_simd_source,',
         '                  "SIMD elementwise",\n'
         '                  metal_m1_math_prelude, metal_dn_m1_rounding_source, metal_dn_m1_beta_k_source,\n'
         '                  metal_elem_simd_source,'),
        ('METAL_LIB(rope_append, "RoPE + KV append", metal_rope_append_source),',
         'METAL_LIB(rope_append, "RoPE + KV append", metal_m1_math_prelude,\n'
         '                  metal_dn_m1_rounding_source, metal_dn_m1_beta_k_source, metal_rope_append_source),'),
    )
    path = metal / "ops.c"
    source = path.read_text()
    for function, marker in (
        ("metal_encode_rmsnorm_rows", "params, sizeof(*params), 3"),
        ("metal_encode_rmsnorm_add_rows", "params, sizeof(*params), 4"),
        ("metal_encode_rmsnorm_add_norm_rows", "params, sizeof(*params), 6"),
        ("metal_attn_rope_append", "&ra, sizeof(ra), 9"),
    ):
        start = source.index(f" {function}(")
        end = source.index("\n}", start)
        line = f"    metal_msg_send_set_bytes(st, enc, {marker});"
        body = once(source[start:end], line, line +
                    "\n    metal_msg_send_set_buffer(st, enc, st->q1_rsqrt, 0, 17);")
        source = source[:start] + body + source[end:]
    path.write_text(source)

def patch_decay(metal, probe):
    update(
        metal / "metal_internal.h",
        ("    struct metal_buffer_h *q1_rsqrt;",
         "    struct metal_buffer_h *q1_rsqrt;\n"
         "    struct metal_buffer_h *q1_softplus_pos;\n"
         "    struct metal_buffer_h *q1_softplus_neg;"),
    )
    update(
        metal / "lifecycle.c",
        ('        metal_msg_send_void0(st, st->q1_rsqrt, "release");',
         '        metal_msg_send_void0(st, st->q1_rsqrt, "release");\n'
         '        metal_msg_send_void0(st, st->q1_softplus_pos, "release");\n'
         '        metal_msg_send_void0(st, st->q1_softplus_neg, "release");'),
        ('getenv("QWEN47_M1_RSQRT"), &st->q1_rsqrt, st)) {',
         'getenv("QWEN47_M1_RSQRT"), &st->q1_rsqrt, st) ||\n'
         '        !q1_load_buffer(Q1_EXP_FIRST, Q1_EXP_LAST, getenv("QWEN47_M1_SOFTPLUS_POS"), &st->q1_softplus_pos, st) ||\n'
         '        !q1_load_buffer(Q1_EXP_FIRST, Q1_EXP_LAST, getenv("QWEN47_M1_SOFTPLUS_NEG"), &st->q1_softplus_neg, st)) {'),
    )
    update(
        metal / "ops.c",
        ('        metal_msg_send_set_buffer(st, enc, st->q1_rsqrt, 0, 17);',
         '        metal_msg_send_set_buffer(st, enc, st->q1_rsqrt, 0, 17);\n'
         '        metal_msg_send_set_buffer(st, enc, st->q1_softplus_pos, 0, 18);\n'
         '        metal_msg_send_set_buffer(st, enc, st->q1_softplus_neg, 0, 19);'),
    )
    shader = metal / "metal_shaders.h"
    source = shader.read_text()
    start = source.index("static const char metal_dn_decode_v_source[]")
    end = source.index("static const char metal_dn_decode_states_source[]", start)
    v = source[start:end]
    marker = '        "device const uint*rsqrt_m1[[buffer(17)]],"'
    v = once(v, marker, marker +
             '\n        "device const uint*sp_pos[[buffer(18)]],device const uint*sp_neg[[buffer(19)]],"')
    v = once(v, "float sp=log(1.0f+exp(alpha[p.ao+h]+dt[p.dto+h]));float decay=exp(aw[p.awo+h]*sp);",
             "float sp=as_type<float>(q1_softplus_bits(as_type<uint>(alpha[p.ao+h]+dt[p.dto+h]),sp_pos,sp_neg));"
             "const float decay_exp=aw[p.awo+h]*sp;"
             "float decay=decay_exp<=0.0f?as_type<float>(q1_exp_bits(as_type<uint>(-decay_exp)&0x7fffffffu,exp_m1)):NAN;")
    helper = once((probe / "qwen-m1-softplus.h").read_text(), '#include "qwen-m1-rounding.h"', "")
    literal = "".join(f"        {json.dumps(line)}\n" for line in helper.splitlines(keepends=True))
    declaration = "static const char metal_m1_softplus_source[]=\n" + literal.rstrip() + ";\n\n"
    shader.write_text(source[:start] + declaration + v + source[end:])
    update(metal / "pipelines.c",
           ("                  metal_dn_m1_output_source,",
            "                  metal_dn_m1_output_source,\n                  metal_m1_softplus_source,"))

def patch_attention(metal, probe):
    shader = metal / "metal_shaders.h"
    source = shader.read_text()
    # Float32 matvec, RMSNorm and decode softmax use the M1 SIMD addition tree.
    reductions = (
        ("metal_f32_sg_mv_source", "static const char metal_f32_mm_source", 1),
        ("metal_elem_simd_source", "static const char metal_embed_source", 2),
        ("GEIST_METAL_ROPE_APPEND", "static const char metal_attn_source", 1),
        ("metal_attn_dec_combine_source", "static const char metal_attn_decv_source", 1),
        ("metal_attn_decv_source", "/* Correctness-first Gated-DeltaNet mixer kernel", 3),
    )
    for name, end, count in reductions:
        start = source.index("#define " + name if name.startswith("GEIST_")
                             else "static const char " + name)
        finish = source.index(end, start)
        body = source[start:finish]
        if body.count("simd_sum(") != count:
            raise ValueError(f"unexpected SIMD reductions in {name}")
        source = source[:start] + body.replace("simd_sum(", "q1_simd_sum(") + source[finish:]

    start = source.index("static const char metal_attn_source[]")
    end = source.index("static const char metal_q4k_gate_up_n4_source[]", start)
    scalar = source[start:end]
    exps = "device const uint*exp_m1[[buffer(14)]],"
    rcps = "device const uint*rcp_m1[[buffer(15)]],"
    scalar = once(scalar, "constant Attn&p[[buffer(4)]],uint2 ",
                  f"constant Attn&p[[buffer(4)]],{exps}{rcps}uint2 ")
    for arg in ("mold-mnew", "sc-mnew"):
        scalar = once(scalar, f"exp({arg})", f"q1_nonpositive_exp({arg},exp_m1)")
    scalar = once(scalar, "float inv=1.0f/mls[1];", "float inv=q1_positive_rcp(mls[1],rcp_m1);")
    source = source[:start] + scalar + source[end:]

    start = source.index("static const char metal_attn_dec_combine_source[]")
    end = source.index("/* Correctness-first Gated-DeltaNet mixer kernel", start)
    split = source[start:end]
    split = once(split, "constant CB&p[[buffer(2)]],uint2 ", f"constant CB&p[[buffer(2)]],{exps}{rcps}uint2 ")
    split = once(split, "ln<p.ns?exp(m-M):0.0f;", "ln<p.ns?q1_nonpositive_exp(m-M,exp_m1):0.0f;")
    split = once(split, "sf[sg*32u+ln]=e/L;", "sf[sg*32u+ln]=e*q1_positive_rcp(L,rcp_m1);")
    binding = "constant A&p[[buffer(4)]],constant SP&sp[[buffer(5)]],uint2 "
    if split.count(binding) != 2 or split.count("exp(M-Mn)") != 2 or split.count("exp(s-Mn)") != 2:
        raise ValueError("unexpected F32 split attention")
    split = split.replace(binding, binding[:-6] + exps + "uint2 ")
    split = split.replace("exp(M-Mn)", "q1_nonpositive_exp(M-Mn,exp_m1)")
    split = split.replace("exp(s-Mn)", "q1_nonpositive_exp(s-Mn,exp_m1)")
    source = source[:start] + split + source[end:]

    marker = 'constant Bin&p[[buffer(3)]],uint "'
    start = source.index("static const char metal_qgate_source[]")
    end = source.index("static const char metal_silu_source[]", start)
    gate = once(source[start:end], marker, marker[:-6] + exps + rcps +
                'device const uint*beta_neg_m1[[buffer(16)]],uint "')
    gate = once(gate, "x[p.a_offset+r*p.a_row_stride+c]/(1.0f+exp(-v));",
                "x[p.a_offset+r*p.a_row_stride+c]*q1_beta(v,exp_m1,rcp_m1,beta_neg_m1);")
    source = source[:start] + gate + source[end:]
    helper = (probe / "qwen-m1-operations.metal").read_text()
    literal = "".join(f"        {json.dumps(line)}\n" for line in helper.splitlines(keepends=True))
    marker = "static const char metal_dn_m1_rounding_source[] ="
    shader.write_text(once(source, marker, "static const char metal_m1_operations_source[] =\n"
                          + literal.rstrip() + ";\n\n" + marker))

    pipelines = metal / "pipelines.c"
    prelude = "metal_m1_math_prelude, metal_dn_m1_rounding_source,"
    ops = "metal_m1_operations_source,"
    update(
        pipelines,
        ('METAL_LIB(qgate, "Qwen gate", metal_qgate_source),',
         f'METAL_LIB(qgate, "Qwen gate", {prelude} metal_dn_m1_beta_k_source, metal_qgate_source),'),
        ('                  metal_elem_simd_source,', f'                  {ops} metal_elem_simd_source,'),
        ('                  metal_f32_sg_mv_source,',
         f'                  {prelude} {ops} metal_f32_sg_mv_source,'),
        ('metal_dn_m1_rounding_source, metal_dn_m1_beta_k_source, metal_rope_append_source),',
         f'metal_dn_m1_rounding_source, metal_dn_m1_beta_k_source, {ops} metal_rope_append_source),'),
        ('METAL_LIB(attn, "attention", metal_attn_source),',
         f'METAL_LIB(attn, "attention", {prelude} {ops} metal_attn_source),'),
        ('METAL_LIB(attn_dec_combine, "attention combine", metal_attn_dec_combine_source),',
         f'METAL_LIB(attn_dec_combine, "attention combine", {prelude} {ops} metal_attn_dec_combine_source),'),
        ('                  "block decode attention",',
         f'                  "block decode attention",\n                  {prelude} {ops}'),
    )
    path = metal / "ops.c"
    source = path.read_text()
    for function, marker, buffers, cond in (
        ("metal_encode_sigmoid_mul", "params, sizeof(*params), 3", ("exp", "rcp", "beta_neg"), ""),
        ("metal_encode_attention_rows", "params, sizeof(*params), 4", ("exp", "rcp"),
         "pipeline == st->attention_rows_pipeline"),
        ("metal_attention_dec_kv", "&nsplit, sizeof(nsplit), 5", ("exp",), "block_pipeline != nullptr"),
        ("metal_attention_dec_kv", "cb, sizeof(cb), 2", ("exp", "rcp"), ""),
    ):
        start = source.index(f" {function}(")
        end = source.index("\n}", start)
        line = f"    metal_msg_send_set_bytes(st, enc, {marker});"
        bindings = "\n".join(
            f"{'        ' if cond else '    '}metal_msg_send_set_buffer(st, enc, st->q1_{b}, 0, {14 + i});"
            for i, b in enumerate(buffers)
        )
        if cond:
            bindings = f"    if ({cond}) {{\n{bindings}\n    }}"
        body = once(source[start:end], line, f"{line}\n{bindings}")
        source = source[:start] + body + source[end:]
    path.write_text(source)

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] not in ("m1-beta-k", "m1-state", *OUTPUT_MODES)):
        sys.exit("usage: qwen47-m1-rounding-patch.py ENGINE [m1-beta-k|m1-state|m1-output|m1-ffn]")
    patch(Path(sys.argv[1]), Path(__file__).resolve().parent, *sys.argv[2:])
