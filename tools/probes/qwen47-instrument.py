"""Apply trace-only instrumentation to a temporary copy of the pinned engine."""
from pathlib import Path
import shutil
import sys

engine, diagnostic = map(Path, sys.argv[1:])
forward = engine / "src/archs/transformer/forward"


def insert(source, anchor, change):
    if source.count(anchor) != 1:
        raise ValueError(f"Expected a unique instrument point: {anchor}")
    index = source.index(anchor)
    return source[:index] + change + source[index:]


def add_header(source):
    anchor = '#include "heap.h"'
    if source.count(anchor) != 1:
        raise ValueError("Expected heap header")
    return source.replace(anchor, anchor + '\n#include "qwen47-trace.h"')


shutil.copyfile(diagnostic / "qwen47-trace.h", forward / "qwen47-trace.h")
path = forward / "layer.c"
source = add_header(path.read_text())
source = insert(source, "    ctx.profile = transformer_profile_enabled(&g_layer_profile);",
                '    qwen47_trace(&ctx, "input", st->d_model, ctx.h_in_buf);\n')
source = insert(source, "    s  = fam->ffn_block(&ctx);",
                '    qwen47_trace(&ctx, "mixer", st->d_model, sess->scratch_h_post_attn);\n')
source = insert(source, "    if (ctx.advance_kv) {",
                '    qwen47_trace(&ctx, "ffn", st->d_model, ctx.h_out_buf);\n')
path.write_text(source)

path = forward / "layer_deltanet.c"
source = add_header(path.read_text())
source = insert(source, "    struct geist_tensor t_qkv = view_2d(sess->dn_scratch_qkv, ctx->SEQ, (int64_t) convd);\n"
                "    s                         = linear_w_or_legacy(be,",
                '    qwen47_trace(ctx, "dn_qkv_input", st->d_model, sess->scratch_normed);\n')
for key, traces in (
    ("NORM", [("norm", "st->d_model", "sess->scratch_normed")]),
    ("BETA_ALPHA", [("beta", "n_vh", "sess->dn_scratch_b"),
                    ("alpha", "n_vh", "sess->dn_scratch_a")]),
    ("QKV", [("qkv", "convd", "sess->dn_scratch_qkv")]),
    ("Z", [("z", "vald", "sess->dn_scratch_z")]),
):
    changes = "".join(f'    qwen47_trace(ctx, "dn_{name}", {count}, {buffer});\n'
                      for name, count, buffer in traces)
    source = insert(source, f"    dn_profile_mark(DN_PROFILE_{key}, &t0);", changes)
source = insert(source, "    dn_profile_mark(DN_PROFILE_MIX, &t0);",
                '    qwen47_trace(ctx, "dn_mix", vald, sess->dn_scratch_z);\n')
source = insert(source, "    dn_profile_mark(DN_PROFILE_OUT, &t0);",
                '    qwen47_trace(ctx, "dn_out_projection", st->d_model, sess->scratch_o);\n')
path.write_text(source)
