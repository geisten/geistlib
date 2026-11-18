from pathlib import Path
import sys

forward = Path(sys.argv[1]) / "src/archs/transformer/forward"
header = forward / "qwen47-trace.h"
c = forward / "blocks.c"

def only(text, old, new):
    if text.count(old) != 1:
        raise ValueError(old)
    return text.replace(old, new, 1)

s = header.read_text()
s = only(s, '(ctx->q_position != 0 && ctx->q_position != 90)',
            '(ctx->q_position != 0 && ctx->q_position != 1 && ctx->q_position != 7 && ctx->q_position != 90)')
s = only(s, '(!strcmp(stage, "dn_qkv_input")',
            '(!strncmp(stage, "ffn_", 4) || !strcmp(stage, "mixer") || !strcmp(stage, "ffn") ||\n'
            '         !strcmp(stage, "dn_qkv_input")')
header.write_text(s)
s = c.read_text()
s = only(s, '#include "blocks.h"', '#include "blocks.h"\n#include "heap.h"\n#include "qwen47-trace.h"')
lo = s.index('enum geist_status blk_ffn_swiglu(')
hi = s.index('enum geist_status blk_ffn_geglu(', lo)
f = s[lo:hi]
f = only(f, '    if (s == GEIST_OK && ctx->seq == 1', '    if (s == GEIST_OK)\n        qwen47_trace(ctx, "ffn_input", d, x);\n    if (s == GEIST_OK && ctx->seq == 1')
f = only(f, '        if (s == GEIST_OK)\n            s = blk_silu_mul(', '        if (s == GEIST_OK) {\n'
            '            qwen47_trace(ctx, "ffn_gate", ctx->inter, sess->scratch_gate);\n'
            '            qwen47_trace(ctx, "ffn_up", ctx->inter, sess->scratch_up);\n'
            '        }\n        if (s == GEIST_OK)\n            s = blk_silu_mul(')
f = only(f, '    if (s == GEIST_OK)\n        s = blk_down(', '    if (s == GEIST_OK)\n'
            '        qwen47_trace(ctx, "ffn_middle", ctx->inter, sess->scratch_gate);\n'
            '    if (s == GEIST_OK)\n        s = blk_down(')
f = only(f, '    if (s == GEIST_OK)\n        s = blk_residual_out(', '    if (s == GEIST_OK)\n'
            '        qwen47_trace(ctx, "ffn_down", d, sess->scratch_ffn_out);\n'
            '    if (s == GEIST_OK)\n        s = blk_residual_out(')
c.write_text(s[:lo] + f + s[hi:])
