"""Change one group of math intrinsics in the scratch single-token DeltaNet shader."""

from pathlib import Path
import re
import sys

shader = Path(sys.argv[1]) / 'src/backends/metal/metal_shaders.h'
variant = sys.argv[2]
source = shader.read_text()
activation_values = {
    'convolution': 'qwen47_x',
    'exp': 'exp(-fabs(qwen47_x))',
    'reciprocal': '1.0f/(1.0f+exp(-fabs(qwen47_x)))',
}
start = 'static const char metal_dn_decode_source[] =\n'
end = '/* dn_dec_qk_b / dn_dec_v_b'
if source.count(start) != 1 or source.count(end) != 1:
    raise ValueError('Expected one single-token DeltaNet section')
before, decode_and_after = source.split(start, 1)
decode, after = decode_and_after.split(end, 1)
if variant == 'precise-rsqrt':
    decode, count = re.subn(r'\brsqrt\(', 'precise::rsqrt(', decode)
    if count != 4:
        raise ValueError(f'Expected four reciprocal square roots, found {count}')
elif variant == 'precise-exp-log':
    decode, count = re.subn(r'\b(exp|log)\(', r'precise::\1(', decode)
    if count != 4 or decode.count('silu1(') != 4:
        raise ValueError('Expected the pinned DeltaNet exp, log and SiLU call sites')
    decode = ('        "inline float qwen47_silu(float x){float e=precise::exp(-fabs(x));return "\n'
              '        "x>=0.0f?x/(1.0f+e):x*e/(1.0f+e);}\\n"\n'
              + decode.replace('silu1(', 'qwen47_silu('))
elif variant in activation_values:
    cut = 'vc=silu1(vc);qkv[qb+cv]=vc;vv[j]=vc;'
    if decode.count(cut) != 1:
        raise ValueError('Expected one DeltaNet V activation store')
    # The recurrent state consumes vv. Only the traced V scratch changes.
    decode = decode.replace(cut, 'float qwen47_x=vc;vc=silu1(vc);qkv[qb+cv]='
                            + activation_values[variant] + ';vv[j]=vc;')
elif variant in ('divide', 'exp-divide'):
    if decode.count('silu1(') != 4:
        raise ValueError('Expected four single-token SiLU sites')
    exp = 'precise::exp' if variant == 'exp-divide' else 'exp'
    helper = (f'        "inline float qwen47_silu_div(float x){{float e={exp}(-fabs(x));"\n'
              '        "return precise::divide(x>=0.0f?x:x*e,1.0f+e);}\\n"\n')
    decode = helper + decode.replace('silu1(', 'qwen47_silu_div(')
else:
    raise ValueError(f'Unknown variant: {variant}')
shader.write_text(before + start + decode + end + after)
