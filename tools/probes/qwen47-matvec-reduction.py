"""Select an explicit SIMD reduction in a scratch copy of the F32 probe shader."""

from pathlib import Path
import sys

shader = Path(sys.argv[1]) / 'src/backends/metal/metal_shaders.h'
order = sys.argv[2]
steps = {'descending': (16, 8, 4, 2, 1), 'ascending': (1, 2, 4, 8, 16)}[order]
source = shader.read_text()
anchor = 'return simd_sum((a.x+a.y)+(a.z+a.w));'
if source.count(anchor) != 1:
    raise ValueError('Expected one mv_row4 reduction in the pinned engine')
lines = ['float t=(a.x+a.y)+(a.z+a.w);']
lines.extend(f't+=simd_shuffle_xor(t,ushort({step}));' for step in steps)
lines.append('return t;')
shader.write_text(source.replace(anchor, '"\n        "'.join(lines)))
