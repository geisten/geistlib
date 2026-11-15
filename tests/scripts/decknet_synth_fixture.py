#!/usr/bin/env python3
"""decknet_synth_fixture.py — writes tests/data/decknet_synth/ for test_decknet_unit (#532).

A parity fixture that can be public: a decknet file whose every number is
INVENTED here, an image pair of noise and shapes, and what the numpy
reference computes for them.

    synth.gguf     a network that describes no device: three slots "PQR", a
                   160 x 120 image, a 58 x 42 crop, 16 homographies, kernels,
                   1x1 weights, pooling rows, thresholds and gains drawn from
                   a seeded generator or picked by hand below
    ref.ppm        smooth noise and a few shapes
    cur.ppm        the same, darker, with other shapes on top
    expected.txt   the reference's outputs, per case

Nothing in it comes from geistbox's deck model, its camera or its exporter;
no value below is a geistbox value or derived from one. The network makes no
sense as a deck check. It is there for the arithmetic: the same loader and
the same forward pass, with an asymmetric kernel, uneven pooling weights and
other sizes than the real file has, so every output of every slot is a
number the C code and the reference have to agree on.

The reference is the judge, and only its input is invented. geistbox's
decknet.py keeps its network in module globals, so this script replaces
every one of them the forward pass reads with the invented value (`install`)
and then calls its functions (warp, normalise, changed, forward) unchanged.
A global left at geistbox's value would make the reference and the C code
compute two networks, and the parity test would fail: the test is also the
check that nothing real went into expected.txt. The file is written here,
not by decknet.export, which would write geistbox's homographies.

Cases in expected.txt:

    file    both images through the file's homographies
    moved   the current image through other (equally invented) homographies
    dark    the current image at half brightness: not checkable

    python3 tests/scripts/decknet_synth_fixture.py --geistbox ../geistbox/software

Needs numpy. Regenerating gives the same bytes.
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

SEED = 532
SLOTS = 'PQR'
IMG_W, IMG_H = 160, 120
CROP_W, CROP_H = 58, 42
BG_W, BG_H = 14, 30
PLANES = (0.0, 9.0, 4.0, 2.5, 6.0)              # five distinct numbers; the pass uses them as names only
BG_NOMINAL = (0.61, 0.56, 0.52)
N_TIPS, N_TUBES = 18, 8                          # the reference's forward() splits the tube units at 8
# Thresholds picked so that the units of the noise images spread over 0..1 (some of them negative: the
# invented pools are differences around zero).
SCALARS = dict(t_change=0.21, g_change=3.0, t_wall=0.31, g_wall=4.0, t_tip=-0.2, t_dark=-0.12, t_tube=0.01,
               g_unit=4.0, t_white=0.29, g_white=6.0, gain_min=0.7, gain_max=1.4, bg_rest_max=0.06,
               occ=0.4, smooth=0.45)
EXPOSURE = (0.90, 0.86, 0.93)                    # current image against the other
STRIP = (4, 6)                                   # where homography 0 puts the strip in the image (x, y)


def network(rng):
    """Every tensor of the file, F32, as numpy arrays in numpy shapes; pools dense (units, CROP_H, CROP_W)."""
    f32 = np.float32

    def homography(x0, y0, x1, y1):
        """Crop onto about the image rectangle (x0, y0)..(x1, y1), with a little shear and perspective."""
        sx, sy = (x1 - x0) / CROP_W, (y1 - y0) / CROP_H
        return np.array([[sx, rng.uniform(-0.06, 0.06), x0 + rng.uniform(0, 3)],
                         [rng.uniform(-0.06, 0.06), sy, y0 + rng.uniform(0, 3)],
                         [rng.uniform(-4e-4, 4e-4), rng.uniform(-4e-4, 4e-4), 1.0]])

    hs = [np.array([[1.0, 0.0, STRIP[0] + 0.25], [0.0, 1.0, STRIP[1] + 0.75], [0.0, 0.0, 1.0]])]
    for s in range(len(SLOTS)):                  # each slot somewhere right of the strip, each plane a bit elsewhere
        x0, y0 = 24 + 40 * s + rng.uniform(0, 6), 8 + rng.uniform(0, 30)
        for _ in PLANES:
            hs.append(homography(x0 + rng.uniform(-4, 4), y0 + rng.uniform(-4, 4),
                                 x0 + rng.uniform(43, 76), y0 + rng.uniform(40, 66)))

    def kernel(side, norm):                      # not symmetric, sum 0
        k = rng.normal(size=(side, side))
        k -= k.mean()
        return k * norm / np.abs(k).sum()

    def pool(units, lo, hi):                     # each unit: a few pixels, uneven weights that sum to 1
        m = np.zeros((units, CROP_H * CROP_W))
        for u in range(units):
            px = rng.choice(CROP_H * CROP_W, size=rng.integers(lo, hi + 1), replace=False)
            w = rng.uniform(0.2, 1.0, size=px.size)
            m[u, px] = w / w.sum()
        return m.reshape(units, CROP_H, CROP_W)

    t = {'decknet.homography': np.array(hs), 'decknet.planes_mm': np.array(PLANES),
         'decknet.bg_nominal': np.array(BG_NOMINAL),
         'decknet.conv.blob': kernel(7, 3.0), 'decknet.conv.tube': kernel(9, 4.0),
         'decknet.change.w': rng.uniform(-1, 1, size=(5, 6)), 'decknet.wall.w': rng.uniform(-1, 1, size=(3, 6)),
         'decknet.pool.on': pool(N_TIPS, 3, 7), 'decknet.pool.off': pool(N_TIPS, 3, 7),
         'decknet.pool.tube': pool(2 * N_TUBES, 2, 5), 'decknet.pool.wall': pool(1, 30, 30),
         'decknet.pool.occ_front': pool(1, 48, 48), 'decknet.pool.occ_rear': pool(1, 70, 70)}
    # A 1x1 row that sees the difference of the two images, so that equal strips count as unchanged.
    t['decknet.change.w'][0] = (0.9, 1.1, 0.7, -0.9, -1.1, -0.7)
    t['decknet.change.w'][1] = -t['decknet.change.w'][0]
    t['decknet.change.w'][2:] *= 0.025
    return {k: v.astype(f32) for k, v in t.items()}


def write_gguf(path, t):
    """GGUF v3, general.architecture = decknet; pools as rows (unit, pixel, weight) in np.nonzero order."""
    meta = {'general.architecture': 'decknet', 'general.name': 'decknet synthetic test network',
            'decknet.slots': SLOTS, 'decknet.image.width': IMG_W, 'decknet.image.height': IMG_H,
            'decknet.crop.width': CROP_W, 'decknet.crop.height': CROP_H, 'decknet.bg.width': BG_W,
            'decknet.bg.height': BG_H, **{f'decknet.{k}': float(v) for k, v in SCALARS.items()}}
    st = lambda s: struct.pack('<Q', len(s.encode())) + s.encode()
    head = struct.pack('<4sIQQ', b'GGUF', 3, len(t), len(meta))
    for k, v in meta.items():                    # value types: 8 string, 6 F32, 4 U32
        head += st(k) + (struct.pack('<I', 8) + st(v) if isinstance(v, str) else
                         struct.pack('<If', 6, v) if isinstance(v, float) else struct.pack('<II', 4, v))
    data = b''
    for name, a in t.items():
        if '.pool.' in name:
            m = a.reshape(a.shape[0], -1)
            u, px = np.nonzero(m)
            a = np.stack([u, px, m[u, px]], 1)
        a = np.ascontiguousarray(a, '<f4')
        head += st(name) + struct.pack(f'<I{a.ndim}QIQ', a.ndim, *a.shape[::-1], 0, len(data))
        data += a.tobytes() + b'\0' * (-a.nbytes % 32)
    Path(path).write_bytes(head + b'\0' * (-len(head) % 32) + data)


def install(d, t):
    """Replace every global of the reference that its forward pass reads with the invented network."""
    s = {k: float(np.float32(v)) for k, v in SCALARS.items()}        # as the C loader reads them
    d.H, d.W, d.BG_SHAPE, d.PLANES = CROP_H, CROP_W, (BG_H, BG_W), PLANES
    _, d.H_PLATE, d.H_DEEP, d.H_HOLE, d.H_TIP = PLANES
    d.BG_NOMINAL = t['decknet.bg_nominal']
    d.K_BLOB, d.K_TUBE = t['decknet.conv.blob'], t['decknet.conv.tube']
    d.W_A1, d.W_WALL = t['decknet.change.w'], t['decknet.wall.w']
    d.M_ON, d.M_OFF, d.M_TUBE = t['decknet.pool.on'], t['decknet.pool.off'], t['decknet.pool.tube']
    d.M_WALL = t['decknet.pool.wall'][0]
    d.M_REGION = {'occ_front': t['decknet.pool.occ_front'][0], 'occ_rear': t['decknet.pool.occ_rear'][0]}
    d.T_CHANGE, d.G_CHANGE, d.T_WALL, d.G_WALL = s['t_change'], s['g_change'], s['t_wall'], s['g_wall']
    d.T_TIP, d.T_DARK, d.T_TUBE, d.G_UNIT = s['t_tip'], s['t_dark'], s['t_tube'], s['g_unit']
    d.T_WHITE, d.G_WHITE = s['t_white'], s['g_white']
    d.GAIN, d.BG_REST_MAX = (s['gain_min'], s['gain_max']), s['bg_rest_max']


def images(rng):
    """ref and cur, IMG_H x IMG_W x 3 uint8: smooth noise and shapes. The strip's rectangle stays as it is."""
    yy, xx = np.mgrid[:IMG_H, :IMG_W]
    ref = np.full((IMG_H, IMG_W, 3), 0.55)
    for _ in range(40):                          # low-frequency noise
        fx, fy, ph = rng.uniform(0.025, 0.45), rng.uniform(0.025, 0.45), rng.uniform(0, 6.28, size=3)
        ref += 0.035 * np.sin(fx * xx[..., None] + fy * yy[..., None] + ph)
    ref += rng.normal(0, 0.025, size=ref.shape)   # pixel noise
    cur = ref.copy()
    free = xx >= STRIP[0] + BG_W + 2             # right of the strip
    for _ in range(64):                          # discs, lighter or darker or coloured, on the current image
        cx, cy, r = rng.uniform(0, IMG_W), rng.uniform(0, IMG_H), rng.uniform(1.5, 9)
        on = free & ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r)
        cur[on] = cur[on] * rng.uniform(0.35, 1.0) + rng.uniform(-0.1, 0.45, size=3)
    for _ in range(11):                          # and a few on both
        cx, cy, r = rng.uniform(0, IMG_W), rng.uniform(0, IMG_H), rng.uniform(3, 11)
        on = free & ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r)
        shade = rng.uniform(-0.2, 0.28, size=3)
        ref[on] += shade
        cur[on] += shade
    to_bytes = lambda a: np.clip(np.rint(a * 255), 0, 255).astype(np.uint8)
    return to_bytes(ref), to_bytes(cur * np.array(EXPOSURE))


def reference(d, cur, ref, hs, ref_hs):
    """The reference's read_deck without its head and without its slot table: units per slot."""
    cur, ref = cur.astype(np.float32) / 255, ref.astype(np.float32) / 255
    a, b = d.warp(cur, hs[0], d.BG_SHAPE), d.warp(ref, ref_hs[0], d.BG_SHAPE)
    ga, gb = d.BG_NOMINAL / np.maximum(a.mean((0, 1)), 1e-3), d.BG_NOMINAL / np.maximum(b.mean((0, 1)), 1e-3)
    out = dict(gain=ga / gb, bg_changed=d.changed(np.concatenate([a * ga, b * gb], -1)).mean())
    both = d.normalise(cur, ref, hs, ref_hs)     # the reference's own verdict
    out['checkable'] = both is not None
    if both is None:
        return out
    cur, ref = both
    n = len(PLANES)
    out['slots'] = {}
    for i, s in enumerate(SLOTS):
        planes = {h: d.warp(cur, hs[1 + n * i + k], (d.H, d.W)) for k, h in enumerate(PLANES)}
        out['slots'][s] = d.forward(planes, d.warp(ref, ref_hs[1 + n * i], (d.H, d.W)))
    return out


UNITS = ('occ_front', 'occ_rear', 'wall_front', 'rack', 'plate', 'deep', 'tube_n', 'tube_r')


def write_case(f, name, r, extra=()):
    num = lambda xs: ' '.join(f'{float(x):.9g}' for x in np.ravel(xs))
    f.write(f'case {name}\n')
    for line in extra:
        f.write(line + '\n')
    f.write(f"checkable {int(r['checkable'])}\ngain {num(r['gain'])}\nbg_changed {num(r['bg_changed'])}\n")
    for s, o in r.get('slots', {}).items():
        f.write(f"slot {s} {num([o[k] for k in UNITS])}\n {num(o['tips'])}\n {num(o['tubes'])}\n {num(o['tubes_rot'])}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--geistbox', required=True, type=Path,
                    help='geistbox checkout, its software/ directory (for the reference forward pass only)')
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'data' / 'decknet_synth')
    args = ap.parse_args()
    sys.dont_write_bytecode = True               # leave the other checkout as it is
    sys.path.insert(0, str(args.geistbox))
    import geistbox.decknet as d

    rng = np.random.default_rng(SEED)
    t = network(rng)
    ref, cur = images(rng)
    hs = t['decknet.homography']
    moved = hs.copy()                            # the strip stays; every slot plane goes somewhere else
    moved[1:, :2, 2] += rng.uniform(-2.5, 2.5, size=(len(hs) - 1, 2)).astype(np.float32)
    moved[1:, 0, 0] *= rng.uniform(0.9, 1.1, size=len(hs) - 1).astype(np.float32)

    args.out.mkdir(parents=True, exist_ok=True)
    write_gguf(args.out / 'synth.gguf', t)
    for name, img in (('ref.ppm', ref), ('cur.ppm', cur)):
        (args.out / name).write_bytes(b'P6\n%d %d\n255\n' % (IMG_W, IMG_H) + img.tobytes())

    install(d, t)
    cases = {'file': reference(d, cur, ref, hs, hs), 'moved': reference(d, cur, ref, moved, hs),
             'dark': reference(d, cur >> 1, ref, hs, hs)}
    with open(args.out / 'expected.txt', 'w') as f:
        f.write('# decknet reference outputs for the invented network synth.gguf, written by\n'
                '# tests/scripts/decknet_synth_fixture.py; do not edit.\n'
                '# slot <letter> occ_front occ_rear wall_front rack plate deep tube_n tube_r, then tips, tubes, tubes_rot\n')
        write_case(f, 'file', cases['file'])
        write_case(f, 'moved', cases['moved'],
                   ['homographies ' + ' '.join(f'{float(x):.9g}' for x in moved.ravel())])
        write_case(f, 'dark', cases['dark'], ['cur_half 1'])

    for name, r in cases.items():
        if 'slots' not in r:
            print(f'{name}: not checkable, gain {r["gain"]}, bg_changed {r["bg_changed"]:.4f}')
            continue
        vals = np.concatenate([np.r_[[o[k] for k in UNITS], o['tips'], o['tubes'], o['tubes_rot']]
                               for o in r['slots'].values()])
        print(f'{name}: checkable, gain {r["gain"]}, bg_changed {r["bg_changed"]:.4f}; {vals.size} outputs, '
              f'{int(((vals > 1e-3) & (vals < 1 - 1e-3)).sum())} strictly inside (0, 1), '
              f'{int((vals >= 1 - 1e-3).sum())} at 1, {int((vals <= 1e-3).sum())} at 0')
        for s, o in r['slots'].items():
            print(f'  {s}: ' + ' '.join(f'{k}={o[k]:.3f}' for k in UNITS)
                  + f' tips {np.round(o["tips"], 2)} tubes {np.round(o["tubes"], 2)} rot {np.round(o["tubes_rot"], 2)}')


if __name__ == '__main__':
    main()
