#!/usr/bin/env python3
"""Replay measured Bonsai logits with the final executable after host-only edits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from bench_host import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-results', type=Path, required=True)
    args = parser.parse_args()
    out = args.model_results
    meta = json.loads((out / 'metadata.json').read_text())
    binary = ROOT / 'bin/mac-omp/release/tools/dump_geist_logits'
    rows = []
    for cfg in meta['configs']:
        label = cfg['label']
        env = dict(os.environ, GEIST_DUMP_BACKEND='metal', GEIST_METAL_TENSOR='1',
                   GEIST_METAL_PQ2_MODE=cfg['mode'], GEIST_METAL_PQ2_TILE=str(cfg['tile']),
                   GEIST_METAL_PQ2_SIMDGROUPS=str(cfg['simdgroups']),
                   GEIST_METAL_PQ2_SHARED_ACC=str(int(cfg['shared'])), OMP_NUM_THREADS='4')
        for case in (0, 3, 4):
            ids = out / f'final-ids-{case}-{label}.bin'
            logits = out / f'final-logits-{case}-{label}.bin'
            with (out / f'final-logits-{case}-{label}.err').open('w') as err:
                subprocess.run([str(binary), meta['model']['path'], str(out / f'prompt-{case}.txt'),
                                str(ids), str(logits)], env=env, stderr=err, check=True, timeout=300)
            for kind, value in [('ids', ids), ('logits', logits)]:
                if value.read_bytes() != (out / f'{kind}-{case}-{label}.bin').read_bytes():
                    raise RuntimeError(f'final executable changed {label} {kind}, case {case}')
            rows.append(dict(arm=label, case=case, bit_identical=True,
                             ids_sha256=sha256_file(ids), logits_sha256=sha256_file(logits)))
            print(label, case, 'bit-identical', flush=True)
    (out / 'final-replay.json').write_text(json.dumps(dict(
        binary_sha256=sha256_file(binary), cases=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
