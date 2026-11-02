#!/usr/bin/env python3
"""Verify that PQ2 flags leave other model formats bit-identical.

Run after building dump_geist_logits under the exclusive Metal host lease.
The optional clean main executable also checks the default path across revisions.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
from bench_host import sha256_file
from pq2_quality_cases import CASES

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline-binary', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    binary = ROOT / 'bin/mac-omp/release/tools/dump_geist_logits'
    prompt = args.out / 'prompt.txt'
    prompt.write_text(CASES[0][1])
    records = []
    for model in args.model:
        arms = [('default', binary, 'off'), ('int8', binary, 'staged_int8'),
                ('fp8', binary, 'staged_fp8')]
        if args.baseline_binary:
            arms.insert(0, ('clean-main', args.baseline_binary, 'off'))
        reference = None
        for label, executable, mode in arms:
            prefix = args.out / (model.stem + '-' + label)
            ids, logits = Path(str(prefix) + '-ids.bin'), Path(str(prefix) + '-logits.bin')
            env = dict(os.environ, GEIST_DUMP_BACKEND='metal', GEIST_METAL_TENSOR='1',
                       GEIST_METAL_PQ2_MODE=mode, GEIST_METAL_PQ2_SIMDGROUPS='4',
                       OMP_NUM_THREADS='4')
            with Path(str(prefix) + '.err').open('w') as err:
                subprocess.run([str(executable), str(model), str(prompt), str(ids), str(logits)],
                               env=env, stderr=err, check=True, timeout=300)
            value = (sha256_file(ids), sha256_file(logits))
            if reference is None:
                reference = value
            if value != reference:
                raise RuntimeError(f'{model.name}: {label} changed token IDs or logits')
            records.append(dict(model=model.name, model_sha256=sha256_file(model), arm=label,
                                executable_sha256=sha256_file(executable),
                                ids_sha256=value[0], logits_sha256=value[1], bit_identical=True))
            print(model.name, label, 'bit-identical', flush=True)
    (args.out / 'results.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
