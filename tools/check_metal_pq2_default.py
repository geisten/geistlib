#!/usr/bin/env python3
"""Compare clean-main Bonsai logits with the recorded default candidate arm."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from bench_host import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-results', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for case in (0, 3, 4):
        out = args.model_results
        ids, logits = out / f'ids-{case}-clean-main.bin', out / f'logits-{case}-clean-main.bin'
        env = dict(os.environ, GEIST_DUMP_BACKEND='metal', GEIST_METAL_TENSOR='1',
                   GEIST_METAL_PQ2_MODE='off', OMP_NUM_THREADS='4')
        with (out / f'logits-{case}-clean-main.err').open('w') as err:
            subprocess.run([str(args.binary), str(args.model), str(out / f'prompt-{case}.txt'),
                            str(ids), str(logits)], env=env, stderr=err, check=True, timeout=300)
        for kind, value in [('ids', ids), ('logits', logits)]:
            if value.read_bytes() != (out / f'{kind}-{case}-baseline.bin').read_bytes():
                raise RuntimeError(f'default Bonsai {kind} changed for case {case}')
        rows.append(dict(case=case, bit_identical=True, ids_sha256=sha256_file(ids),
                         logits_sha256=sha256_file(logits)))
        print('Bonsai default case', case, 'bit-identical', flush=True)
    (args.model_results / 'default-isolation.json').write_text(json.dumps(dict(
        clean_main_binary_sha256=sha256_file(args.binary), cases=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
