#!/usr/bin/env python3
"""validate_greedy.py — actual greedy generation test via repeated full forward.

Inefficient (no KV cache for generation; re-runs full forward each step) but
the right correctness test. Compares first N generated tokens against
greedy_T2.json.

Run:
    source ../gemma-4-E2B-it/.venv/bin/activate
    python validate_greedy.py [--n_steps 5]
"""
import argparse, json, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

SCRIPT = Path(__file__).parent.resolve()
SAFE = SCRIPT.parent / "gemma-4-E2B-it" / "model.safetensors"
TB = SCRIPT / "test_full_logits"
VOCAB = 262144


def forward_logits(token_ids: np.ndarray) -> np.ndarray:
    """Returns full logits (seq, VOCAB) by invoking test_full_logits."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ids_path = tmp / "ids.bin"; out_path = tmp / "logits.bin"
        ids_path.write_bytes(token_ids.astype(np.int32).tobytes())
        proc = subprocess.run(
            [str(TB), str(SAFE), str(ids_path), str(out_path)],
            capture_output=True, text=True, check=True, timeout=180,
        )
        return np.frombuffer(out_path.read_bytes(), dtype=np.float32).reshape(-1, VOCAB)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=5,
                        help="How many tokens to generate (HF reference is 25 to EOS).")
    args = parser.parse_args()

    ref = json.loads((ROOT_DIR / "greedy_T2.json").read_text())
    prompt_ids = np.array(ref["prompt_token_ids"], dtype=np.int32)
    expected = np.array(ref["generated_token_ids"], dtype=np.int32)
    n_gen = min(args.n_steps, len(expected))
    print(f"Prompt ({len(prompt_ids)} tok): {ref['prompt']!r}")
    print(f"Expected first {n_gen} tokens: {expected[:n_gen].tolist()}")
    print()

    cur = prompt_ids.copy()
    generated = []
    matched = 0
    for step in range(n_gen):
        t0 = time.time()
        logits = forward_logits(cur)
        last = logits[-1]                                      # (VOCAB,)
        next_tok = int(np.argmax(last))
        elapsed = time.time() - t0

        exp = int(expected[step])
        ok = next_tok == exp
        matched += int(ok)
        flag = "✓" if ok else "✗"
        print(f"  step {step+1:2d}: c={next_tok:>6} expected={exp:>6} "
              f"{flag}  ({elapsed:.1f}s, seq={len(cur)})")
        if not ok:
            # Show top-5 to see how close we were
            top5 = np.argpartition(-last, 5)[:5]
            top5_sorted = top5[np.argsort(-last[top5])]
            print(f"           top-5: {top5_sorted.tolist()}")
        generated.append(next_tok)
        cur = np.concatenate([cur, [next_tok]])

    print(f"\n{matched}/{n_gen} tokens match HF greedy reference")
    print(f"  generated: {generated}")
    print(f"  expected:  {expected[:n_gen].tolist()}")
    return 0 if matched == n_gen else 1


if __name__ == "__main__":
    sys.exit(main())
n__":
    sys.exit(main())
