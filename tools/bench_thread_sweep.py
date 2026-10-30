#!/usr/bin/env python3
"""Sweep the thread count for both engines under the cross-engine protocol.

Answers one question: is the matched thread count in a host profile a defensible
choice, or does it sit where only one of the two engines is happy? Everything
else — gate, cooldown, validation, sample retention — is imported from
bench_cross_engine.py so the sweep cannot drift from the protocol it informs.

Two independent runs per (engine, thread count), not two cycles inside one run:
the AMD cell showed llama.cpp's prefill drifting further between runs than
within one, so a curve built from within-run samples alone would be reading its
own noise.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_cross_engine as ce  # noqa: E402

SEQ_LEN = 512  # the shape the matrix cell headlines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geist", required=True, type=Path)
    parser.add_argument("--llama", required=True, type=Path)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--host-profile", required=True)
    parser.add_argument("--protocol", type=Path, default=ce.PROTOCOL_PATH,
                        help="protocol JSON (default: the CPU protocol)")
    parser.add_argument("--model",
                        help="key in the protocol's models map (required for v2 protocols)")
    parser.add_argument("--threads", default="4,8,12,16,24,32")
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.expanduser().resolve()
    protocol = json.loads(protocol_path.read_text())
    profile = protocol["host_profiles"][args.host_profile]
    # The same model and gate resolution as bench_cross_engine.main: v2
    # protocols name their models by key, and a host profile may refine the gate.
    if "models" in protocol:
        if args.model not in protocol["models"]:
            parser.error(f"--model must be one of: {', '.join(sorted(protocol['models']))}")
        expected_model = protocol["models"][args.model]
    else:
        expected_model = protocol["model"]
    model = args.gguf.expanduser().resolve()
    if ce.sha256_file(model) != expected_model["sha256"]:
        parser.error("model SHA-256 mismatch")

    engines = [ce.engine_metadata("geist", "geist", args.geist.expanduser().resolve()),
               ce.engine_metadata("llama.cpp", "llama", args.llama.expanduser().resolve())]
    points = [int(t) for t in args.threads.split(",")]
    gate = {**protocol["host_gate"], **profile.get("host_gate", {})}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = args.out_dir / f"{stamp}_thread_sweep_{args.host_profile}.jsonl"

    with out.open("w") as stream:
        stream.write(json.dumps({
            "kind": "metadata",
            "schema": "geist.benchmark.thread-sweep.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": ce.sha256_file(protocol_path),
            "runner_sha256": ce.sha256_file(Path(__file__).resolve()),
            "host_profile": args.host_profile,
            "profile_in_use": profile,
            "seq_len": SEQ_LEN,
            "thread_points": points,
            "replicates": args.replicates,
            "model": {"file": model.name, "sha256": expected_model["sha256"],
                      **({"key": args.model} if "models" in protocol else {})},
            "host_gate": gate,
            "engines": engines,
            "system": ce.system_metadata(),
        }, sort_keys=True) + "\n")
        stream.flush()

        for threads in points:
            workload = {**protocol["workload"], **profile, "host_profile": args.host_profile,
                        "seq_lens": [SEQ_LEN],
                        "prefill_threads": threads, "decode_threads": threads}
            for replicate in range(args.replicates):
                # Alternate which engine goes first, so a slow warm-up of the
                # machine cannot systematically favour one of them.
                order = engines if replicate % 2 == 0 else list(reversed(engines))
                for engine in order:
                    print(f"threads {threads} rep {replicate + 1}: {engine['label']}",
                          file=sys.stderr, flush=True)
                    try:
                        before = ce.wait_for_quiet(gate)
                        result = (ce.run_geist(engine, model, workload)
                                  if engine["kind"] == "geist"
                                  else ce.run_llama(engine, model, workload, replicate))
                    except (RuntimeError, TimeoutError) as error:
                        print(f"ERROR: {error}", file=sys.stderr)
                        return ce.ERROR
                    row = result["rows"][0]
                    stream.write(json.dumps({
                        "kind": "point", "threads": threads, "replicate": replicate,
                        "engine": engine["label"], "machine_before": before,
                        "machine_after": ce.machine_state(),
                        "prefill_tps": row["prefill_tps"], "decode_tps": row["decode_tps"],
                        # recorded, not enforced: how steady a thread count is
                        # is part of what the sweep answers
                        "repeat_spread_pct": ce.repeat_spread_pct(result),
                    }, sort_keys=True) + "\n")
                    stream.flush()
    print(f"raw: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
