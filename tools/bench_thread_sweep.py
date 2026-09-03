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
    parser.add_argument("--threads", default="4,8,12,16,24,32")
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(ce.PROTOCOL_PATH.read_text())
    profile = protocol["host_profiles"][args.host_profile]
    model = args.gguf.expanduser().resolve()
    if ce.sha256_file(model) != protocol["model"]["sha256"]:
        parser.error("model SHA-256 mismatch")

    engines = [ce.engine_metadata("geist", "geist", args.geist.expanduser().resolve()),
               ce.engine_metadata("llama.cpp", "llama", args.llama.expanduser().resolve())]
    points = [int(t) for t in args.threads.split(",")]
    gate = protocol["host_gate"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = args.out_dir / f"{stamp}_thread_sweep_{args.host_profile}.jsonl"

    with out.open("w") as stream:
        stream.write(json.dumps({
            "kind": "metadata",
            "schema": "geist.benchmark.thread-sweep.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": ce.sha256_file(ce.PROTOCOL_PATH),
            "runner_sha256": ce.sha256_file(Path(__file__).resolve()),
            "host_profile": args.host_profile,
            "profile_in_use": profile,
            "seq_len": SEQ_LEN,
            "thread_points": points,
            "replicates": args.replicates,
            "model": {"file": model.name, "sha256": protocol["model"]["sha256"]},
            "engines": engines,
            "system": ce.system_metadata(),
        }, sort_keys=True) + "\n")
        stream.flush()

        for threads in points:
            workload = {**protocol["workload"], "seq_lens": [SEQ_LEN],
                        "prefill_threads": threads, "decode_threads": threads,
                        "expect_backend": profile["expect_backend"]}
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
                    }, sort_keys=True) + "\n")
                    stream.flush()
    print(f"raw: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
