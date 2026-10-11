#!/usr/bin/env python3
"""fetch_bench_model.py — put a benchmark protocol's model into the host cache.

Resolves a models key of a cross-engine protocol to its download URL
(benchmark/model_sources.json, keyed by the protocol's sha256) and keeps the
file in a persistent per-host cache outside any workspace, so a runner
downloads each model once. Prints the cached path on stdout.

    python3 tools/fetch_bench_model.py --model bonsai2-27b-pq2 \
        [--protocol benchmark/cross_engine_gpu_protocol.json] [--cache DIR]

Cache: --cache, else $GEIST_MODEL_CACHE, else ~/geist-models. Downloads go to
<file>.part and resume after an interruption; the file is renamed into place
only after its sha256 matches the protocol. HF_TOKEN is sent when set.
Stdlib only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "benchmark" / "model_sources.json"
CHUNK = 8 << 20
MARGIN = 1 << 30  # keep 1 GiB free beyond the download


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, part: Path, size: int) -> None:
    """Fetch url into part, resuming from part's current length."""
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    if have == size:
        return
    free = shutil.disk_usage(part.parent).free
    if size - have + MARGIN > free:
        raise SystemExit(f"not enough disk for {part.name}: need {size - have} B "
                         f"+ 1 GiB margin, {free} B free in {part.parent}")
    headers = {"Range": f"bytes={have}-"} if have else {}
    if token := os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as resp:
        # A server that ignores Range answers 200 with the whole file.
        mode = "ab" if resp.status == 206 else "wb"
        with part.open(mode) as out:
            while chunk := resp.read(CHUNK):
                out.write(chunk)
    got = part.stat().st_size
    if got != size:
        raise SystemExit(f"short download of {part.name}: {got} of {size} B (rerun to resume)")


def fetch(model_key: str, protocol_path: Path, cache: Path, sources_path: Path = SOURCES) -> Path:
    protocol = json.loads(protocol_path.read_text())
    if model_key not in protocol["models"]:
        raise SystemExit(f"--model must be one of: {', '.join(sorted(protocol['models']))}")
    entry = protocol["models"][model_key]
    source = json.loads(sources_path.read_text())["sources"].get(entry["sha256"])
    if source is None:
        raise SystemExit(f"no source for {model_key} ({entry['sha256']}) in {sources_path.name}")
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / entry["file"]
    if target.exists():
        if sha256_file(target) == entry["sha256"]:
            return target
        print(f"{target} does not match the protocol sha256; refetching", file=sys.stderr)
        target.unlink()
    part = target.with_name(target.name + ".part")
    print(f"fetching {model_key} ({source['size'] / 1e9:.1f} GB) from {source['url']}", file=sys.stderr)
    download(source["url"], part, source["size"])
    got = sha256_file(part)
    if got != entry["sha256"]:
        part.unlink()
        raise SystemExit(f"{model_key}: sha256 {got} != protocol {entry['sha256']}")
    part.rename(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="models key of the protocol")
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "benchmark" / "cross_engine_cpu_protocol.json")
    parser.add_argument("--cache", type=Path,
                        default=Path(os.environ.get("GEIST_MODEL_CACHE") or Path.home() / "geist-models"))
    args = parser.parse_args()
    print(fetch(args.model, args.protocol, args.cache.expanduser()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
