#!/usr/bin/env python3
"""fetch_bench_model.py — put a benchmark protocol's model into the host cache.

Resolves a models key of a cross-engine protocol to its download URL
(benchmark/model_sources.json, keyed by the protocol's sha256) and keeps the
file in a persistent per-host cache outside any workspace, so a runner
downloads each model once. Prints the cached path on stdout.

    python3 tools/fetch_bench_model.py --model bonsai2-27b-pq2 \
        [--protocol benchmark/cross_engine_gpu_protocol.json] [--cache DIR]
    python3 tools/fetch_bench_model.py --url URL --sha256 SHA [--cache DIR]

Cache: --cache, else $GEIST_MODEL_CACHE, else ~/geist-models. Downloads go to
<file>.part and resume after an interruption; the file is renamed into place
only after its sha256 matches the protocol (or --sha256). HF_TOKEN is sent
when set. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_host import sha256_file  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "benchmark" / "model_sources.json"
CHUNK = 8 << 20
MARGIN = 1 << 30  # keep 1 GiB free beyond the download


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
    headers = auth_headers() | ({"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as resp:
        # A server that ignores Range answers 200 with the whole file.
        mode = "ab" if resp.status == 206 else "wb"
        with part.open(mode) as out:
            while chunk := resp.read(CHUNK):
                out.write(chunk)
    got = part.stat().st_size
    if got != size:
        raise SystemExit(f"short download of {part.name}: {got} of {size} B (rerun to resume)")


def auth_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers=auth_headers())
    with urllib.request.urlopen(request) as resp:
        return int(resp.headers["Content-Length"])


def fetch_file(url: str, sha256: str, target: Path, size: int | None = None) -> Path:
    """Put url into target unless target already has sha256; size from the
    server when not given. A mismatching cached file is fetched again."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) == sha256:
            return target
        print(f"{target} does not match sha256 {sha256}; refetching", file=sys.stderr)
        target.unlink()
    if size is None:
        size = remote_size(url)
    part = target.with_name(target.name + ".part")
    print(f"fetching {target.name} ({size / 1e9:.1f} GB) from {url}", file=sys.stderr)
    download(url, part, size)
    got = sha256_file(part)
    if got != sha256:
        part.unlink()
        raise SystemExit(f"{target.name}: sha256 {got} != expected {sha256}")
    part.rename(target)
    return target


def fetch(model_key: str, protocol_path: Path, cache: Path, sources_path: Path = SOURCES) -> Path:
    protocol = json.loads(protocol_path.read_text())
    if model_key not in protocol["models"]:
        raise SystemExit(f"--model must be one of: {', '.join(sorted(protocol['models']))}")
    entry = protocol["models"][model_key]
    source = json.loads(sources_path.read_text())["sources"].get(entry["sha256"])
    if source is None:
        raise SystemExit(f"no source for {model_key} ({entry['sha256']}) in {sources_path.name}")
    return fetch_file(source["url"], entry["sha256"], cache / entry["file"], source["size"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--model", help="models key of the protocol")
    what.add_argument("--url", help="any GGUF URL, cached by its file name (needs --sha256)")
    parser.add_argument("--sha256", help="expected sha256 of --url")
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "benchmark" / "cross_engine_cpu_protocol.json")
    parser.add_argument("--cache", type=Path,
                        default=Path(os.environ.get("GEIST_MODEL_CACHE") or Path.home() / "geist-models"))
    args = parser.parse_args()
    cache = args.cache.expanduser()
    if args.url:
        if not args.sha256:
            parser.error("--url needs --sha256")
        print(fetch_file(args.url, args.sha256.lower(), cache / Path(args.url).name))
    else:
        print(fetch(args.model, args.protocol, cache))
    return 0


if __name__ == "__main__":
    sys.exit(main())
