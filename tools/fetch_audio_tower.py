#!/usr/bin/env python3
"""fetch_audio_tower.py — extract the Gemma 4 audio tower from a hosted
checkpoint without downloading the whole thing.

The audio encoder (src/archs/audio_conformer/) loads the Conformer tower and
the multimodal projector by their Hugging Face tensor names:

    model.audio_tower.*     subsample convs, 12 Conformer layers, output proj
    model.embed_audio.*     embedding_projection (audio -> LM residual stream)

Those tensors are ~590 MB of a ~9.7 GB single-file checkpoint. safetensors
puts every tensor's byte range in the JSON header, and Hugging Face serves
HTTP Range requests — so this script reads the header, fetches only the
wanted ranges, and writes a standalone audio_tower.safetensors. Stdlib only:
no torch, no huggingface_hub.

Usage:
    python3 tools/fetch_audio_tower.py \
        [-o audio_bench/audio_tower.safetensors] [--url URL] [--sha256 HEX]

Default source is unsloth/gemma-4-E2B-it (public mirror, apache-2.0 model
card). --sha256 verifies the *output* file, so a changed upstream fails
loudly before a test consumes it (same idea as fetch-llama-model's pin).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://huggingface.co/unsloth/gemma-4-E2B-it/resolve/main/model.safetensors"
)
PREFIXES = ("model.audio_tower.", "model.embed_audio.")
CHUNK = 8 << 20  # 8 MiB read chunks


def fetch_range(url: str, start: int, end: int):
    """Yield chunks of bytes [start, end) via one HTTP Range request."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
    with urllib.request.urlopen(req) as resp:
        if resp.status not in (200, 206):
            raise RuntimeError(f"HTTP {resp.status} for range {start}-{end}")
        got = 0
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            got += len(chunk)
            yield chunk
        if got != end - start:
            raise RuntimeError(f"short read: wanted {end - start} bytes, got {got}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="audio_bench/audio_tower.safetensors")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--sha256", default=None, help="expected hex digest of the output")
    args = ap.parse_args()

    # safetensors: u64-LE header length, then that many bytes of JSON, then data.
    head = b"".join(fetch_range(args.url, 0, 8))
    (hdr_len,) = struct.unpack("<Q", head)
    hdr = json.loads(b"".join(fetch_range(args.url, 8, 8 + hdr_len)))
    data_base = 8 + hdr_len

    picked = {
        name: meta
        for name, meta in hdr.items()
        if name != "__metadata__" and name.startswith(PREFIXES)
    }
    if not picked:
        print(f"error: no tensors match {PREFIXES} at {args.url}", file=sys.stderr)
        return 1

    # New file: tensors in name order, repacked back to back.
    out_hdr, off = {}, 0
    for name in sorted(picked):
        b, e = picked[name]["data_offsets"]
        out_hdr[name] = {
            "dtype": picked[name]["dtype"],
            "shape": picked[name]["shape"],
            "data_offsets": [off, off + (e - b)],
        }
        off += e - b
    hdr_bytes = json.dumps(out_hdr, separators=(",", ":"), sort_keys=True).encode()

    total_mb = off / 1048576
    print(f"{len(picked)} tensors, {total_mb:.0f} MB of data from {args.url}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".part")

    # Fetch in SOURCE-offset order, coalescing near-adjacent tensors into a
    # handful of large Range requests (one TLS handshake each) instead of
    # one request per tensor — 752 round-trips used to dominate wall clock.
    # Each tensor is seek-written to its name-sorted OUTPUT offset.
    prelude = struct.pack("<Q", len(hdr_bytes)) + hdr_bytes
    out_off = {}
    for name in sorted(picked):
        out_off[name] = len(prelude) + out_hdr[name]["data_offsets"][0]

    by_src = sorted(picked, key=lambda n: picked[n]["data_offsets"][0])
    runs, gap = [], 4 << 20
    for name in by_src:
        b, e = picked[name]["data_offsets"]
        if runs and b - runs[-1][1] <= gap:
            runs[-1][1] = max(runs[-1][1], e)
            runs[-1][2].append(name)
        else:
            runs.append([b, e, [name]])
    print(f"{len(runs)} coalesced range requests")

    with open(part, "wb") as f:
        f.write(prelude)
        done = 0
        for rb, re_, names in runs:
            names_iter = iter(names)
            cur = next(names_iter)
            cur_b, cur_e = picked[cur]["data_offsets"]
            pos = rb
            for chunk in fetch_range(args.url, data_base + rb, data_base + re_):
                cstart = pos
                pos += len(chunk)
                while cur is not None and cstart < pos:
                    lo = max(cstart, cur_b)
                    hi = min(pos, cur_e)
                    if lo < hi:
                        f.seek(out_off[cur] + (lo - cur_b))
                        f.write(chunk[lo - cstart : hi - cstart])
                    if pos >= cur_e:
                        done += cur_e - cur_b
                        cur = next(names_iter, None)
                        if cur is not None:
                            cur_b, cur_e = picked[cur]["data_offsets"]
                    else:
                        break
            print(f"  {done / 1048576:7.0f} / {total_mb:.0f} MB", flush=True)

    sha = hashlib.sha256()
    with open(part, "rb") as f:
        while True:
            buf = f.read(CHUNK)
            if not buf:
                break
            sha.update(buf)
    digest = sha.hexdigest()
    print(f"sha256 {digest}")
    if args.sha256 and digest != args.sha256.lower():
        part.unlink()
        print(f"error: sha256 mismatch (expected {args.sha256})", file=sys.stderr)
        return 1
    part.rename(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
