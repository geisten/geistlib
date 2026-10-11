#!/usr/bin/env python3
"""tests/test_fetch_bench_model_py.py — the benchmark model fetcher.

Every protocol model has a pinned download source; a fetch resumes a
partial file over HTTP Range, caches the result, and refuses content whose
sha256 differs from the protocol. Hermetic: a local HTTP server, no network.
"""
from __future__ import annotations

import hashlib
import http.server
import json
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import fetch_bench_model as fbm  # noqa: E402

fails = 0


def expect(cond: bool, label: str) -> None:
    global fails
    print(f"  {'✓' if cond else '✗'} {label}")
    fails += not cond


BLOB = bytes(range(256)) * 4096  # 1 MiB
served = []


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        start = int(self.headers.get("Range", "bytes=0-")[6:].rstrip("-") or 0)
        served.append(start)
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(BLOB) - start))
        self.end_headers()
        self.wfile.write(BLOB[start:])

    def log_message(self, *args):
        pass


sources = json.loads(fbm.SOURCES.read_text())["sources"]
for name in ("cross_engine_cpu_protocol.json", "cross_engine_gpu_protocol.json"):
    models = json.loads((ROOT / "benchmark" / name).read_text())["models"]
    expect(all(m["sha256"] in sources and sources[m["sha256"]]["model"] == k
               for k, m in models.items()), f"{name}: every model has a source")

server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{server.server_port}/m.gguf"

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    good = hashlib.sha256(BLOB).hexdigest()
    (tmp / "p.json").write_text(json.dumps({"models": {
        "m": {"file": "m.gguf", "sha256": good},
        "bad": {"file": "bad.gguf", "sha256": "0" * 64}}}))
    (tmp / "s.json").write_text(json.dumps({"sources": {
        good: {"model": "m", "url": url, "size": len(BLOB)},
        "0" * 64: {"model": "bad", "url": url, "size": len(BLOB)}}}))
    cache = tmp / "cache"
    cache.mkdir()
    (cache / "m.gguf.part").write_bytes(BLOB[:1000])

    path = fbm.fetch("m", tmp / "p.json", cache, tmp / "s.json")
    expect(path.read_bytes() == BLOB, "resumed download is complete")
    expect(served == [1000], "resume asked for the missing range only")
    expect(not (cache / "m.gguf.part").exists(), "partial file renamed into place")

    fbm.fetch("m", tmp / "p.json", cache, tmp / "s.json")
    expect(served == [1000], "cached file is not downloaded again")

    try:
        fbm.fetch("bad", tmp / "p.json", cache, tmp / "s.json")
        expect(False, "sha256 mismatch is refused")
    except SystemExit:
        expect(not (cache / "bad.gguf").exists() and not (cache / "bad.gguf.part").exists(),
               "sha256 mismatch is refused and leaves nothing behind")

server.shutdown()
print("PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
