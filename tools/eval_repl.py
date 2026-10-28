"""eval_repl.py — eval_geist (tools/eval_geist.c) as a persistent subprocess.

eval_geist loads a GGUF once and answers one line per command on stdout
("OK ..." or "ERR ..."). Its stderr goes to a log file or is inherited, never
to a pipe nobody reads: a full pipe blocks the child mid-run.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

TOK_TEXT_CAP = 65536  # eval_geist unescapes TOK text into a buffer of this many bytes


class EvalRepl:
    """`command` returns the fields after "OK" and raises RuntimeError on
    anything else; `bos` is the model's BOS id, or None when its metadata says
    not to add one; `startup_ms` is the time to READY."""

    def __init__(self, binary, model, log: Path | None = None, args: list[str] = ()):
        self.log = log.open("w") if log is not None else None
        start = time.perf_counter()
        try:
            self.process = subprocess.Popen([str(binary), str(model), *args],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.log, text=True, bufsize=1)
            while True:
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError("eval_geist exited before READY"
                                       + (f"; see {log}" if log is not None else ""))
                if line.strip() == "READY":
                    break
            self.startup_ms = (time.perf_counter() - start) * 1000
            parts = self.command("BOS")
            if len(parts) != 2 or parts[0] not in ("0", "1"):
                raise ValueError("invalid tokenizer BOS policy")
            bos = int(parts[1])
            if parts[0] == "1" and bos < 0:
                raise ValueError("invalid tokenizer BOS token")
            self.bos = bos if parts[0] == "1" else None
        except BaseException:
            self.close()
            raise

    def command(self, text: str) -> list[str]:
        self.process.stdin.write(text + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline().strip().split()
        if not response or response[0] != "OK":
            raise RuntimeError(f"eval_geist {text[:40]!r} failed: {' '.join(response)[:200]}")
        return response[1:]

    def encode(self, text: str) -> list[int]:
        """Token ids of text (no BOS) from the model's own GGUF tokenizer."""
        if "\0" in text or len(text.encode("utf-8")) >= TOK_TEXT_CAP:
            raise ValueError("prompt does not fit the lossless TOK protocol")
        # The REPL is line-oriented: newlines and tabs travel escaped, and so
        # does CR -- a raw CR ending the line would be stripped (MMLU text keeps
        # its CRs, college_mathematics exemplars, #171).
        escaped = (text.replace("\\", "\\\\").replace("\n", "\\n")
                   .replace("\t", "\\t").replace("\r", "\\r"))
        response = self.command("TOK " + escaped)
        if not response:
            raise ValueError("missing token count")
        ids = [int(x) for x in response[1:]]
        if int(response[0]) != len(ids) or not ids or any(i < 0 for i in ids):
            raise ValueError("invalid tokenizer result")
        return ids

    def close(self) -> None:
        if hasattr(self, "process"):
            if self.process.poll() is None:
                try:
                    self.process.stdin.write("QUIT\n")
                    self.process.stdin.flush()
                    self.process.wait(timeout=20)
                except (BrokenPipeError, ValueError, subprocess.TimeoutExpired):
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
        if self.log is not None:
            self.log.close()
