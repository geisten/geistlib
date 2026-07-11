#!/usr/bin/env python3
"""eval_agent_judge.py — advisory answer-coherence judge for the agent eval.

A second AI reads the agent's final answers: for every case in the JSONL dump
written by `bench_agent_eval --dump` it asks a local Ollama model whether the
answer is a COHERENT, plausible response to the request. This closes the gap
the mechanical `expect` substring check cannot: "report report Wohner der
Heimat report.md" contains the expected token but is not an answer.

Advisory ONLY — the exit code is always 0 and nothing here gates CI. LLM
judges drift with model updates and misjudge; the deterministic substring
gate stays authoritative, the judge tells you which answers to eyeball.

Usage:
  make bench-agent-judge                      # dump + judge (JUDGE_MODEL=...)
  python3 tools/eval_agent_judge.py --answers bench_runs/agent_eval/answers.jsonl \
      [--model gemma4:26b] [--host http://localhost:11434]

Stdlib only (urllib against the Ollama HTTP API).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

PROMPT = """You are reviewing the answers of a small on-device assistant.
Judge ONLY whether the answer is a coherent, plausible response to the request
— not whether it is factually perfect. Tool output quoted verbatim (a file
listing, a page excerpt, a note) counts as a coherent answer. Reply with
exactly one line: YES: <short reason>  or  NO: <short reason>

Request: {req}
Answer: {answer}"""


def judge(host: str, model: str, req: str, answer: str, timeout: float) -> str:
    body = json.dumps({
        "model": model,
        "prompt": PROMPT.format(req=req, answer=answer),
        "stream": False,
        "options": {"temperature": 0},
    }).encode()
    r = urllib.request.Request(f"{host}/api/generate", data=body,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.load(resp).get("response", "").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True, help="JSONL from bench_agent_eval --dump")
    ap.add_argument("--model", default="gemma4:26b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--timeout", type=float, default=120.0)
    a = ap.parse_args()

    try:
        cases = [json.loads(l) for l in open(a.answers) if l.strip()]
    except OSError as e:
        print(f"cannot read {a.answers}: {e}", file=sys.stderr)
        return 0  # advisory: never fail the build
    if not cases:
        print("no answers to judge")
        return 0

    coherent, flagged, errors = 0, [], 0
    for c in cases:
        try:
            verdict = judge(a.host, a.model, c["req"], c["answer"], a.timeout)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            print(f"[judge] {c['id']:<10} ERROR {e}")
            errors += 1
            continue
        line = verdict.splitlines()[0] if verdict else "NO: empty judge reply"
        ok = line.upper().startswith("YES")
        coherent += ok
        if not ok:
            flagged.append(c["id"])
        print(f"[judge] {c['id']:<10} {line}")

    judged = len(cases) - errors
    print(f"\nJUDGE model={a.model} coherent={coherent}/{judged}"
          + (f" errors={errors}" if errors else "")
          + (f" flagged: {', '.join(flagged)}" if flagged else ""))
    print("advisory only — eyeball the flagged answers; the substring gate stays authoritative")
    return 0


if __name__ == "__main__":
    sys.exit(main())
