#!/usr/bin/env python3
"""required_checks.py — keep main's required status checks in step with CI.

main's branch protection on git.geisten.net requires every CI job by its
exact status name (a glob such as "CI / *" matched no status while a new
commit's CI had not started, and Gitea merged on it). An exact list must
follow the workflow, so it lives in .gitea/required-checks.txt and:

    check    derive the job names from .gitea/workflows/ci.yml (matrix
             include entries expanded) and compare them with the file;
             exit 1 on any difference. Runs in CI, needs no token.
    print    print the derived list (to regenerate the file).
    verify   compare the file with the live branch protection.
    sync     write the file's list into the live branch protection.

verify/sync read GITEA_TOKEN (repository admin) and talk to
https://git.geisten.net/api/v1/repos/geisten/geistlib/branch_protections/main.

Stdlib only: the CI image has no PyYAML, and the parser needs just the job
names and the scalar fields of matrix include entries.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".gitea" / "workflows" / "ci.yml"
LIST = ROOT / ".gitea" / "required-checks.txt"
API = "https://git.geisten.net/api/v1/repos/geisten/geistlib/branch_protections/main"
EVENT = "pull_request"


def scalar(value: str) -> str:
    value = value.split(" #", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]
    return value


def jobs(text: str) -> list[tuple[str, list[dict]]]:
    """(name template, matrix include entries) per job, in file order."""
    lines = text.splitlines()
    start = lines.index("jobs:")
    out: list[tuple[str, list[dict]]] = []
    i = start + 1
    while i < len(lines):
        if re.match(r"^\S", lines[i]):
            break
        m = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", lines[i])
        if not m:
            i += 1
            continue
        key, name, entries = m.group(1), m.group(1), []
        i += 1
        while i < len(lines) and not re.match(r"^  \S", lines[i]) and not re.match(r"^\S", lines[i]):
            line = lines[i]
            nm = re.match(r"^    name:\s*(.+)$", line)
            if nm:
                name = scalar(nm.group(1))
            inc = re.match(r"^( +)include:\s*$", line)
            if inc:
                base = len(inc.group(1))
                i += 1
                item_indent = None
                while i < len(lines):
                    l2 = lines[i]
                    if l2.strip() == "" or l2.lstrip().startswith("#"):
                        i += 1
                        continue
                    ind = len(l2) - len(l2.lstrip())
                    if ind <= base:
                        break
                    im = re.match(r"^( *)- ([A-Za-z0-9_-]+):\s*(.*)$", l2)
                    if im and (item_indent is None or len(im.group(1)) == item_indent):
                        item_indent = len(im.group(1))
                        entries.append({im.group(2): scalar(im.group(3))})
                    elif item_indent is not None and ind == item_indent + 2:
                        km = re.match(r"^ *([A-Za-z0-9_-]+):\s*(.*)$", l2)
                        if km and entries:
                            entries[-1][km.group(1)] = scalar(km.group(2))
                    i += 1
                continue
            i += 1
        out.append((name, entries))
        _ = key
    return out


def expected(text: str, workflow_name: str = "CI") -> list[str]:
    names = []
    for template, entries in jobs(text):
        fields = re.findall(r"\$\{\{\s*matrix\.([A-Za-z0-9_-]+)\s*\}\}", template)
        if not fields:
            names.append(template)
            continue
        if not entries:
            raise SystemExit(f"job name {template!r} uses the matrix, but no include entries were found")
        for entry in entries:
            name = template
            for field in fields:
                if field not in entry:
                    raise SystemExit(f"matrix entry {entry} has no {field!r} for {template!r}")
                name = re.sub(r"\$\{\{\s*matrix\." + field + r"\s*\}\}", entry[field], name)
            names.append(name)
    return sorted(f"{workflow_name} / {n} ({EVENT})" for n in names)


def listed() -> list[str]:
    return sorted(l.strip() for l in LIST.read_text().splitlines()
                  if l.strip() and not l.lstrip().startswith("#"))


def api(method: str, body: dict | None = None) -> dict:
    token = os.environ.get("GITEA_TOKEN")
    if not token:
        raise SystemExit("GITEA_TOKEN (repository admin) is not set")
    req = urllib.request.Request(API, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"token {token}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def diff(label_a: str, a: list[str], label_b: str, b: list[str]) -> int:
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    for n in only_a:
        print(f"  only in {label_a}: {n}")
    for n in only_b:
        print(f"  only in {label_b}: {n}")
    return 1 if only_a or only_b else 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    derived = expected(WORKFLOW.read_text())
    if cmd == "print":
        print("\n".join(derived))
        return 0
    if cmd == "check":
        rc = diff(WORKFLOW.relative_to(ROOT).as_posix(), derived, LIST.relative_to(ROOT).as_posix(), listed())
        print(f"required checks: {len(derived)} CI jobs, "
              + ("list current" if rc == 0 else
                 "list out of date: regenerate with `scripts/required_checks.py print`, "
                 "then `GITEA_TOKEN=... scripts/required_checks.py sync` after the merge"))
        return rc
    if cmd == "verify":
        live = sorted(api("GET").get("status_check_contexts") or [])
        rc = diff("branch protection", live, LIST.relative_to(ROOT).as_posix(), listed())
        print(f"branch protection: {len(live)} required, " + ("matches the file" if rc == 0 else "differs"))
        return rc
    if cmd == "sync":
        want = listed()
        if diff(WORKFLOW.relative_to(ROOT).as_posix(), derived, LIST.relative_to(ROOT).as_posix(), want):
            raise SystemExit("the file does not match ci.yml; fix it first (check)")
        got = api("PATCH", {"enable_status_check": True, "status_check_contexts": want})
        print(f"branch protection now requires {len(got['status_check_contexts'])} checks")
        return 0
    raise SystemExit(f"unknown command {cmd!r}: check | print | verify | sync")


if __name__ == "__main__":
    sys.exit(main())
