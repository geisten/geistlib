#!/usr/bin/env python3
"""required_checks.py — keep main's required status checks in step with CI.

main's branch protection on git.geisten.net requires every job of CI's
required tier by its exact status name (a glob such as "CI / *" matched no
status while a new commit's CI had not started, and Gitea merged on it).
The required tier is every job of .gitea/workflows/ci.yml whose job-level
`if:` does not test needs.changes.outputs.full; the jobs that do are the
post-merge tier (docs/CI_COVERAGE.md "Tiers"), which a pull request skips.
An exact list must follow the workflow, so it lives in
.gitea/required-checks.txt and:

    check    derive the required job names from .gitea/workflows/ci.yml
             (matrix include entries expanded) and compare them with the
             file; also check that the ci-issues job waits for every other
             job. Exit 1 on any difference. Runs in CI, needs no token.
    print    print the derived list (to regenerate the file).
    post-merge  print the post-merge tier's names (for the docs).
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


POST_MERGE = "needs.changes.outputs.full"
REPORT_JOB = "ci-issues"


def jobs(text: str) -> list[tuple[str, list[dict]]]:
    """(name template, matrix include entries) per job, in file order."""
    return [(j["name"], j["entries"]) for j in parse(text)]


def parse(text: str) -> list[dict]:
    """Per job, in file order: key, name (template), entries (matrix include),
    post (its `if:` tests POST_MERGE), needs (job keys)."""
    lines = text.splitlines()
    start = lines.index("jobs:")
    out: list[dict] = []
    i = start + 1
    while i < len(lines):
        if re.match(r"^\S", lines[i]):
            break
        m = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", lines[i])
        if not m:
            i += 1
            continue
        key, name, entries, post, needs = m.group(1), m.group(1), [], False, []
        i += 1
        while i < len(lines) and not re.match(r"^  \S", lines[i]) and not re.match(r"^\S", lines[i]):
            line = lines[i]
            nm = re.match(r"^    name:\s*(.+)$", line)
            if nm:
                name = scalar(nm.group(1))
            if re.match(r"^    if:", line):
                post = POST_MERGE in line
            nd = re.match(r"^    needs:\s*(.*)$", line)
            if nd:
                inline = nd.group(1).strip()
                if inline.startswith("["):
                    needs = [scalar(x) for x in inline.strip("[]").split(",") if x.strip()]
                elif inline:
                    needs = [scalar(inline)]
                else:
                    while i + 1 < len(lines) and re.match(r"^      - ", lines[i + 1]):
                        i += 1
                        needs.append(scalar(lines[i].strip()[2:]))
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
        out.append({"key": key, "name": name, "entries": entries, "post": post, "needs": needs})
    return out


def expected(text: str, workflow_name: str = "CI", post_merge: bool = False) -> list[str]:
    """Status names of the required tier (or, post_merge=True, of the other)."""
    names = []
    for job in parse(text):
        if job["post"] != post_merge:
            continue
        template, entries = job["name"], job["entries"]
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


def report_gaps(text: str) -> list[str]:
    """Jobs the ci-issues job does not wait for (it would miss their failures)."""
    js = parse(text)
    report = [j for j in js if j["key"] == REPORT_JOB]
    if not report:
        return [f"(no {REPORT_JOB} job)"]
    return [j["key"] for j in js if j["key"] != REPORT_JOB and j["key"] not in report[0]["needs"]]


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
    if cmd == "post-merge":
        print("\n".join(expected(WORKFLOW.read_text(), post_merge=True)))
        return 0
    if cmd == "check":
        rc = diff(WORKFLOW.relative_to(ROOT).as_posix(), derived, LIST.relative_to(ROOT).as_posix(), listed())
        missing = report_gaps(WORKFLOW.read_text())
        if missing:
            print(f"  {REPORT_JOB} does not wait for: {', '.join(missing)}")
            rc = 1
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
