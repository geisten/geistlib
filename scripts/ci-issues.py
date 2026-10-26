#!/usr/bin/env python3
"""ci-issues.py — one Gitea issue per CI job that fails on main.

The post-merge tier of .gitea/workflows/ci.yml (sanitizers, real models,
perf floors, ...) gates no merge, so a failure there must not depend on
someone reading the run. The last job of a run on main (push, nightly,
dispatch) calls this script, which looks at every other job of the run:

    failure   open an issue "CI on main: <job name>" labelled ci-failure,
              or comment on the open one (one issue per job name, however
              many runs fail)
    success   comment and close that job's open issue
    other     nothing: skipped or cancelled says nothing about the code

Environment (set by the runner): GITEA_TOKEN (the job's token: it needs
issue write and actions read on this repository), GITHUB_SERVER_URL,
GITHUB_REPOSITORY, GITHUB_RUN_ID, GITHUB_SHA, GITHUB_EVENT_NAME, and SELF
(this job's name, left out of the scan).

    ci-issues.py            act
    ci-issues.py --dry-run  print what it would do, write nothing

Stdlib only (with scripts/gitea_api.py), like scripts/required_checks.py.
"""
from __future__ import annotations

import os
import sys
import urllib.parse

from gitea_api import Client

LABEL = "ci-failure"
PREFIX = "CI on main: "


def plan(jobs: list[dict], open_issues: dict[str, int], self_name: str) -> list[tuple[str, str, int | None]]:
    """(action, job name, issue number) per job: open | comment | close."""
    out = []
    for job in jobs:
        name = job["name"]
        if name == self_name:
            continue
        state = job.get("conclusion") or job.get("status")
        issue = open_issues.get(PREFIX + name)
        if state == "failure":
            out.append(("comment", name, issue) if issue else ("open", name, None))
        elif state == "success" and issue:
            out.append(("close", name, issue))
    return out


def main() -> int:
    dry = "--dry-run" in sys.argv[1:]
    env = os.environ
    server = env.get("GITHUB_SERVER_URL", "").rstrip("/")
    api = Client(env["GITEA_TOKEN"], f"{server}/api/v1/repos/{env['GITHUB_REPOSITORY']}")
    run_id, sha = env["GITHUB_RUN_ID"], env.get("GITHUB_SHA", "")[:12]
    event = env.get("GITHUB_EVENT_NAME", "")

    jobs = api.pages(f"/actions/runs/{run_id}/jobs", "jobs")
    labels = {l["name"]: l["id"] for l in api.pages("/labels")}
    if LABEL not in labels:
        raise SystemExit(f"ci-issues: label {LABEL!r} is missing in the repository")
    q = urllib.parse.urlencode({"state": "open", "type": "issues", "labels": LABEL})
    open_issues = {i["title"]: i["number"] for i in api.pages(f"/issues?{q}")}

    steps = plan(jobs, open_issues, env.get("SELF", ""))
    url = {j["name"]: j.get("html_url", "") for j in jobs}
    where = f"run {run_id} ({event}, {sha})"
    for action, name, issue in steps:
        print(f"ci-issues: {action} {name!r}" + (f" #{issue}" if issue else ""))
        if dry:
            continue
        if action == "open":
            body = (f"The CI job **{name}** failed on main: {url[name]}\n\n"
                    f"First seen in {where}. Main's branch protection does not run "
                    "this job before a merge (post-merge tier, docs/CI_COVERAGE.md); "
                    "this issue is updated by every later run on main and closed "
                    "by the first green one (scripts/ci-issues.py).")
            got = api.call("POST", "/issues", {"title": PREFIX + name, "body": body,
                                               "labels": [labels[LABEL]]})
            print(f"ci-issues:   opened #{got['number']}")
        elif action == "comment":
            api.call("POST", f"/issues/{issue}/comments",
                     {"body": f"Still failing in {where}: {url[name]}"})
        else:
            api.call("POST", f"/issues/{issue}/comments",
                     {"body": f"Green again in {where}: {url[name]}"})
            api.call("PATCH", f"/issues/{issue}", {"state": "closed"})
    if not steps:
        print(f"ci-issues: nothing to report for {len(jobs)} jobs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
