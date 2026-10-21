#!/usr/bin/env python3
"""tests/test_ci_bot_py.py — the decisions of scripts/ci-bot.py.

Feeds the bot recorded Gitea API responses (tests/data/ci_bot/, trimmed
from runs 1516, 1531, 1538, 1539 and PR #122 on 2026-10-20) and checks:
the infra/real job classifier, the rerun decision (infra only, at most
RERUN_MAX, never with a real failure in the run), who may be force-pushed,
"Depends on #N" / auto-merge parsing, the CHANGELOG union resolver, and a
trial rebase in throwaway git repositories (CHANGELOG-only conflict
resolved, any other conflict refused, nothing pushed). Hermetic: no
network; needs only git.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "data" / "ci_bot"
spec = importlib.util.spec_from_file_location("ci_bot", ROOT / "scripts" / "ci-bot.py")
bot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bot)

fails = 0


def expect(cond: bool, label: str) -> None:
    global fails
    print(f"  {'✓' if cond else '✗'} {label}")
    if not cond:
        fails += 1


JOBS = json.loads((DATA / "failed_jobs.json").read_text())
PR = json.loads((DATA / "pr122.json").read_text())


def kinds(run: str, logs_found=None) -> dict[int, str]:
    return {j["id"]: bot.classify_job(j, logs_found)[0] for j in JOBS[run]}


print("classifier, recorded runs")
expect(set(kinds("1531").values()) == {"infra"}, "run 1531: lost tasks (every step failed, end :x5:25) are infra")
expect(set(kinds("1538").values()) == {"infra"}, "run 1538: act-cache lock, no step ran -> infra")
k = kinds("1539")
expect(set(k.values()) == {"infra"}, "run 1539: a job whose only run step was an always() upload is infra")
k = kinds("1516")
expect(k[12868] == "infra", "run 1516: job 12868 (lost task) infra")
expect(k[12880] == "logs?", "run 1516: coverage gate failure asks for the log")
expect(bot.classify_job(next(j for j in JOBS["1516"] if j["id"] == 12880), True)[0] == "real",
       "run 1516: coverage gate failure with a log is real")

print("classifier, synthetic jobs")


def job(steps, start="2026-10-20T10:00:00Z", end="2026-10-20T10:00:40Z"):
    return {"id": 1, "name": "j", "started_at": start, "completed_at": end, "conclusion": "failure",
            "steps": [{"number": i, "name": n, "conclusion": c} for i, (n, c) in enumerate(steps)]}


checkout_fail = [("Run actions/checkout@x", "failure"), ("Build", "skipped"), ("Test", "skipped")]
expect(bot.classify_job(job(checkout_fail))[0] == "infra", "checkout fails after 40 s -> infra")
slow = job(checkout_fail, end="2026-10-20T10:20:03Z")
expect(bot.classify_job(slow)[0] == "logs?", "checkout fails after 20 min, no sweep -> ask for the log")
expect(bot.classify_job(slow, True)[0] == "real", "  ... with a log -> real")
expect(bot.classify_job(slow, False)[0] == "infra", "  ... logs not found -> infra")
expect(bot.classify_job(job(checkout_fail, end="2026-10-20T10:25:25Z"))[0] == "infra",
       "checkout fails at a zombie sweep (:x5:25) -> infra")
ccache = [("Run actions/checkout@x", "success"), ("ccache", "failure"), ("Build", "skipped")]
expect(bot.classify_job(job(ccache))[0] == "infra", "ccache before the build fails quickly -> infra")
build = [("Run actions/checkout@x", "success"), ("Build", "failure"), ("Test", "skipped")]
expect(bot.classify_job(job(build), True)[0] == "real", "build step fails -> real")
after_build = [("Run actions/checkout@x", "success"), ("Build", "failure"), ("ccache save", "failure")]
expect(bot.classify_job(job(after_build), True)[0] == "real", "a ccache step after a failed build is not setup")
expect(bot.is_zombie_sweep("2026-10-20T09:35:25Z") and not bot.is_zombie_sweep("2026-10-20T09:36:25Z"),
       "zombie sweep pattern hh:x5:25 / not hh:x6:25")

print("statuses -> runs")
expect(bot.run_ids(PR["statuses"]) == [1534],
       "PR #122: one failed status (run 1534, still running: ci() waits for 'completed')")
expect(bot.run_ids([{"status": "success", "target_url": "/o/r/actions/runs/9/jobs/1"}]) == [],
       "successful statuses are ignored")


class FakeApi(bot.Gitea):
    """Serves fixtures; records writes instead of sending them."""

    def __init__(self, routes, dry=False):
        super().__init__("t", dry)
        self.routes, self.writes = routes, []

    def call(self, method, path, body=None, raw=False, tries=4):
        if method != "GET":
            self.writes.append((method, path, body))
            return None
        for prefix, val in self.routes.items():
            if path.startswith(prefix):
                if isinstance(val, Exception):
                    raise val
                return val
        raise bot.ApiError(404, path)


def ci_bot(run_id, jobs, attempt=1, dry=False, logs=None):
    sha = PR["pr"]["head"]["sha"]
    statuses = [{"status": "failure", "target_url": f"/geisten/geistlib/actions/runs/{run_id}/jobs/1"}]
    routes = {f"/repos/geisten/geistlib/commits/{sha}/status": {"statuses": statuses},
              f"/repos/geisten/geistlib/actions/runs/{run_id}/jobs": {"jobs": jobs},
              f"/repos/geisten/geistlib/actions/runs/{run_id}": {"status": "completed", "head_sha": sha,
                                                                 "run_attempt": attempt},
              "/repos/geisten/geistlib/actions/jobs/": logs if logs is not None else b"log"}
    tmp = tempfile.mkdtemp()
    b = bot.Bot(FakeApi(routes, dry), tmp, os.path.join(tmp, "s.json"), "t")
    return b


print("rerun decisions")
b = ci_bot(1531, JOBS["1531"])
b.ci("geisten/geistlib", "geisten/geistlib#122", PR["pr"])
expect([w[1] for w in b.api.writes] == ["/repos/geisten/geistlib/actions/runs/1531/rerun-failed-jobs"],
       "infra-only run 1531 -> one rerun-failed-jobs")
b.ci("geisten/geistlib", "geisten/geistlib#122", PR["pr"])
expect(len(b.api.writes) == 1, "same attempt polled again -> no second rerun")
for attempt in (2, 3, 4):
    b.api.routes[f"/repos/geisten/geistlib/actions/runs/1531"]["run_attempt"] = attempt
    b.ci("geisten/geistlib", "geisten/geistlib#122", PR["pr"])
expect(len(b.api.writes) == bot.RERUN_MAX, f"failing again and again -> at most {bot.RERUN_MAX} reruns")
b = ci_bot(1516, JOBS["1516"])
b.ci("geisten/geistlib", "geisten/geistlib#122", PR["pr"])
expect(b.api.writes == [], "run 1516 with a real coverage failure -> no rerun")
b = ci_bot(1516, [j for j in JOBS["1516"] if j["id"] == 12880], logs=bot.ApiError(404, "logs not found"))
b.ci("geisten/geistlib", "geisten/geistlib#122", PR["pr"])
expect(len(b.api.writes) == 1, "a long job whose logs are gone -> rerun")
b = ci_bot(1538, JOBS["1538"], dry=True)
b.ci("geisten/geistlib", "geisten/geistlib#122", PR["pr"])
expect(b.api.writes == [], "dry-run -> no write at all")

print("state survives a flaky listing")
b = ci_bot(1531, JOBS["1531"])
b.state["reruns"]["geisten/geist-runtime#9:run1:abc"] = 2
b.api.routes["/repos/geisten/geistlib/pulls?"] = [PR["pr"]]
b.api.routes["/repos/geisten/geist-runtime/pulls?"] = bot.ApiError(0, "timed out")
b.tick()
expect(b.state["reruns"].get("geisten/geist-runtime#9:run1:abc") == 2, "listing timed out -> state kept")
b.api.routes["/repos/geisten/geist-runtime/pulls?"] = []
b.tick()
expect("geisten/geist-runtime#9:run1:abc" not in b.state["reruns"], "PR gone from a good listing -> forgotten")
expect(json.loads(Path(b.state_path).read_text())["reruns"], "state written to disk")

print("who may be force-pushed")
pr, commits = PR["pr"], PR["commits"]
expect(bot.may_rewrite(pr, commits, "geisten/geistlib")[0],
       "PR #122: every non-merge commit carries the Claude trailer (merge commits ignored)")
human = copy.deepcopy(commits)
human[3]["commit"]["message"] = "a human's fix"
expect(not bot.may_rewrite(pr, human, "geisten/geistlib")[0], "one human commit -> no")
labelled = dict(pr, labels=[{"name": bot.LABEL_REBASE}])
expect(bot.may_rewrite(labelled, human, "geisten/geistlib")[0], f"... unless labelled {bot.LABEL_REBASE}")
fork = copy.deepcopy(labelled)
fork["head"]["repo"]["full_name"] = "someone/geistlib"
expect(not bot.may_rewrite(fork, commits, "geisten/geistlib")[0], "a fork's branch -> never")
expect(not bot.may_rewrite(pr, commits, "geisten/geist-runtime")[0], "head repo must be the PR's repo")

print("dependencies, auto-merge")
expect(bot.depends_on({"number": 5, "body": "Depends on #3.\nalso depends on #4", "labels": []}) == [3, 4],
       "'Depends on #N' in the body, case-insensitive")
expect(bot.depends_on({"number": 5, "body": "", "labels": [{"name": "depends-on-#7"}]}) == [7],
       "label depends-on-#N")
expect(bot.depends_on({"number": 5, "body": "Fixes #3", "labels": []}) == [], "no dependency")
expect(bot.wants_automerge({"body": "Enable auto-merge once #3 is in.", "labels": []}), "body asks for auto-merge")
expect(bot.wants_automerge({"body": "", "labels": [{"name": "automerge"}]}), "label automerge")
expect(not bot.wants_automerge({"body": "Please do not auto-merge.", "labels": []}), "'do not auto-merge'")
expect(not bot.wants_automerge({"body": "Depends on #3", "labels": []}), "silent PR -> no")

print("CHANGELOG union")
conf = "# C\n## [Unreleased]\n<<<<<<< HEAD\n- main entry\n=======\n- pr entry\n>>>>>>> abc (pr)\n\n- old\n"
expect(bot.resolve_union(conf) == "# C\n## [Unreleased]\n- main entry\n- pr entry\n\n- old\n",
       "both sides kept, main's first")
diff3 = "<<<<<<< HEAD\n- a\n- shared\n||||||| base\n- shared\n=======\n- shared\n- b\n>>>>>>> x\n"
expect(bot.resolve_union(diff3) == "- a\n- shared\n- b\n", "diff3 base dropped, duplicate line once")
expect(bot.resolve_union("<<<<<<< HEAD\n- a\n=======\n- b\n") is None, "unbalanced markers -> None")


print("trial rebase in throwaway repositories")


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def commit(cwd, files, msg):
    for name, text in files.items():
        Path(cwd, name).write_text(text)
    git(cwd, "add", *files)
    git(cwd, "commit", "-q", "-m", msg)
    return git(cwd, "rev-parse", "HEAD")


with tempfile.TemporaryDirectory() as tmp:
    r = os.path.join(tmp, "r")
    os.makedirs(r)
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.name", "t")
    git(r, "config", "user.email", "t@t")
    log = "# Changelog\n\n## [Unreleased]\n\n- base\n"
    base = commit(r, {"CHANGELOG.md": log, "a.c": "int a;\n", "b.c": "int b;\n"}, "base")
    git(r, "checkout", "-q", "-b", "pr")
    pr_head = commit(r, {"CHANGELOG.md": log.replace("- base", "- pr\n- base"), "a.c": "int a2;\n"}, "pr")
    git(r, "checkout", "-q", "main")
    main_cl = commit(r, {"CHANGELOG.md": log.replace("- base", "- main\n- base")}, "main changelog")
    g = bot.Git(tmp, "geisten/x", "t")
    g.dir = r
    how, files, new = g.trial_rebase(pr_head, main_cl)
    text = git(r, "show", f"{new}:CHANGELOG.md") if new else ""
    expect(how == "changelog" and "- main\n- pr\n- base" in text and git(r, "show", f"{new}:a.c") == "int a2;",
           "CHANGELOG-only conflict -> rebased, both entries kept")
    expect(git(r, "rev-parse", "pr") == pr_head, "the branch itself is untouched (push is separate)")
    main_code = commit(r, {"a.c": "int a3;\n"}, "main code")
    how, files, new = g.trial_rebase(pr_head, main_code)
    expect(how == "conflict" and "a.c" in files and new is None,
           f"conflict in a.c -> refused ({how}, {files})")
    expect(not git(r, "worktree", "list").count("\n"), "scratch worktree removed")
    how, _, _ = g.trial_rebase(pr_head, base)
    expect(how == "uptodate", "PR already on its base -> nothing to do")
    git(r, "checkout", "-q", "-b", "pr2", base)
    pr2 = commit(r, {"b.c": "int b2;\n"}, "pr2")
    how, files, new = g.trial_rebase(pr2, main_code)
    expect(how == "clean" and new and git(r, "merge-base", "--is-ancestor", main_code, new) == "",
           "no conflict -> clean rebase onto the new base")

if fails:
    print(f"test_ci_bot_py: {fails} FAILED")
    sys.exit(1)
print("test_ci_bot_py: PASS")
