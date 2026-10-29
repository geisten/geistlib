#!/usr/bin/env python3
"""agent_gate.py — the pipeline-enforced rules of the agent pipeline
(.gitea/workflows/agent.yml, AGENT.md §9). The prompts ask the model to
behave; these checks hold whatever the model does. Stdlib only.

Model output is data. The jobs that hold AGENT_BOT_TOKEN run this script
from a trusted checkout (main, or the PR's base) and never execute anything
the model wrote:

  outdir DIR                 the sandbox's output: regular files only, no
                             symlinks, patches named NNNN-*.patch, size cap
  paths BASE HEAD            the commits BASE..HEAD touch no protected path
                             (workflows, actions, scripts, build system, git
                             and agent configuration) and add no symlink or
                             submodule
  pr-body IN ISSUE HASH      the PR description the bot posts: closing
                             keywords, auto-merge requests, depends-on and
                             @mentions neutralised; "Closes #ISSUE" and the
                             issue pin appended by the pipeline itself
  verdict FILE               PASS or FAIL from the reviewer's last
                             non-empty line (anything else: FAIL)
  issue-hash JSON [KEY]      sha256 of an issue's title and body (KEY: the
                             object inside JSON holding it, e.g. "issue")
  pin FILE                   "<issue> <hash>" from a PR body's pin, or nothing
  ci-hold                    ci.yml's `changes` job: a pull request from an
                             agent/* branch runs CI only after an approver
                             commented "/ci-ok <sha>" (the whole comment) for
                             its head commit; see HOLD_STEP
  claim-work | claim-review | claim-fix   (agent.yml, bot token)
                             decide from the live labels and comments whether
                             this run should act; writes GITHUB_OUTPUT
  rerun-held                 after a "/ci-ok SHA" comment: rerun the held CI run
  text FILE                  model text for a bot comment: capped, no @mentions

Environment for the API commands: GITEA_TOKEN, GITHUB_SERVER_URL,
GITHUB_REPOSITORY, GITHUB_EVENT_PATH, GITHUB_OUTPUT.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gitea_api import ApiError, Client  # noqa: E402

BOT = "geist-agent"
# Who may approve CI on an agent head and start agent work. ci-bot posts as
# germar too; its comments never consist of a bare "/ci-ok <sha>".
APPROVERS = frozenset({"germar"})
# Who may move the status labels (the bot moves them from trusted steps).
STEERS = APPROVERS | {BOT}
AGENT_REF = re.compile(r"^agent/issue-[0-9]+$")
ROUNDS = 3
# Paths a model-written patch may not touch: what CI or the pipeline runs or
# trusts before a human has looked (workflows, local actions, scripts, the
# build system), git's own configuration, and the agents' instructions and
# settings (a .claude/settings.json hook would run in the next agent job).
DENY = re.compile(r"""^(?:
      \.gitea/ | \.github/ | scripts/ | mk/ | Makefile$ | GNUmakefile$
    | (?:.*/)?\.git[^/]*$          # .gitattributes, .gitmodules, .gitignore at any depth
    | (?:.*/)?\.claude/ | (?:.*/)?CLAUDE(?:\.local)?\.md$ | (?:.*/)?AGENTS?\.md$
    | \.mcp\.json$ | \.clang-format$ | CHANGELOG\.md$
    | (?:.*/)?[^/]*\.mk$ | build/ | bin/   # make includes (common.mk: -include $(DEPS))
)""", re.X)
MAX_OUT_BYTES = 4 << 20
PATCH_NAME = re.compile(r"^[0-9]{4}-[A-Za-z0-9._-]{1,80}\.patch$")
CI_OK = re.compile(r"\A\s*/ci-ok\s+([0-9a-f]{7,40})\s*\Z")
# The step in ci.yml's `changes` job that holds an agent head; ci-bot leaves
# a run alone whose only failure is this step (scripts/ci-bot.py).
HOLD_STEP = "Hold an agent head until a human approves it"
REVIEW_HEAD = "**Agent review**"
FIX_HEAD = "**Agent:**"
CI_FAILED_HEAD = "**CI failed**"


# -- pure rules (tests/test_agent_gate_py.py)

def denied(paths: list[str]) -> list[str]:
    return [p for p in paths if DENY.match(p)]


def outdir_problems(root: Path) -> list[str]:
    bad, total = [], 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            p = Path(dirpath, name)
            rel = p.relative_to(root).as_posix()
            st = p.lstat()
            if stat.S_ISLNK(st.st_mode):
                bad.append(f"{rel}: a symlink")
            elif stat.S_ISDIR(st.st_mode):
                continue
            elif not stat.S_ISREG(st.st_mode):
                bad.append(f"{rel}: not a regular file")
            else:
                total += st.st_size
                if rel.startswith("patches/") and not PATCH_NAME.match(name):
                    bad.append(f"{rel}: not a NNNN-*.patch name")
    if total > MAX_OUT_BYTES:
        bad.append(f"{total} bytes in all (cap {MAX_OUT_BYTES})")
    return bad


CLOSING = re.compile(r"\b(close[sd]?|fix(?:e[sd])?|resolve[sd]?)(\s*:?\s+)(?=[\w./-]*#[0-9])", re.I)
AUTOMERGE = re.compile(r"auto-?merge", re.I)
DEPENDS = re.compile(r"\bdepends(\s+on\s+#)", re.I)
MENTION = re.compile(r"(?<![\w`])@(?=[A-Za-z0-9])")
PIN = re.compile(r"<!-- agent-issue: ([0-9]+) sha256:([0-9a-f]{64}) -->")
LINKED = re.compile(r"\b(close[sd]?|fix(?:e[sd])?|resolve[sd]?) #([0-9]+)", re.I)


def sanitize_body(text: str, issue: int, issue_hash: str) -> str:
    """The PR body the bot posts: the model wrote `text`; the pipeline owns
    which issue it closes, whether it asks for auto-merge (ci-bot) or a
    dependency, and whom it pings."""
    text = text[:60000]
    text = PIN.sub("", text)
    text = CLOSING.sub(lambda m: "refs ", text)
    text = AUTOMERGE.sub("auto merge", text)
    text = DEPENDS.sub(r"follows\1", text)
    text = MENTION.sub("@​", text)
    return f"{text.rstrip()}\n\nCloses #{issue}\n<!-- agent-issue: {issue} sha256:{issue_hash} -->\n"


def comment_text(text: str) -> str:
    """Model text the bot posts as a comment: capped, no @mentions."""
    return MENTION.sub("@\u200b", text[:60000])


def pin(body: str) -> tuple[int, str] | None:
    m = PIN.search(body or "")
    return (int(m.group(1)), m.group(2)) if m else None


def verdict(text: str) -> str:
    lines = [l for l in text.splitlines() if l.strip()]
    last = re.sub(r"[*_`#>\s]", "", lines[-1]) if lines else ""
    return "PASS" if last == "VERDICT:PASS" else "FAIL"


def issue_hash(title: str, body: str) -> str:
    return hashlib.sha256(f"{title}\n\n{body or ''}".encode()).hexdigest()


def ci_ok_approved(comments: list[dict], sha: str, approvers=APPROVERS) -> bool:
    for c in comments:
        m = CI_OK.match(c.get("body") or "")
        if m and (c.get("user") or {}).get("login") in approvers and sha.startswith(m.group(1)):
            return True
    return False


def by_trusted(c: dict) -> bool:
    return (c.get("user") or {}).get("login") in STEERS


def fix_rounds(comments: list[dict]) -> int:
    return sum(1 for c in comments if by_trusted(c) and (c.get("body") or "").startswith(FIX_HEAD))


def work_order(comments: list[dict]) -> str | None:
    """The newest review or CI failure, if no fix answered it yet."""
    for c in reversed(comments):
        body = c.get("body") or ""
        if not by_trusted(c):
            continue
        if body.startswith(FIX_HEAD):
            return None
        if body.startswith((REVIEW_HEAD, CI_FAILED_HEAD)):
            return body
    return None


def reviewed(comments: list[dict], sha: str) -> bool:
    head = f"{REVIEW_HEAD} of {sha[:8]}"
    return any(by_trusted(c) and (c.get("body") or "").startswith(head) for c in comments)


def labels_of(obj: dict) -> list[str]:
    return [l.get("name", "") for l in obj.get("labels") or []]


# -- I/O

def api() -> tuple[Client, str]:
    base = os.environ["GITHUB_SERVER_URL"].rstrip("/") + "/api/v1"
    return Client(os.environ["GITEA_TOKEN"], base), "/repos/" + os.environ["GITHUB_REPOSITORY"]


def event() -> dict:
    return json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())


def output(**kv) -> None:
    lines = "".join(f"{k}={v}\n" for k, v in kv.items())
    sys.stdout.write(lines)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(lines)


def same_repo(pr: dict) -> bool:
    return ((pr.get("head") or {}).get("repo") or {}).get("full_name") == \
           ((pr.get("base") or {}).get("repo") or {}).get("full_name")


def cmd_paths(base: str, head: str) -> int:
    git = ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false"]
    names = subprocess.run(git + ["diff", "--name-only", "--no-renames", f"{base}..{head}"],
                           check=True, capture_output=True, text=True).stdout.split("\n")
    raw = subprocess.run(git + ["diff", "--raw", "--no-renames", f"{base}..{head}"],
                         check=True, capture_output=True, text=True).stdout
    bad = [f"{p}: protected path" for p in denied([n for n in names if n])]
    for line in raw.splitlines():   # ":old new oldsha newsha status\tpath"
        meta, _, path = line.partition("\t")
        new_mode = meta.split()[1] if meta.startswith(":") else ""
        if new_mode in ("120000", "160000"):
            bad.append(f"{path}: {'a symlink' if new_mode == '120000' else 'a submodule'}")
    for b in bad:
        print(f"::error::{b}")
    return 1 if bad else 0


def cmd_ci_hold() -> int:
    ev = event()
    pr = ev.get("pull_request")
    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request" or not pr:
        print("not a pull request: nothing to hold")
        return 0
    ref, sha = pr["head"]["ref"], pr["head"]["sha"]
    agent = ref.startswith("agent/") or (pr.get("user") or {}).get("login") == BOT
    if os.environ.get("GITHUB_SERVER_URL", "").rstrip("/") == "https://github.com":
        print("GitHub mirror: no agent heads there")
        return 0
    try:
        c, repo = api()
        comments = c.pages(f"{repo}/issues/{pr['number']}/comments")
    except (ApiError, KeyError) as e:
        if not agent:
            print(f"::warning::comments unreadable ({e}); not an agent head, so no hold")
            return 0
        print(f"::error::cannot read the comments ({e}): an agent head stays held")
        return 1
    if not agent:
        print(f"{ref} is not an agent head: no hold ({len(comments)} comments readable)")
        return 0
    if ci_ok_approved(comments, sha):
        print(f"{sha[:12]} approved by a human: CI runs")
        return 0
    print(f"::error::{ref} at {sha[:12]} is model-written. CI runs it on runners "
          f"whose job containers reach the docker socket (and on the Mac host). "
          f"Read the change, then comment exactly '/ci-ok {sha[:12]}' on the PR; "
          f"the agent workflow reruns this run.")
    return 1


def cmd_claim(kind: str) -> int:
    ev = event()
    c, repo = api()
    sender = (ev.get("sender") or {}).get("login")

    def no(why: str) -> int:
        print(f"skip: {why}")
        output(go="false")
        return 0

    # Work and /agent spend the subscription and start model-written code:
    # an approver only. Status labels: an approver or the bot's own steps.
    if sender not in (APPROVERS if kind == "work" or ev.get("comment") else STEERS):
        return no(f"sender {sender} may not start {kind}")
    if kind == "work":
        n = ev["issue"]["number"]
        if ev["issue"].get("pull_request"):
            return no("a pull request, not an issue")
        issue = c.call("GET", f"{repo}/issues/{n}")
        labels = labels_of(issue)
        if "agent" not in labels or any(l.startswith("status:") for l in labels):
            return no(f"labels {labels}: not a fresh `agent` issue")
        # Claim: drop `agent` before anything else, so a later label event
        # (Gitea fires `labeled` for any label) finds nothing to start.
        keep = [l for l in labels if l != "agent"]
        c.call("PUT", f"{repo}/issues/{n}/labels", {"labels": keep})
        branch = f"agent/issue-{n}"
        try:
            lease = c.call("GET", f"{repo}/branches/{branch}")["commit"]["id"]
        except ApiError as e:
            if e.code != 404:
                raise
            lease = ""
        # The text the work run sees is the event's; pin that one.
        h = issue_hash(ev["issue"]["title"], ev["issue"].get("body") or "")
        output(go="true", branch=branch, lease=lease, issue_hash=h)
        return 0
    n = ev["pull_request"]["number"] if "pull_request" in ev and ev.get("pull_request") else ev["issue"]["number"]
    pr = c.call("GET", f"{repo}/pulls/{n}")
    if pr.get("state") != "open":
        return no("the PR is not open")
    if not same_repo(pr):
        return no("the head branch lives in a fork")
    labels, sha, ref = labels_of(pr), pr["head"]["sha"], pr["head"]["ref"]
    comments = c.pages(f"{repo}/issues/{n}/comments")
    if kind == "review":
        if "status:validate" not in labels:
            return no(f"labels {labels}: not in status:validate")
        if reviewed(comments, sha):
            return no(f"{sha[:8]} has a review already")
        tmp = Path(os.environ.get("RUNNER_TEMP", "."))
        body = pr.get("body") or ""
        tmp.joinpath("pr.md").write_text(f"# {pr['title']}\n\n{body}")
        pinned = pin(body)
        m = LINKED.search(PIN.sub("", body))
        issue = pinned[0] if pinned else int(m.group(2)) if m else \
            int(ref.rsplit("-", 1)[1]) if AGENT_REF.match(ref) else None
        if issue is None:
            tmp.joinpath("issue.md").write_text("(no linked issue: review against the PR description)\n")
        else:
            it = c.call("GET", f"{repo}/issues/{issue}")
            if pinned and issue_hash(it["title"], it.get("body") or "") != pinned[1]:
                c.call("POST", f"{repo}/issues/{n}/comments", {"body": (
                    f"{REVIEW_HEAD} of {sha[:8]}: not reviewed. Issue #{issue} changed after the "
                    "agent worked on it, so the requirements are no longer the ones the change "
                    "was written against. status:stuck: a human decides (relabel the issue "
                    "`agent` for a fresh run, or edit the pin in the PR body).")})
                set_status(c, repo, n, "stuck")
                return no("the issue changed since the work run")
            tmp.joinpath("issue.md").write_text(f"# #{issue}: {it['title']}\n\n{it.get('body') or ''}\n")
        output(go="true", sha=sha, ref=ref, base=pr["base"]["sha"], issue=issue or "")
        return 0
    # fix
    if not AGENT_REF.match(ref):
        return no(f"head {ref} is not an agent/issue-<n> branch")
    if ev.get("comment"):
        body = ev["comment"].get("body") or ""
        if not body.startswith("/agent"):
            return no("not an /agent comment")
        feedback = body
    else:
        if "status:fix" not in labels:
            return no(f"labels {labels}: not in status:fix")
        rounds = fix_rounds(comments)
        if rounds >= ROUNDS:
            c.call("POST", f"{repo}/issues/{n}/comments", {"body": (
                f"{FIX_HEAD} {rounds} fix rounds without a PASS and a green CI; "
                "status:stuck, a human's turn.")})
            set_status(c, repo, n, "stuck")
            return no("round limit")
        feedback = work_order(comments)
        if feedback is None:
            return no("the newest review or CI failure has been answered")
    Path(os.environ.get("RUNNER_TEMP", "."), "feedback.md").write_text(feedback)
    Path(os.environ.get("RUNNER_TEMP", "."), "pr.md").write_text(f"# {pr['title']}\n\n{pr.get('body') or ''}")
    output(go="true", sha=sha, ref=ref)
    return 0


def set_status(c: Client, repo: str, n: int, status: str) -> None:
    labels = labels_of(c.call("GET", f"{repo}/issues/{n}"))
    keep = [l for l in labels if l != "agent" and not l.startswith("status:")]
    c.call("PUT", f"{repo}/issues/{n}/labels", {"labels": keep + [f"status:{status}"]})


def cmd_rerun_held() -> int:
    ev = event()
    m = CI_OK.match((ev.get("comment") or {}).get("body") or "")
    if (ev.get("sender") or {}).get("login") not in APPROVERS or not m:
        print("skip: not a bare '/ci-ok <sha>' from an approver")
        return 0
    n, prefix = ev["issue"]["number"], m.group(1)
    c, repo = api()
    pr = c.call("GET", f"{repo}/pulls/{n}")
    sha = pr["head"]["sha"]
    if not sha.startswith(prefix):
        c.call("POST", f"{repo}/issues/{n}/comments", {"body": (
            f"/ci-ok {prefix} does not name the head {sha[:12]}: nothing rerun.")})
        return 0
    st = c.call("GET", f"{repo}/commits/{sha}/status?limit=100") or {}
    runs = sorted({int(m.group(1)) for s in st.get("statuses") or []
                   if s.get("status") in ("failure", "error")
                   for m in [re.search(r"/actions/runs/(\d+)", s.get("target_url") or "")] if m})
    for run in runs:
        c.call("POST", f"{repo}/actions/runs/{run}/rerun-failed-jobs")
        print(f"rerun-failed-jobs: run {run}")
    if not runs:
        print("no failed run on the head yet; its CI will see the approval")
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    cmd, args = argv[0], argv[1:]
    if cmd == "outdir":
        bad = outdir_problems(Path(args[0]))
        for b in bad:
            print(f"::error::sandbox output: {b}")
        return 1 if bad else 0
    if cmd == "paths":
        return cmd_paths(args[0], args[1])
    if cmd == "pr-body":
        sys.stdout.write(sanitize_body(Path(args[0]).read_text(errors="replace"), int(args[1]), args[2]))
        return 0
    if cmd == "verdict":
        print(verdict(Path(args[0]).read_text(errors="replace")))
        return 0
    if cmd == "issue-hash":
        obj = json.loads(Path(args[0]).read_text())
        if len(args) > 1:
            obj = obj[args[1]]
        print(issue_hash(obj["title"], obj.get("body") or ""))
        return 0
    if cmd == "pin":
        p = pin(Path(args[0]).read_text(errors="replace"))
        if p:
            print(f"{p[0]} {p[1]}")
        return 0
    if cmd == "ci-hold":
        return cmd_ci_hold()
    if cmd in ("claim-work", "claim-review", "claim-fix"):
        return cmd_claim(cmd.split("-", 1)[1])
    if cmd == "rerun-held":
        return cmd_rerun_held()
    if cmd == "text":
        sys.stdout.write(comment_text(Path(args[0]).read_text(errors="replace")))
        return 0
    print(f"unknown command {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
