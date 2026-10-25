#!/usr/bin/env python3
"""ci-bot — keeps the open pull requests of geisten/geistlib and
geisten/geist-runtime moving (docs/HARDWARE_CI.md, "CI bot").

Polls Gitea every INTERVAL seconds and, within fixed rules, does three
things; everything else it leaves to a human:

1. Reruns infrastructure failures. A completed run of the PR's head whose
   failed jobs all died of infrastructure (job never got past "Set up job",
   every step marked failed = task lost by the runner, a setup step —
   checkout, ccache — failing within 5 min or at a zombie sweep, or "logs
   not found") gets POST .../rerun-failed-jobs, at most RERUN_MAX times per
   run and head SHA. A run with any real failure is left alone: a rerun
   would repeat the real failure along with the infra ones.
2. Rebases a PR Gitea reports as not mergeable when the only conflicting
   file is CHANGELOG.md (both sides' lines kept, main's first). It pushes
   (--force-with-lease on the head SHA it saw) only to a branch in the
   same repository whose non-merge commits are all Claude's or a bot's, or
   that carries the label `ci-bot:rebase`. Conflicts beyond CHANGELOG.md
   (or a human branch without the label) get one comment, never a repeat.
3. Dependent merges. A PR whose body says "Depends on #N" (or labelled
   `depends-on-#N`) waits until every #N is merged; then it is rebased onto
   its base under rule 2's terms, and auto-merge is enabled if the PR asks
   for it (label `automerge`, or "auto-merge" in the body). It never
   cancels an auto-merge someone else scheduled.

State (what it already did) lives in a JSON file, so restarts are
idempotent. Every decision is logged. --dry-run reads everything and runs
the trial rebases locally, but performs no write: no rerun, push, comment
or merge.

Python stdlib only, plus scripts/gitea_api.py next to this file. git must
be on PATH.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

from gitea_api import SERVER, ApiError, Client

REPOS = ("geisten/geistlib", "geisten/geist-runtime")
INTERVAL = 90  # seconds between polls
RERUN_MAX = 2
INFRA_MAX_SECONDS = 300
LABEL_REBASE = "ci-bot:rebase"
LABEL_AUTOMERGE = "automerge"
MARKER = "<!-- ci-bot:{} -->"
SETUP_STEP = re.compile(r"^set up job|checkout|ccache|^prepare", re.I)
BOT_COMMIT = re.compile(r"^co-authored-by:\s*claude\b|noreply@anthropic\.com", re.I | re.M)
DEPENDS = re.compile(r"\bdepends\s+on\s+#(\d+)", re.I)
DEPENDS_LABEL = re.compile(r"^depends-on-#?(\d+)$", re.I)
AUTOMERGE = re.compile(r"\bauto-?merge", re.I)
NO_AUTOMERGE = re.compile(r"\b(no|not|don'?t|do not|without)\s+auto-?merge", re.I)


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


# ---- pure decision logic (tests/test_ci_bot_py.py) -------------------------

def parse_ts(s: str | None) -> dt.datetime | None:
    if not s or s.startswith(("0001-", "1970-")):
        return None
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def is_zombie_sweep(completed_at: str | None) -> bool:
    """Gitea's zombie sweep stops tasks at hh:x0:25 / hh:x5:25 (a hint)."""
    t = parse_ts(completed_at)
    return bool(t) and t.minute % 5 == 0 and 20 <= t.second <= 30


def job_seconds(job: dict) -> float | None:
    a, b = parse_ts(job.get("started_at")), parse_ts(job.get("completed_at"))
    return (b - a).total_seconds() if a and b else None


def classify_job(job: dict, logs_found: bool | None = None) -> tuple[str, str]:
    """('infra'|'real'|'logs?', reason). 'logs?' asks the caller to look up
    whether the job's log exists and call again with logs_found."""
    steps = sorted(job.get("steps") or [], key=lambda s: s.get("number", 0))
    failed = [s for s in steps if s.get("conclusion") == "failure"]
    secs = job_seconds(job)
    zombie = is_zombie_sweep(job.get("completed_at"))
    if not failed and (not steps or steps[0].get("conclusion") != "success"):
        # Only always() steps (diagnostics upload) may have run.
        return "infra", "first step never ran: the job died in 'Set up job'"
    if len(steps) > 1 and len(failed) == len(steps):
        return "infra", "every step marked failed: task lost by the runner"
    if failed:
        first = failed[0]
        before_ok = all(s.get("conclusion") in ("success", "skipped")
                        for s in steps if s.get("number", 0) < first.get("number", 0))
        quick = secs is not None and secs < INFRA_MAX_SECONDS
        if SETUP_STEP.search(first.get("name", "")) and before_ok and (quick or zombie):
            how = f"after {secs:.0f} s" if quick else "at a zombie sweep"
            return "infra", f"setup step '{first['name'][:40]}' failed {how}"
    if logs_found is None:
        return "logs?", ""
    if not logs_found:
        return "infra", "logs not found"
    where = f"step '{failed[0]['name'][:50]}'" if failed else "no failed step"
    return "real", f"failed at {where}"


def run_ids(statuses: list[dict]) -> list[int]:
    """Run IDs behind the failing commit statuses (target_url .../runs/<id>/...)."""
    ids = set()
    for s in statuses:
        if s.get("status") in ("failure", "error"):
            m = re.search(r"/actions/runs/(\d+)", s.get("target_url") or "")
            if m:
                ids.add(int(m.group(1)))
    return sorted(ids)


def is_bot_commit(c: dict) -> bool:
    msg = c.get("commit", {}).get("message", "")
    email = c.get("commit", {}).get("author", {}).get("email", "")
    login = (c.get("author") or {}).get("login", "")
    return bool(BOT_COMMIT.search(msg) or email.endswith("@anthropic.com")
                or login.endswith(("[bot]", "-bot")) or login == "ci-bot")


def may_rewrite(pr: dict, commits: list[dict], repo: str) -> tuple[bool, str]:
    """May ci-bot force-push this PR's branch?"""
    head_repo = ((pr.get("head") or {}).get("repo") or {}).get("full_name")
    if head_repo != repo:
        return False, f"head branch lives in {head_repo}, not {repo}"
    if any(l.get("name") == LABEL_REBASE for l in pr.get("labels") or []):
        return True, f"label {LABEL_REBASE}"
    own = [c for c in commits if len(c.get("parents") or []) < 2]
    if own and all(is_bot_commit(c) for c in own):
        return True, "every non-merge commit is Claude's or a bot's"
    return False, f"a human's commits and no label {LABEL_REBASE}"


def depends_on(pr: dict) -> list[int]:
    deps = {int(n) for n in DEPENDS.findall(pr.get("body") or "")}
    for l in pr.get("labels") or []:
        m = DEPENDS_LABEL.match(l.get("name", ""))
        if m:
            deps.add(int(m.group(1)))
    deps.discard(pr.get("number"))
    return sorted(deps)


def wants_automerge(pr: dict) -> bool:
    if any(l.get("name") == LABEL_AUTOMERGE for l in pr.get("labels") or []):
        return True
    body = pr.get("body") or ""
    return bool(AUTOMERGE.search(body)) and not NO_AUTOMERGE.search(body)


def resolve_union(text: str) -> str | None:
    """Resolves every conflict block by keeping both sides (ours first, then
    the lines of theirs that ours lacks); handles diff3 base sections.
    None if the markers are unbalanced."""
    out, ours, theirs, part = [], [], [], None
    for line in text.splitlines(keepends=True):
        if line.startswith("<<<<<<< ") or line.rstrip("\n") == "<<<<<<<":
            if part is not None:
                return None
            part, ours, theirs = "ours", [], []
        elif part and line.startswith("|||||||"):
            part = "base"
        elif part and line.rstrip("\n") == "=======":
            part = "theirs"
        elif part == "theirs" and line.startswith(">>>>>>>"):
            seen = set(ours)
            out += ours + [l for l in theirs if l not in seen or not l.strip()]
            part = None
        elif part == "ours":
            ours.append(line)
        elif part == "theirs":
            theirs.append(line)
        elif part == "base":
            pass
        else:
            out.append(line)
    return None if part else "".join(out)


# ---- Gitea API --------------------------------------------------------------

class Gitea(Client):
    """Reads retry; writes are skipped (logged) in dry-run."""

    def __init__(self, token: str, dry_run: bool, server: str = SERVER):
        super().__init__(token, server + "/api/v1")
        self.dry = dry_run

    def get(self, path: str, raw: bool = False, tries: int = 4):
        return self.call("GET", path, raw=raw, tries=tries)

    def write(self, method: str, path: str, body=None, what: str = ""):
        if self.dry:
            log(f"  DRY-RUN would {what or method + ' ' + path}")
            return None
        return self.call(method, path, body, tries=2)


# ---- git (a private partial clone per repository) ---------------------------

class Git:
    def __init__(self, home: str, repo: str, token: str):
        self.dir = os.path.join(home, "repos", repo.replace("/", "_") + ".git")
        self.url = f"{SERVER}/{repo}.git"
        askpass = os.path.join(home, "askpass.sh")
        if not os.path.exists(askpass):
            with open(askpass, "w") as f:
                f.write('#!/bin/sh\ncase "$1" in Username*) echo ci-bot ;; '
                        '*) printf "%s\\n" "$CI_BOT_GIT_TOKEN" ;; esac\n')
            os.chmod(askpass, 0o700)
        self.env = dict(os.environ, GIT_ASKPASS=askpass, CI_BOT_GIT_TOKEN=token,
                        GIT_TERMINAL_PROMPT="0", GIT_EDITOR="true", LC_ALL="C")

    def run(self, *args, cwd=None, check=True) -> subprocess.CompletedProcess:
        p = subprocess.run(["git", "-c", "credential.helper=", *args], cwd=cwd or self.dir,
                           env=self.env, capture_output=True, text=True, timeout=600)
        if check and p.returncode:
            raise RuntimeError(f"git {args[0]}: {(p.stderr or p.stdout).strip()[-300:]}")
        return p

    def fetch(self, *refspecs: str) -> None:
        if not os.path.isdir(self.dir):
            os.makedirs(os.path.dirname(self.dir), exist_ok=True)
            subprocess.run(["git", "init", "-q", "--bare", self.dir], check=True)
            self.run("config", "remote.origin.url", self.url)
            self.run("config", "remote.origin.promisor", "true")
            self.run("config", "remote.origin.partialclonefilter", "blob:none")
            self.run("config", "user.name", "ci-bot")
            self.run("config", "user.email", "ci-bot@geisten.net")
        self.run("fetch", "-q", "--filter=blob:none", "origin", *refspecs)

    def rev(self, ref: str) -> str:
        return self.run("rev-parse", ref).stdout.strip()

    def trial_rebase(self, head: str, onto: str) -> tuple[str, list[str], str | None]:
        """Rebases head onto onto in a scratch worktree. Returns
        ('clean'|'changelog'|'conflict'|'uptodate', conflicting files, new head)."""
        if self.run("merge-base", "--is-ancestor", onto, head, check=False).returncode == 0:
            return "uptodate", [], head
        wt = tempfile.mkdtemp(prefix="ci-bot-wt-")
        try:
            self.run("worktree", "add", "-q", "--detach", wt, head)
            how, files = "clean", []
            p = self.run("rebase", "-q", onto, cwd=wt, check=False)
            for _ in range(500):
                if p.returncode == 0:
                    break
                files = self.run("diff", "--name-only", "--diff-filter=U", cwd=wt).stdout.split()
                if files != ["CHANGELOG.md"]:
                    self.run("rebase", "--abort", cwd=wt, check=False)
                    return "conflict", files or ["(rebase stopped without a file conflict)"], None
                path = os.path.join(wt, "CHANGELOG.md")
                merged = resolve_union(open(path).read())
                if merged is None:
                    self.run("rebase", "--abort", cwd=wt, check=False)
                    return "conflict", ["CHANGELOG.md (unbalanced markers)"], None
                open(path, "w").write(merged)
                self.run("add", "CHANGELOG.md", cwd=wt)
                how = "changelog"
                p = self.run("rebase", "--continue", cwd=wt, check=False)
            else:
                self.run("rebase", "--abort", cwd=wt, check=False)
                return "conflict", ["(rebase did not finish)"], None
            return how, files if how == "changelog" else [], self.rev_in(wt)
        finally:
            self.run("worktree", "remove", "--force", wt, check=False)
            shutil.rmtree(wt, ignore_errors=True)

    def rev_in(self, wt: str) -> str:
        return self.run("rev-parse", "HEAD", cwd=wt).stdout.strip()

    def push(self, new: str, branch: str, expect: str) -> None:
        self.run("push", "-q", f"--force-with-lease=refs/heads/{branch}:{expect}",
                 "origin", f"{new}:refs/heads/{branch}")


# ---- the bot ----------------------------------------------------------------

class Bot:
    def __init__(self, api: Gitea, home: str, state_path: str, token: str, repos=REPOS):
        self.api, self.home, self.state_path, self.token, self.repos = api, home, state_path, token, repos
        self.gits: dict[str, Git] = {}
        try:
            with open(state_path) as f:
                self.state = json.load(f)
        except (OSError, ValueError):
            self.state = {}
        for k in ("reruns", "seen", "rebased", "commented", "automerge"):
            self.state.setdefault(k, {})

    def save(self) -> None:
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=1, sort_keys=True)
        os.replace(tmp, self.state_path)

    def once_log(self, key: str, msg: str) -> None:
        """Logs a decision once per key (state 'seen')."""
        if self.state["seen"].get(key) != msg:
            self.state["seen"][key] = msg
            log(msg)

    def git(self, repo: str) -> Git:
        if repo not in self.gits:
            self.gits[repo] = Git(self.home, repo, self.token)
        return self.gits[repo]

    def tick(self) -> None:
        open_keys, listed = set(), True
        for repo in self.repos:
            try:
                prs = self.api.get(f"/repos/{repo}/pulls?state=open&limit=50")
            except ApiError as e:
                log(f"{repo}: listing PRs failed: {e}")
                listed = False
                continue
            for pr in prs:
                tag = f"{repo}#{pr['number']}"
                open_keys.add(tag)
                for step in (self.ci, self.deps, self.conflict):
                    try:
                        if step(repo, tag, pr):
                            break      # this PR's head changed; next poll sees it fresh
                    except Exception as e:  # noqa: BLE001 — one bad PR must not stop the others
                        log(f"{tag}: {step.__name__} error: {type(e).__name__}: {e}")
        # Forget closed PRs (a PR key is 'owner/repo#n' or 'owner/repo#n:...'),
        # but only after a poll that saw every repository's list.
        for k in () if not listed else ("reruns", "seen", "rebased", "commented", "automerge"):
            self.state[k] = {key: v for key, v in self.state[k].items()
                             if key.split(":")[0] in open_keys}
        self.save()

    # -- 1. infra reruns
    def ci(self, repo: str, tag: str, pr: dict) -> bool:
        sha = pr["head"]["sha"]
        st = self.api.get(f"/repos/{repo}/commits/{sha}/status?limit=100") or {}
        for run_id in run_ids(st.get("statuses") or []):
            run = self.api.get(f"/repos/{repo}/actions/runs/{run_id}")
            if run.get("status") != "completed" or run.get("head_sha") != sha:
                continue
            key = f"{tag}:run{run_id}:{sha[:12]}"
            attempt = run.get("run_attempt", 1)
            if self.state["seen"].get(f"{key}:a{attempt}"):
                continue                     # this attempt was decided already
            jobs = (self.api.get(f"/repos/{repo}/actions/runs/{run_id}/jobs?limit=100") or {}).get("jobs") or []
            failed = [j for j in jobs if j.get("conclusion") == "failure"]
            if not failed:
                continue
            verdicts = []
            for j in failed:
                kind, why = classify_job(j)
                if kind == "logs?":
                    try:
                        self.api.get(f"/repos/{repo}/actions/jobs/{j['id']}/logs", raw=True, tries=2)
                        found = True
                    except ApiError as e:
                        if e.code != 404:
                            raise
                        found = False
                    kind, why = classify_job(j, found)
                verdicts.append((kind, j, why))
            for kind, j, why in verdicts:
                log(f"{tag} run {run_id} attempt {attempt}: job {j['id']} '{j['name'][:60]}': {kind} ({why})")
            done = self.state["reruns"].get(key, 0)
            if any(k == "real" for k, _, _ in verdicts):
                decision = "real failure present: left alone"
            elif done >= RERUN_MAX:
                decision = f"infra only, but {done} reruns already: left alone"
            else:
                self.api.write("POST", f"/repos/{repo}/actions/runs/{run_id}/rerun-failed-jobs",
                               what=f"POST rerun-failed-jobs on run {run_id}")
                self.state["reruns"][key] = done + 1
                decision = f"infra only: rerun failed jobs (rerun {done + 1}/{RERUN_MAX})"
            log(f"{tag} run {run_id} attempt {attempt}: {decision}")
            self.state["seen"][f"{key}:a{attempt}"] = decision
        return False

    # -- comments, once per PR and kind (marker in the comment body)
    def comment_once(self, repo: str, tag: str, pr: dict, kind: str, text: str) -> None:
        key = f"{tag}:{kind}"
        if self.state["commented"].get(key):
            return
        marker = MARKER.format(kind)
        comments = self.api.get(f"/repos/{repo}/issues/{pr['number']}/comments?limit=50") or []
        if not any(marker in (c.get("body") or "") for c in comments):
            self.api.write("POST", f"/repos/{repo}/issues/{pr['number']}/comments",
                           {"body": f"{text}\n\n{marker}"}, what=f"comment on {tag}: {text.splitlines()[0]}")
            log(f"{tag}: commented ({kind})")
        self.state["commented"][key] = True

    # -- 2. rebase (shared by conflicts and dependencies)
    def rebase(self, repo: str, tag: str, pr: dict, why: str) -> str:
        """Returns 'pushed', 'dry' (would push), 'uptodate', 'conflict',
        'denied' (may not push), 'done' (decided before) or 'moved'."""
        base, head, sha = pr["base"]["ref"], pr["head"]["ref"], pr["head"]["sha"]
        g = self.git(repo)
        g.fetch(f"+refs/heads/{base}:refs/remotes/origin/{base}",
                f"+refs/pull/{pr['number']}/head:refs/ci-bot/pr/{pr['number']}")
        onto = g.rev(f"refs/remotes/origin/{base}")
        key = f"{tag}:{sha[:12]}:{onto[:12]}"
        if key in self.state["rebased"]:
            return self.state["rebased"][key]
        if g.rev(f"refs/ci-bot/pr/{pr['number']}") != sha:
            return "moved"                     # the PR moved since the listing
        how, files, new = g.trial_rebase(sha, onto)
        if how == "uptodate":
            out, msg = "uptodate", f"already on {base}, nothing to rebase"
        elif how == "conflict":
            out, msg = "conflict", f"conflicts beyond CHANGELOG.md ({', '.join(files[:8])}): left alone"
            self.comment_once(repo, tag, pr, "conflict",
                              f"ci-bot: this branch conflicts with `{base}` in "
                              f"{', '.join('`' + f + '`' for f in files[:20])}, beyond "
                              "CHANGELOG.md, so it needs a manual rebase or merge.")
        else:
            commits = self.api.get(f"/repos/{repo}/pulls/{pr['number']}/commits"
                                   "?limit=100&stat=false&verification=false&files=false") or []
            ok, who = may_rewrite(pr, commits, repo)
            if not ok:
                out, msg = "denied", f"rebase onto {base} would be {how}, but may not push ({who})"
                self.comment_once(repo, tag, pr, "needs-label",
                                  f"ci-bot: the only conflict with `{base}` is CHANGELOG.md, which it "
                                  "could resolve, but it rewrites a human's branch only with the "
                                  f"label `{LABEL_REBASE}`.")
            elif self.api.dry:
                out, msg = "dry", f"DRY-RUN would force-push {head} {sha[:10]} -> {new[:10]} ({how}; {who})"
            else:
                g.push(new, head, sha)
                out, msg = "pushed", f"rebased onto {base} {onto[:10]} ({how}; {who}): {sha[:10]} -> {new[:10]}"
        log(f"{tag}: {why}: {msg}")
        self.state["rebased"][key] = out
        return out

    def conflict(self, repo: str, tag: str, pr: dict) -> bool:
        if pr.get("mergeable") is not False:
            return False
        return self.rebase(repo, tag, pr, "not mergeable") == "pushed"

    # -- 3. dependent merges
    def deps(self, repo: str, tag: str, pr: dict) -> bool:
        deps = depends_on(pr)
        if not deps:
            return False
        waiting, closed = [], []
        for n in deps:
            d = self.api.get(f"/repos/{repo}/pulls/{n}")
            if not d.get("merged"):
                (closed if d.get("state") == "closed" else waiting).append(n)
        if closed:
            self.once_log(f"{tag}:deps", f"{tag}: dependency {closed} closed unmerged: left alone")
            self.comment_once(repo, tag, pr, "dep-closed",
                              f"ci-bot: dependency {', '.join('#%d' % n for n in closed)} was closed "
                              "without merging; this PR is left alone.")
            return False
        if waiting:
            self.once_log(f"{tag}:deps", f"{tag}: waits for {', '.join('#%d' % n for n in waiting)}; "
                          "no auto-merge yet")
            return False
        akey = f"{tag}:automerge"
        if self.state["automerge"].get(akey):
            return False                       # handled; later conflicts are rule 2's
        self.once_log(f"{tag}:deps", f"{tag}: dependencies {deps} merged")
        out = self.rebase(repo, tag, pr, f"dependencies {deps} merged")
        if out == "pushed":
            return True                        # enable auto-merge on the new head next poll
        if out in ("conflict", "moved"):
            return False
        if not wants_automerge(pr):
            self.once_log(akey, f"{tag}: ready, but the PR does not ask for auto-merge")
            self.state["automerge"][akey] = "not asked"
            return False
        try:
            self.api.write("POST", f"/repos/{repo}/pulls/{pr['number']}/merge",
                           {"Do": "merge", "merge_when_checks_succeed": True},
                           what=f"enable auto-merge (merge commit) on {tag}")
            log(f"{tag}: auto-merge {'(dry-run) ' if self.api.dry else ''}enabled")
        except ApiError as e:
            if e.code != 409:                  # 409: already scheduled
                raise
            log(f"{tag}: auto-merge was already scheduled")
        self.state["automerge"][akey] = "enabled"
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--home", default=os.path.expanduser("~/ci-bot"))
    ap.add_argument("--token-file", default=os.path.expanduser("~/.config/ci-bot/token"))
    ap.add_argument("--state", help="default: <home>/state.json (dry-run: state-dry.json)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="no rerun, push, comment or merge")
    a = ap.parse_args()
    with open(a.token_file) as f:
        token = f.read().strip()
    if not token:
        print("ci-bot: empty token file", file=sys.stderr)
        return 1
    os.makedirs(a.home, exist_ok=True)
    state = a.state or os.path.join(a.home, "state-dry.json" if a.dry_run else "state.json")
    bot = Bot(Gitea(token, a.dry_run), a.home, state, token)
    log(f"ci-bot start ({'dry-run' if a.dry_run else 'live'}; repos {','.join(REPOS)}; every {INTERVAL} s)")
    while True:
        try:
            bot.tick()
        except Exception as e:  # noqa: BLE001 — the service must outlive any single bad poll
            log(f"tick failed: {type(e).__name__}: {e}")
        if a.once:
            return 0
        time.sleep(INTERVAL)


if __name__ == "__main__":
    sys.exit(main())
