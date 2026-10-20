#!/usr/bin/env python3
"""Read-only caching git smart-HTTP proxy for the CI runners' job containers.

Every CI job clones geistlib from git.geisten.net, a 1-vCPU VM; a run of 19
jobs made it pack and send ~9 MB nineteen times, and on busy days its network
dropped packets and checkouts hung until Gitea killed them as zombies. This
server keeps a mirror per repository on the runner host. Before it answers
a fetch (`info/refs`) it brings the mirror up to date with one incremental
fetch from Gitea — kilobytes — and then serves the clone itself through
`git http-backend`. Concurrent requests share one upstream fetch.

The job containers reach it through git configuration in the runner's
environment (docs/HARDWARE_CI.md, "CI git cache"):
  url.http://<host>:8417/.insteadOf      = https://git.geisten.net/
  url.https://git.geisten.net/.pushInsteadOf = https://git.geisten.net/
so fetches come here and pushes still go to Gitea. Pushes to this server are
refused; only container networks and localhost may connect.

Usage: git-cache.py --listen 192.168.178.43:8417 --root ~/git-cache/repos \
           --token-file ~/.config/git-cache/token
--listen may be given more than once (the Pi: its docker bridge for the
container jobs and 127.0.0.1 for the host-mode jobs).
"""
import argparse
import ipaddress
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "https://git.geisten.net"
ALLOWED = [ipaddress.ip_network(n) for n in ("127.0.0.0/8", "172.16.0.0/12", "::1/128")]
# /owner/repo(.git)/info/refs | /owner/repo(.git)/git-upload-pack | .../HEAD
PATH_RE = re.compile(r"^/([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+?)(?:\.git)?(/info/refs|/git-upload-pack|/HEAD)$")

ARGS = None
LOCKS = {}
LOCKS_GUARD = threading.Lock()
LAST_FETCH_START = {}  # repo dir -> monotonic time the last upstream fetch began


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def git(*args, cwd=None, timeout=45):
    """Upstream git with the token. A hung Gitea must not hang the job: after
    `timeout` the caller serves the mirror it has (a run's later jobs find
    their commit there) instead of holding the checkout until Gitea stops
    the job as a zombie (10 min)."""
    token = open(ARGS.token_file).read().strip()
    cmd = ["git", "-c", f"http.extraHeader=Authorization: token {token}", *args]
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", f"timed out after {timeout} s")


def refresh(owner, repo, asked_at):
    """Make the mirror at least as new as the moment the request arrived."""
    d = os.path.join(ARGS.root, owner, repo + ".git")
    with LOCKS_GUARD:
        lock = LOCKS.setdefault(d, threading.Lock())
    with lock:
        if LAST_FETCH_START.get(d, -1) >= asked_at:
            return True  # a fetch that began after this request covers it
        LAST_FETCH_START[d] = time.monotonic()
        t0 = time.monotonic()
        kind = "fetch" if os.path.isdir(d) else "clone"
        if kind == "clone":
            os.makedirs(os.path.dirname(d), exist_ok=True)
            r = git("clone", "--mirror", "--quiet", f"{UPSTREAM}/{owner}/{repo}.git", d, timeout=600)
            if r.returncode == 0:
                # actions/checkout fetches the PR head by object id
                subprocess.run(["git", "config", "uploadpack.allowAnySHA1InWant", "true"], cwd=d)
        else:
            r = git("fetch", "--prune", "--quiet", "origin", cwd=d)
        ok = r.returncode == 0
        log(f"{owner}/{repo}: {kind} "
            f"{'ok' if ok else 'FAILED: ' + r.stderr.strip()[-300:]} in {time.monotonic() - t0:.2f} s")
        return ok or os.path.isdir(d)  # stale but present: serve what we have


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def allowed(self):
        try:
            return any(ipaddress.ip_address(self.client_address[0]) in n for n in ALLOWED)
        except ValueError:
            return False

    def fail(self, code, text):
        body = (text + "\n").encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_git(self):
        asked_at = time.monotonic()
        if not self.allowed():
            return self.fail(403, "forbidden")
        path, _, query = self.path.partition("?")
        m = PATH_RE.match(path)
        if not m or "receive-pack" in query:
            return self.fail(403, "read-only git cache")
        owner, repo, tail = m.groups()
        if tail == "/info/refs" and not refresh(owner, repo, asked_at):
            return self.fail(502, "upstream fetch failed and no mirror")
        body = b""
        if self.command == "POST":
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "GIT_PROJECT_ROOT": ARGS.root,
            "GIT_HTTP_EXPORT_ALL": "1",
            "PATH_INFO": f"/{owner}/{repo}.git{tail}",
            "QUERY_STRING": query,
            "REQUEST_METHOD": self.command,
            "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            "CONTENT_LENGTH": str(len(body)),
            "REMOTE_ADDR": self.client_address[0],
        }
        for h in ("Git-Protocol", "Content-Encoding"):
            if self.headers.get(h):
                env["HTTP_" + h.upper().replace("-", "_")] = self.headers[h]
        if self.headers.get("Git-Protocol"):
            env["GIT_PROTOCOL"] = self.headers["Git-Protocol"]
        p = subprocess.Popen(["git", "http-backend"], env=env, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p.stdin.write(body)
        p.stdin.close()
        status, headers = 200, []
        for line in iter(p.stdout.readline, b""):
            line = line.rstrip(b"\r\n")
            if not line:
                break
            k, _, v = line.decode().partition(":")
            if k.lower() == "status":
                status = int(v.split()[0])
            else:
                headers.append((k, v.strip()))
        self.send_response(status)
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()  # HTTP/1.0: the body ends when the connection closes
        while chunk := p.stdout.read(65536):
            self.wfile.write(chunk)
        p.wait()
        if tail == "/git-upload-pack":
            log(f"{owner}/{repo}: served {self.client_address[0]}")

    do_GET = handle_git
    do_POST = handle_git


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", action="append", help="host:port, repeatable (default 127.0.0.1:8417)")
    ap.add_argument("--root", default=os.path.expanduser("~/git-cache/repos"))
    ap.add_argument("--token-file", default=os.path.expanduser("~/.config/git-cache/token"))
    ARGS = ap.parse_args()
    listen = ARGS.listen or ["127.0.0.1:8417"]
    os.makedirs(ARGS.root, exist_ok=True)
    servers = []
    for addr in listen:
        host, port = addr.rsplit(":", 1)
        servers.append(ThreadingHTTPServer((host, int(port)), Handler))
    log(f"git cache on {' '.join(listen)}, mirrors in {ARGS.root}")
    for srv in servers[1:]:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    servers[0].serve_forever()


if __name__ == "__main__":
    sys.exit(main())
