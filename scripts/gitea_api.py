"""gitea_api.py — the Gitea REST client of the CI scripts (ci-bot.py,
ci-issues.py, required_checks.py). Stdlib only: the CI images have nothing
else, and ci-bot runs on the host's system python.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

SERVER = "https://git.geisten.net"


class ApiError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(f"HTTP {code}: {msg}")
        self.code = code


class Client:
    """Calls below `base` (e.g. SERVER + "/api/v1") with a token."""

    def __init__(self, token: str, base: str = SERVER + "/api/v1"):
        self.token, self.base = token, base

    def call(self, method: str, path: str, body=None, raw: bool = False, tries: int = 1):
        """Parsed JSON (None for an empty body), or the bytes with raw=True.

        With tries > 1, a 5xx or a network error is retried with a growing
        pause; a 4xx never is. The last failure raises ApiError (code 0: no
        HTTP answer)."""
        data = json.dumps(body).encode() if body is not None else None
        for i in range(tries):
            req = urllib.request.Request(self.base + path, data=data, method=method, headers={
                "Authorization": f"token {self.token}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    b = r.read()
                    return b if raw else (json.loads(b) if b.strip() else None)
            except urllib.error.HTTPError as e:
                msg = e.read()[:200].decode(errors="replace").strip()
                if e.code < 500 or i == tries - 1:
                    raise ApiError(e.code, msg) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                if i == tries - 1:
                    raise ApiError(0, str(e)) from None
            time.sleep(5 * (i + 1))

    def pages(self, path: str, key: str | None = None) -> list:
        """Every item of a paginated GET (50 per page); `key` picks the list
        out of an object answer."""
        out, page = [], 1
        sep = "&" if "?" in path else "?"
        while True:
            got = self.call("GET", f"{path}{sep}limit=50&page={page}")
            items = got.get(key, []) if key else got
            out += items
            if len(items) < 50:
                return out
            page += 1
