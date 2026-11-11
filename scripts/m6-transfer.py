#!/usr/bin/env python3
"""Transfer one reviewed, private Gitea package and verify it before staging."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

ORIGIN = "https://git.geisten.net/api/packages/geisten/generic/geist-lab-qwen32k-m6/v2026.11.10"
MAX_PART = 128 * 1024 * 1024
BLOCK = 1024 * 1024


def check(ok, message):
    if not ok:
        raise ValueError(message)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request(name, method="GET", data=None, size=None):
    check(re.fullmatch(r"[a-z0-9.-]+", name), "Invalid package file name")
    user = os.environ.get("QWEN_M6_PACKAGE_USER")
    token = os.environ.get("QWEN_M6_PACKAGE_TOKEN")
    check(user and token, "Supply the package username and scoped token via environment")
    auth = base64.b64encode((user + ":" + token).encode()).decode()
    headers = {"Authorization": "Basic " + auth}
    if size is not None:
        headers.update({"Content-Type": "application/octet-stream", "Content-Length": str(size)})
    req = Request(f"{ORIGIN}/{name}", data=data, headers=headers, method=method)
    return build_opener(NoRedirect()).open(req, timeout=60)


def load_manifest(data, expected):
    check(re.fullmatch(r"[0-9a-f]{64}", expected)
          and hashlib.sha256(data).hexdigest() == expected, "Manifest SHA-256 mismatch")
    m = json.loads(data)
    name = m["snapshot"]
    check(m["schema"] == "geist-lab-m6-transfer-v1" and m["origin"] == ORIGIN
          and re.fullmatch(r"snapshot-\d{8}(-cutover)?", name)
          and set(m["archives"]) == {"assets", name}, "Invalid transfer manifest")
    for archive, a in m["archives"].items():
        parts = a["parts"]
        check(0 < len(parts) < 1000 and re.fullmatch(r"[0-9a-f]{64}", a["archive_sha256"]),
              "Invalid archive descriptor")
        for i, part in enumerate(parts):
            check(part["file"] == f"{archive}.tar.{i:03d}"
                  and 0 < part["bytes"] <= MAX_PART
                  and re.fullmatch(r"[0-9a-f]{64}", part["sha256"]), "Invalid part descriptor")
    return m


def verify(stream, part, output=None, archive_hash=None):
    total, digest = 0, hashlib.sha256()
    while True:
        data = stream.read(BLOCK)
        if not data:
            break
        total += len(data)
        check(total <= part["bytes"], f"Part too large: {part['file']}")
        digest.update(data)
        if archive_hash is not None:
            archive_hash.update(data)
        if output is not None:
            output.write(data)
    check(total == part["bytes"] and digest.hexdigest() == part["sha256"],
          f"Part verification failed: {part['file']}")


def verify_archive(source, archive):
    digest = hashlib.sha256()
    for part in archive["parts"]:
        with (source / part["file"]).open("rb") as f:
            verify(f, part, archive_hash=digest)
    check(digest.hexdigest() == archive["archive_sha256"], "Archive SHA-256 mismatch")


def publish(path, part):
    with path.open("rb") as body:
        verify(body, part)
        body.seek(0)
        try:
            with request(part["file"], "PUT", body, part["bytes"]) as resp:
                check(resp.status == 201, f"Unexpected upload status {resp.status}")
        except HTTPError as e:
            if e.code != 409:
                raise
            with request(part["file"]) as existing:
                verify(existing, part)
    print(f"Stored {part['file']}", flush=True)


def upload(source, expected):
    data = (source / "transfer.json").read_bytes()
    manifest = load_manifest(data, expected)
    for a in manifest["archives"].values():
        verify_archive(source, a)
    for a in manifest["archives"].values():
        for part in a["parts"]:
            publish(source / part["file"], part)
    publish(source / "transfer.json",
            {"file": "transfer.json", "bytes": len(data), "sha256": expected})


def extract(source, name, archive, incoming, expected=None):
    verify_archive(source, archive)
    dest = incoming / name
    check(not dest.exists(), f"Refusing to overwrite {dest}")
    with tempfile.TemporaryDirectory(prefix=".m6-import-", dir=incoming) as folder:
        root = Path(folder)
        parts = [str(source / p["file"]) for p in archive["parts"]]
        with subprocess.Popen(["/bin/cat", *parts], stdout=subprocess.PIPE) as cat:
            with tarfile.open(fileobj=cat.stdout, mode="r|") as tar:
                for member in tar:
                    rel = PurePosixPath(member.name)
                    check(not rel.is_absolute() and rel.parts and rel.parts[0] == name
                          and ".." not in rel.parts and (member.isdir() or member.isfile()),
                          "Disallowed archive member")
                    path = root.joinpath(*rel.parts)
                    if member.isdir():
                        path.mkdir(mode=0o700, parents=True, exist_ok=True)
                    else:
                        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                        with tar.extractfile(member) as f, path.open("xb") as out:
                            shutil.copyfileobj(f, out, BLOCK)
                        path.chmod(0o700 if rel.as_posix() == "assets/runner" else 0o600)
                # Consume tar padding before waiting for the producer.
                while cat.stdout.read(BLOCK):
                    pass
            check(cat.wait() == 0, "Reading archive parts failed")
        check((root / name).is_dir(), "Missing archive root")
        if expected is not None:
            (root / name / ".transfer-verified").write_text(expected + "\n")
        (root / name).rename(dest)
    print(f"Staged {dest}", flush=True)


def receive(incoming, expected):
    host = subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.model"], text=True).strip()
    check(host == "Mac18,5", "This transfer job is restricted to the Mac mini M6")
    incoming.mkdir(mode=0o700, parents=True, exist_ok=True)
    with request("transfer.json") as resp:
        data = resp.read(1024 * 1024 + 1)
    check(len(data) <= 1024 * 1024, "Transfer manifest too large")
    manifest = load_manifest(data, expected)
    store = incoming / ".transfer" / expected
    store.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name, archive in manifest["archives"].items():
        receipt = incoming / name / ".transfer-verified"
        if receipt.is_file():
            check((incoming / name).is_dir() and receipt.read_text().strip() == expected,
                  f"Invalid receipt for {name}")
            continue
        for part in archive["parts"]:
            path = store / part["file"]
            if path.is_file():
                with path.open("rb") as f:
                    verify(f, part)
                continue
            tmp = path.with_suffix(path.suffix + ".tmp")
            with request(part["file"]) as resp, tmp.open("wb") as out:
                verify(resp, part, output=out)
            tmp.replace(path)
        extract(store, name, archive, incoming, expected)
    print(f"Mac mini M6 received and verified assets and {manifest['snapshot']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("upload", "receive"))
    p.add_argument("path", type=Path)
    p.add_argument("manifest_sha256")
    args = p.parse_args()
    try:
        if args.operation == "upload":
            upload(args.path.resolve(), args.manifest_sha256)
        else:
            receive(args.path.resolve(), args.manifest_sha256)
    except HTTPError as e:
        p.exit(1, f"Package request denied or failed with HTTP {e.code}.\n")
    except (OSError, ValueError, KeyError, tarfile.TarError) as e:
        p.exit(1, f"Transfer failed: {e}\n")


if __name__ == "__main__":
    main()
