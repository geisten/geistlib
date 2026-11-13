#!/usr/bin/env python3
"""Export, validate and run an explicitly approved Qwen handoff."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile

PYTHON = Path("/opt/homebrew/opt/python@3.12/bin/python3.12")
DYLIBS = (
    Path("/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/Python"),
    Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"),
)
MODELS = {"baseline": "base-f32.gguf", "edited": "p50c-f32.gguf"}
PACKAGES = {
    "math-verify": "math_verify",
    "latex2sympy2-extended": "latex2sympy2_extended",
    "antlr4-python3-runtime": "antlr4",
    "sympy": "sympy",
    "mpmath": "mpmath",
}
PARITY_FIELDS = ("index", "seed", "tokens", "prompt_tokens", "found", "parsed",
                 "correct", "cutoff", "prediction", "generation_sha256", "logits_sha256")
RELOCATED = {"baseline_model", "edited_model", "baseline_stamp", "edited_stamp", "python"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def stamp(path):
    s = path.stat()
    m, c = divmod(s.st_mtime_ns, 10**9), divmod(s.st_ctime_ns, 10**9)
    return f"{s.st_dev}:{s.st_ino}:{s.st_size}:{m[0]}:{m[1]}:{c[0]}:{c[1]}"


def checked_copy(src, dst, digest=None, clone=False):
    before = stamp(src)
    if not clone or subprocess.run(["/bin/cp", "-c", src, dst], capture_output=True).returncode:
        shutil.copy2(src, dst)
    actual = sha(dst)
    require(actual == (digest or sha(src)) and stamp(src) == before,
            f"Copy integrity check failed for {src}")
    return actual


def read_plan(root):
    plan_sha = sha(root / "plan.json")
    require((root / "plan.sha256").read_text().strip() == plan_sha, "Frozen plan hash differs")
    return read_json(root / "plan.json"), plan_sha


def manifest_sha(plan):
    settings = {k: v for k, v in plan.items() if k not in RELOCATED}
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def case_status(root):
    proc = subprocess.run([root / "runner", "--benchmark-status", root],
                          capture_output=True, text=True)
    require(proc.returncode in (0, 10), proc.stderr.strip() or "Invalid C journal")
    return json.loads(proc.stdout)


def model_hashes(plan):
    return {name: plan[f"{key}_sha256"] for key, name in MODELS.items()}


def make_assets(source, bundle, plan):
    assets = bundle / "assets"
    if assets.exists():
        a = read_json(assets / "assets.json")
        require(a["runner_sha256"] == plan["binary_sha256"]
                and a["model_sha256"] == model_hashes(plan)
                and a["checker_sha256"] == plan["checker_sha256"], "Bundle assets differ from plan")
        return
    with tempfile.TemporaryDirectory(prefix=".assets-", dir=bundle) as folder:
        temp = Path(folder) / "assets"
        temp.mkdir()
        checked_copy(source / "runner", temp / "runner", plan["binary_sha256"])
        models = temp / "models"
        models.mkdir()
        for key, name in MODELS.items():
            checked_copy(Path(plan[f"{key}_model"]), models / name, plan[f"{key}_sha256"], clone=True)
        for name in ("LICENSE", "README.md", "provenance.json", "bf16-edit-audit.json"):
            shutil.copy2(Path(plan["baseline_model"]).parent / name, models / name)
        runtime = source / "checker-runtime.txt"
        checked_copy(runtime, temp / runtime.name, plan["checker_sha256"])
        source_site = Path(plan["python"]).parent.parent / "lib/python3.12/site-packages"
        packages = temp / "site-packages"
        packages.mkdir()
        versions = dict(line.split("=", 1) for line in runtime.read_text().splitlines())
        for distribution, module in PACKAGES.items():
            dist = f"{distribution.replace('-', '_')}-{versions[distribution]}.dist-info"
            for name in (module, dist):
                shutil.copytree(source_site / name, packages / name,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        bottles = temp / "bottles"
        bottles.mkdir()
        for pattern in ("*--python@3.12--3.12.14.*.bottle.1.tar.gz",
                        "*--libomp--23.1.5.*.bottle.tar.gz"):
            matches = list((Path.home() / "Library/Caches/Homebrew/downloads").glob(pattern))
            require(len(matches) == 1, f"Missing or ambiguous cached bottle {pattern}")
            checked_copy(matches[0], bottles / matches[0].name.split("--", 1)[1])
        files = {str(path.relative_to(temp)): sha(path)
                 for directory in (packages, bottles) for path in sorted(directory.rglob("*"))
                 if path.is_file()}
        write_json(temp / "assets.json", {
            "runner_sha256": plan["binary_sha256"],
            "checker_sha256": plan["checker_sha256"],
            "model_sha256": model_hashes(plan),
            "dylib_sha256": {str(p): sha(p) for p in DYLIBS},
            "package_and_bottle_sha256": files,
        })
        temp.rename(assets)


def stopped(source):
    require((source / "STOP").is_file(), "Cutover requires STOP on the source")
    pid = int((source / "campaign.pid").read_text())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    raise ValueError("Source worker is still present. Wait for it to leave the Mac lock.")


def copy_case(source, target, path, old_plan_sha, new_plan_sha=None):
    case = read_json(path)
    index = case["index"]
    require(path.name == f"{index:08d}.json" and case["plan_sha256"] == old_plan_sha
            and re.fullmatch(fr"artifacts/case-{index:08d}-attempt-\d+", case["artifacts"]),
            f"Invalid record identity: {path.name}")
    dest_art = target / case["artifacts"]
    dest_art.mkdir()
    for name, field in (("generation.json", "generation_sha256"),
                        ("final-logits.f32", "logits_sha256"), ("output.bin", None)):
        checked_copy(source / case["artifacts"] / name, dest_art / name,
                     case[field] if field else None)
    new_case = target / "cases" / path.name
    if new_plan_sha is None:
        checked_copy(path, new_case)
    else:
        record = path.read_bytes()
        require(record.count(old_plan_sha.encode()) == 1, "Expected exactly one plan binding")
        new_case.write_bytes(record.replace(old_plan_sha.encode(), new_plan_sha.encode()))
    return case


def export(source, bundle, cutover):
    bundle.mkdir(mode=0o700, parents=True, exist_ok=True)
    plan, plan_sha = read_plan(source)
    status = case_status(source)
    require(not status["complete"], "Source campaign is already complete")
    make_assets(source, bundle, plan)
    # C publishes a case by atomic rename, and committed records/artifacts are immutable.
    # Use one contiguous snapshot of published files while the live writer keeps running.
    with (source / "writer.lock").open("rb+" if cutover else "rb") as lock:
        if cutover:
            stopped(source)
            fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        paths = sorted((source / "cases").glob("*.json"))
        require(len(paths) >= 4 and
                [p.name for p in paths] == [f"{i:08d}.json" for i in range(len(paths))],
                "Expected a nonempty contiguous campaign journal")
        name = f"snapshot-{len(paths):08d}" + ("-cutover" if cutover else "")
        target = bundle / name
        require(not target.exists(), f"Snapshot exists: {target}")
        with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=bundle) as folder:
            temp = Path(folder) / name
            (temp / "cases").mkdir(parents=True)
            (temp / "artifacts").mkdir()
            checked_copy(source / "plan.json", temp / "plan.json", plan_sha)
            checked_copy(source / "plan.sha256", temp / "plan.sha256")
            if (source / "continuation.json").is_file():
                checked_copy(source / "continuation.json", temp / "source-continuation.json")
            for ds in plan["datasets"]:
                path = ds["file"]
                require(re.fullmatch(r"data-\d{2}\.jsonl", path), "Invalid frozen dataset path")
                checked_copy(source / path, temp / path, ds["sha256"])
            cases = [copy_case(source, temp, p, plan_sha) for p in paths]
            selected = []
            for d in range(len(plan["datasets"])):
                pairs = [(a, b) for a, b in zip(cases[::2], cases[1::2]) if a["dataset"] == d]
                require(pairs, "Pilot needs one complete pair per dataset")
                # Prefer long-context parity and keep the pilot bounded in practice.
                a, b = min(pairs, key=lambda pair: (
                    -(pair[0]["cutoff"] + pair[1]["cutoff"]),
                    pair[0]["seconds"] + pair[1]["seconds"],
                ))
                require((a["dataset"], a["task"], a["sample"]) ==
                        (b["dataset"], b["task"], b["sample"]), "Misaligned benchmark pair")
                selected.extend((a["index"], b["index"]))
            write_json(temp / "snapshot.json", {
                "source": str(source), "source_plan_sha256": plan_sha, "cutover": cutover,
                "committed": len(paths), "paired": len(paths) // 2, "pilot_cases": sorted(selected),
                "record_sha256": {p.name: sha(p) for p in paths},
                "raw_sha256": {str(c["index"]): sha(temp / c["artifacts"] / "output.bin")
                               for c in cases},
            })
            temp.rename(target)
    print(target)


def checker_identity(executable):
    code = """import hashlib, importlib, importlib.metadata, sys
print('CPython=' + sys.version)
for name in ('math-verify','latex2sympy2-extended','antlr4-python3-runtime','sympy','mpmath'):
    print(name + '=' + importlib.metadata.version(name))
for name in ('parser','grader'):
    module = importlib.import_module('math_verify.' + name)
    print(name + '_sha256=' + hashlib.sha256(open(module.__file__, 'rb').read()).hexdigest())
"""
    return subprocess.check_output([executable, "-I", "-c", code])


def environment(assets, state, expected):
    env = state / "venv"
    require(all(sha(p) == expected["dylib_sha256"][str(p)] for p in DYLIBS),
            "Homebrew CPython/libomp differ from frozen runtime. Check the staged bottles.")
    if not env.exists():
        with tempfile.TemporaryDirectory(prefix=".venv-", dir=state) as folder:
            temp = Path(folder) / "venv"
            subprocess.run([PYTHON, "-I", "-m", "venv", "--without-pip", temp], check=True)
            for name, digest in expected["package_and_bottle_sha256"].items():
                if name.startswith("site-packages/"):
                    require(sha(assets / name) == digest, f"Damaged dependency {name}")
            shutil.copytree(assets / "site-packages", temp / "lib/python3.12/site-packages",
                            dirs_exist_ok=True)
            require(hashlib.sha256(checker_identity(temp / "bin/python")).hexdigest()
                    == expected["checker_sha256"], "Staged math-verify identity differs")
            temp.rename(env)
    py = env / "bin/python"
    require(hashlib.sha256(checker_identity(py)).hexdigest() == expected["checker_sha256"],
            "Staged math-verify identity differs")
    return py


def prepare(bundle, snapshot, state, mode):
    require(re.fullmatch(r"snapshot-\d{8}(-cutover)?", snapshot), "Invalid snapshot name")
    src = bundle / snapshot
    assets = bundle / "assets"
    plan, old_sha = read_plan(src)
    meta = read_json(src / "snapshot.json")
    require(old_sha == meta["source_plan_sha256"], "Snapshot plan identity differs")
    settings = manifest_sha(plan)
    if mode == "run":
        require(meta["cutover"] is True, "Continuation requires a stopped source snapshot")
        receipt = read_json(state / "pilot-pass.json")
        require(receipt["settings_sha256"] == settings and receipt["host"] == host(),
                "Pilot must pass on this Mac with the same frozen plan")
    target = state / mode
    require(not target.exists(), f"Prepared run already exists: {target}")
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    expected = read_json(assets / "assets.json")
    require(expected["runner_sha256"] == plan["binary_sha256"]
            and expected["model_sha256"] == model_hashes(plan)
            and expected["checker_sha256"] == plan["checker_sha256"], "Assets do not match snapshot")
    py = environment(assets, state, expected)
    for key, name in MODELS.items():
        path = assets / "models" / name
        before = stamp(path)
        require(sha(path) == plan[f"{key}_sha256"] and stamp(path) == before,
                f"Wrong or modified model bytes: {name}")
        plan[f"{key}_model"], plan[f"{key}_stamp"] = str(path), before
    plan["python"] = str(py)
    require(manifest_sha(plan) == settings, "Scientific plan changed on relocation")
    with tempfile.TemporaryDirectory(prefix=f".{mode}-", dir=state) as folder:
        temp = Path(folder) / mode
        (temp / "cases").mkdir(parents=True)
        (temp / "artifacts").mkdir()
        checked_copy(assets / "runner", temp / "runner", plan["binary_sha256"])
        checked_copy(assets / "checker-runtime.txt", temp / "checker-runtime.txt", plan["checker_sha256"])
        write_json(temp / "plan.json", plan)
        new_sha = sha(temp / "plan.json")
        (temp / "plan.sha256").write_text(new_sha + "\n")
        for ds in plan["datasets"]:
            require(re.fullmatch(r"data-\d{2}\.jsonl", ds["file"]), "Invalid dataset path")
            checked_copy(src / ds["file"], temp / ds["file"], ds["sha256"])
        chosen = meta["pilot_cases"] if mode == "pilot" else []
        for i in range(meta["committed"]):
            path = src / "cases" / f"{i:08d}.json"
            case = read_json(path)
            require(sha(path) == meta["record_sha256"][path.name], "Snapshot case hash differs")
            require(re.fullmatch(fr"artifacts/case-{i:08d}-attempt-\d+", case["artifacts"]),
                    "Invalid snapshot artifact path")
            raw = src / case["artifacts"] / "output.bin"
            require(sha(raw) == meta["raw_sha256"][str(i)], "Snapshot raw hash differs")
            if i not in chosen:
                copy_case(src, temp, path, old_sha, new_sha)
        write_json(temp / "handoff.json", {
            "dylib_sha256": expected["dylib_sha256"],
            "snapshot": str(src), "settings_sha256": settings, "source_plan_sha256": old_sha,
            "target_plan_sha256": new_sha, "imported": meta["committed"] - len(chosen),
            "pilot_cases": chosen,
        })
        subprocess.run([temp / "runner", "--benchmark-report", temp], check=True)
        require(case_status(temp)["committed_generations"] == meta["committed"] - len(chosen),
                "Imported journal count differs")
        temp.rename(target)
    print(target)


def host():
    return subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.model"], text=True).strip()


def require_ci_lease():
    require(platform.system() == "Darwin" and platform.machine() == "arm64"
            and host() == "Mac18,5", "Benchmark execution is restricted to the Mac mini M6")
    job = os.environ.get("GITHUB_RUN_ID", "")
    tag = os.environ.get("MAC_RUNNER_TAG", "heavy")
    require(re.fullmatch(r"\d+", job) and re.fullmatch(r"[a-z0-9-]+", tag),
            "Run only under the approved Gitea job")
    holder = Path(f"/tmp/macmini-runner-hold-{tag}-{job}.pid")
    require(holder.is_file(), "Missing macmini job-started lock receipt")
    os.kill(int(holder.read_text()), 0)


def compare_pilot(root, info):
    src = Path(info["snapshot"])
    for i in info["pilot_cases"]:
        path = Path("cases") / f"{i:08d}.json"
        before, after = read_json(src / path), read_json(root / path)
        require(all(before[k] == after[k] for k in PARITY_FIELDS), f"Pilot differs for case {i+1}")
        require(sha(src / before["artifacts"] / "output.bin") ==
                sha(root / after["artifacts"] / "output.bin"), f"Pilot raw bytes differ for case {i+1}")


def step(state, mode):
    require_ci_lease()
    root = state / mode
    info = read_json(root / "handoff.json")
    require(all(sha(p) == info["dylib_sha256"][str(p)] for p in DYLIBS),
            "Homebrew CPython/libomp changed after preparation")
    if (root / "STOP").exists():
        return
    if mode == "pilot":
        pending = [i for i in info["pilot_cases"] if not (root / "cases" / f"{i:08d}.json").exists()]
        if pending:
            subprocess.run([root / "runner", "--benchmark-run", root, "--max-cases", "1"], check=True)
        finished = [i for i in info["pilot_cases"] if i not in pending[1:]]
        compare_pilot(root, {**info, "pilot_cases": finished})
        print(f"Pilot parity: {len(finished)}/{len(info['pilot_cases'])} cases verified.", flush=True)
        if len(pending) <= 1:
            write_json(state / "pilot-pass.json", {
                "host": host(), "settings_sha256": info["settings_sha256"],
                "verified_cases": info["pilot_cases"],
            })
            print("Pilot matched token trace, logits, output bytes and math-verify scores.")
    elif not case_status(root)["complete"]:
        receipt = read_json(state / "pilot-pass.json")
        require(receipt["settings_sha256"] == info["settings_sha256"] and receipt["host"] == host(),
                "Missing matching M6 pilot")
        subprocess.run([root / "runner", "--benchmark-run", root, "--max-cases", "1"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest="command", required=True)
    out = cmd.add_parser("export")
    out.add_argument("source", type=Path)
    out.add_argument("bundle", type=Path)
    out.add_argument("--cutover", action="store_true")
    imp = cmd.add_parser("prepare")
    imp.add_argument("bundle", type=Path)
    imp.add_argument("snapshot")
    imp.add_argument("state", type=Path)
    imp.add_argument("mode", choices=("pilot", "run"))
    run = cmd.add_parser("step")
    run.add_argument("state", type=Path)
    run.add_argument("mode", choices=("pilot", "run"))
    args = parser.parse_args()
    try:
        if args.command == "export":
            export(args.source.resolve(), args.bundle.resolve(), args.cutover)
        elif args.command == "prepare":
            prepare(args.bundle.resolve(), args.snapshot, args.state.resolve(), args.mode)
        else:
            step(args.state.resolve(), args.mode)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"handoff: {error}\n")


if __name__ == "__main__":
    main()
