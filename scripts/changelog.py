#!/usr/bin/env python3
"""changelog.py — changelog fragments under changes/, assembled at release.

Every pull request used to edit CHANGELOG.md under [Unreleased], so any two
open pull requests conflicted there and each merge forced a rebase and a full
CI re-run on the rest. A change now adds one file, changes/<slug>.md, and the
release pull request folds them all into CHANGELOG.md.

Fragment format (changes/README.md):

    ### Fixed
    - **Bold lead sentence (#123).** Body in the CHANGELOG's style.

The first line names the section; the rest is the entry, verbatim.

Usage:
    python3 scripts/changelog.py check
        validate every fragment (CI, version-consistency job)
    python3 scripts/changelog.py release VERSION [--date YYYY-MM-DD] [--dry-run]
        move [Unreleased] entries and every fragment into a dated
        '## [VERSION] — DATE' section, update the comparison links and delete
        the fragments; --dry-run prints the CHANGELOG diff and changes nothing
"""

import argparse
import datetime
import difflib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHANGES = ROOT / "changes"
CHANGELOG = ROOT / "CHANGELOG.md"
# Keep a Changelog order; the CHANGELOG has always used it.
SECTIONS = ("Added", "Changed", "Deprecated", "Removed", "Fixed", "Security")
REPO = "https://github.com/geisten/geistlib"


def fragments():
    return sorted(p for p in CHANGES.glob("*.md") if p.name != "README.md")


def parse_fragment(path):
    """Return (section, body) or raise ValueError naming the problem."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("empty")
    m = re.fullmatch(r"### (\S+)\s*", lines[0])
    if not m or m.group(1) not in SECTIONS:
        raise ValueError(f"first line must be '### <{'|'.join(SECTIONS)}>', got {lines[0]!r}")
    body = "\n".join(lines[1:]).strip("\n")
    if not body.lstrip().startswith("- "):
        raise ValueError("entry must start with a '- ' list item")
    if re.search(r"^#", body, re.M):
        raise ValueError("one section per fragment: no headings in the entry")
    return m.group(1), body


def check():
    bad = 0
    for p in fragments():
        try:
            parse_fragment(p)
        except ValueError as e:
            print(f"changelog: {p.relative_to(ROOT)}: {e}", file=sys.stderr)
            bad += 1
    if bad:
        print("changelog: see changes/README.md for the fragment format", file=sys.stderr)
        return 1
    print(f"changelog: {len(fragments())} fragment(s) OK")
    return 0


def section_rank(name):
    # "Changed (BREAKING)" sorts with "Changed"; unknown names go last.
    word = name.split()[0]
    return SECTIONS.index(word) if word in SECTIONS else len(SECTIONS)


def release(version, date, dry_run):
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        sys.exit(f"changelog: version must be X.Y.Z, got {version!r}")
    old = CHANGELOG.read_text(encoding="utf-8")
    lines = old.split("\n")
    try:
        start = lines.index("## [Unreleased]")
    except ValueError:
        sys.exit("changelog: CHANGELOG.md has no '## [Unreleased]' heading")
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("## "))

    # Entries still written into [Unreleased] directly (pull requests opened
    # before the fragments) come first in their section.
    entries = {}
    order = []
    current = None
    for line in lines[start + 1 : end]:
        m = re.fullmatch(r"### (.+?)\s*", line)
        if m:
            current = m.group(1)
            if current not in entries:
                entries[current] = []
                order.append(current)
            continue
        if line.strip() and current is None:
            sys.exit(f"changelog: [Unreleased] text outside a '### ' section: {line!r}")
        if current is not None:
            entries[current].append(line)
    for name in order:
        entries[name] = ["\n".join(entries[name]).strip("\n")]
        if not entries[name][0]:
            entries[name] = []

    paths = fragments()
    for p in paths:
        try:
            name, body = parse_fragment(p)
        except ValueError as e:
            sys.exit(f"changelog: {p.relative_to(ROOT)}: {e}")
        if name not in entries:
            entries[name] = []
            order.append(name)
        entries[name].append(body)

    names = sorted((n for n in order if entries[n]), key=section_rank)
    if not names:
        sys.exit("changelog: nothing to release — no fragments and [Unreleased] is empty")
    block = ["## [Unreleased]", "", f"## [{version}] — {date}", ""]
    for n in names:
        block += [f"### {n}", "\n".join(entries[n]), ""]
    lines[start:end] = "\n".join(block).split("\n")

    # [Unreleased]: .../compare/vPREV...HEAD  ->  vVERSION...HEAD, plus
    # [VERSION]: .../compare/vPREV...vVERSION right below it.
    link_re = re.compile(r"\[Unreleased\]: " + re.escape(REPO) + r"/compare/v([0-9.]+)\.\.\.HEAD")
    idx = next((i for i, l in enumerate(lines) if link_re.fullmatch(l)), None)
    if idx is None:
        sys.exit("changelog: no '[Unreleased]: .../compare/vX.Y.Z...HEAD' link")
    prev = link_re.fullmatch(lines[idx]).group(1)
    lines[idx : idx + 1] = [
        f"[Unreleased]: {REPO}/compare/v{version}...HEAD",
        f"[{version}]: {REPO}/compare/v{prev}...v{version}",
    ]
    new = "\n".join(lines)

    if dry_run:
        sys.stdout.writelines(
            difflib.unified_diff(
                old.splitlines(True), new.splitlines(True), "a/CHANGELOG.md", "b/CHANGELOG.md"
            )
        )
        for p in paths:
            print(f"would delete {p.relative_to(ROOT)}", file=sys.stderr)
        return 0
    CHANGELOG.write_text(new, encoding="utf-8")
    for p in paths:
        p.unlink()
    print(f"changelog: [{version}] — {date}: {len(paths)} fragment(s) folded in and deleted")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    r = sub.add_parser("release")
    r.add_argument("version")
    r.add_argument("--date", default=datetime.date.today().isoformat())
    r.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "check":
        return check()
    return release(a.version, a.date, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
