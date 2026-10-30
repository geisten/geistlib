#!/usr/bin/env python3
# .github/workflows/ci.yml -> .gitea/workflows/ci.yml for the geisten org's runners.
# Run from the repository root after changing .github/workflows/ci.yml; edit
# that file, not the output. --check writes nothing and exits 1, with a diff,
# when .gitea/workflows/ci.yml is not what this script makes of it (CI runs it,
# #272). A Gitea-only difference belongs in this script.
import difflib,re,sys
s=open('.github/workflows/ci.yml').read()
def swap(s,old,new,what):
    assert s.count(old)==1, what
    return s.replace(old,new,1)
# GitHub runs this for pull requests only; on Gitea it gates main as well
gh_on=s[s.index("# This repository on GitHub is the public mirror"):s.index("\n\n",s.index("\non:\n"))]
# Nightly as well: the post-merge tier (docs/CI_COVERAGE.md "Tiers") then
# also sees a changed runner or toolchain on a quiet day. 03:47 UTC sits
# between native-avx512 (02:40) and release-state (06:17).
s=s.replace(gh_on,"""on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: "47 3 * * *"
  workflow_dispatch:""",1)
# A newer main push cancels the older main run on Gitea; GitHub has no pushes
s=swap(s,"""# Cancel superseded runs on the same PR. A push to main runs to the end:
# merges land faster than the 65 min coverage job, and with cancellation
# most main commits were never fully measured (perf record, coverage).
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}
""","""# A newer push to the same PR or to main cancels the older run: only the
# newest main is fully tested. Uncancelled, main runs queued behind each
# other's ~2 h coverage job and blocked every runner (2026-10-21: four main
# runs in line, the newest waiting 1.5 h). The cost: a failure after a burst
# of merges names the burst, not the merge. Nightly and dispatched runs have
# their own group and are never cancelled by a merge.
concurrency:
  group: ci-${{ github.ref }}-${{ github.event_name == 'schedule' && 'nightly' || github.event_name == 'workflow_dispatch' && 'dispatch' || 'push' }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' || github.event_name == 'push' }}
""","concurrency")
s,n=re.subn(r"\n( +)# Intel Mac: target-mac\*\.mk[^\n]*\n(?:\1#[^\n]*\n)*\1- name: macOS x86_64\n(?:\1  .*\n)+","\n",s)
assert n==2, "intel entries"
assert "macos-15-intel" not in s and "vulkan-gpu:" not in s
# The M1 Max has two runners (docs/HARDWARE_CI.md): macos-15 = mac-light (no
# lock, nice 10) for build + unit legs, macos-15-heavy = mac-m1max (holds the
# Mac's benchmark lock) for legs that load real models or measure time: the
# post-merge macOS arm64 leg (Metal parity, qwen3.5 integration, perf gate).
# Likewise the post-merge Linux arm64 leg (models, perf floor) runs on the Pi 5
# (ubuntu-24.04-arm-pi, under the board lock), not the shared cloud VM.
for title,runner,dedicated in [("models, Metal, ASan + perf (macOS arm64)","macos-15","macos-15-heavy"),
                               ("models + perf (Linux arm64)","ubuntu-24.04-arm","ubuntu-24.04-arm-pi")]:
    old=f"            title: {title}\n            runner: {runner}\n"
    s=swap(s,old,old.replace(runner,dedicated)+"            heavy: true\n",title)
# The heavy legs skip main pushes (they held their host lock 40+ min per merge)
s=swap(s,"""  # 20 tok/s, run 1586).
  build-test-full:
    name: ${{ matrix.title }}
    needs: changes
    if: needs.changes.outputs.code == 'true' && needs.changes.outputs.full == 'true'
""","""  # 20 tok/s, run 1586). Those two legs (heavy: true) skip main pushes: each
  # held its host lock for 40+ minutes after every merge and queued every
  # measurement behind it. They run nightly, on dispatch and with ci:full.
  build-test-full:
    name: ${{ matrix.title }}
    needs: changes
    if: needs.changes.outputs.code == 'true' && needs.changes.outputs.full == 'true' && (github.event_name != 'push' || matrix.heavy != true)
""","build-test-full")
# Coverage: not on main pushes either (desktop-ci's slots), nightly instead
s=swap(s,"""    # Not on pull_request: the instrumented run takes 60-100 minutes. It runs
    # on every push to main, nightly and on dispatch, so a coverage drop is
    # caught one merge later; a pull request opts in with the ci:full label.
    needs: changes
    if: needs.changes.outputs.code == 'true' && needs.changes.outputs.full == 'true' && (github.event_name != 'pull_request' || needs.changes.outputs.label == 'true')
""","""    # Nightly, on dispatch and for a pull request labelled ci:full; not on
    # main pushes. The instrumented run takes ~135-150 min of one of desktop-ci's
    # four slots, and with one per merge it was the biggest CI bottleneck
    # (2026-10-21: three at once, 74 % runner load). A coverage drop is
    # caught by the next nightly run.
    needs: changes
    if: needs.changes.outputs.code == 'true' && needs.changes.outputs.full == 'true' && github.event_name != 'push' && (github.event_name != 'pull_request' || needs.changes.outputs.label == 'true')
""","coverage")
a=s.index("  format:\n"); s=s[:a]+s[a:].replace("    runs-on: ubuntu-24.04-arm\n","    runs-on: ubuntu-24.04\n",1)
a=s.index("  tsan:\n")
s=s[:a]+s[a:].replace("    runs-on: ubuntu-24.04\n","""    # desktop-tsan: its containers may call personality(ADDR_NO_RANDOMIZE), so
    # setarch -R runs the binaries without ASLR — gcc-14's TSan aborts on the
    # host's 32-bit mmap entropy ("unexpected memory mapping")
    runs-on: ubuntu-24.04-tsan
""",1)
m0=s.index("Multi-session tests under TSan",a); e=s.index("\n\n",m0)
seg=s[m0:e]
for t in ["bin/linux/tsan/tests/test_multi_session_parallel_int","bin/linux/tsan/tests/test_session_cancel_unit","bin/linux/tsan/tests/test_par_for_unit"]:
    seg=seg.replace(t,'setarch "$(uname -m)" -R '+t,1)
assert seg.count("setarch")==3
s=s[:m0]+seg+s[e:]
hdr="""# GENERATED from .github/workflows/ci.yml by scripts/gitea-ci.py: edit that
# file and rerun the script; CI's `gitea-ci.py --check` fails when they differ.
# Runners of the geisten org: ubuntu-24.04 = desktop (docker, 4 parallel),
# ubuntu-24.04-tsan = desktop (docker, see the tsan job),
# ubuntu-24.04-arm = the Hetzner cloud VM (docker, nothing timed),
# ubuntu-24.04-arm-pi = Pi 5 (docker, one job at a time under the board
# lock), macos-15 = the M1 Max's light runner (host, no lock: build + unit),
# macos-15-heavy = the M1 Max under its benchmark lock (real models, timing).
# No Intel Mac leg; the hardware tier is in its own workflows here
# (docs/HARDWARE_CI.md). Pull requests run the required tier only; see the
# comment above the `changes` job.
"""
out='.gitea/workflows/ci.yml'
if sys.argv[1:]==['--check']:
    cur=open(out).read()
    if cur!=hdr+s:
        sys.stdout.writelines(difflib.unified_diff(cur.splitlines(True),(hdr+s).splitlines(True),out,'scripts/gitea-ci.py'))
        sys.exit(f"{out} differs from what scripts/gitea-ci.py makes of .github/workflows/ci.yml (above: - is the file, + the script's): edit .github/workflows/ci.yml (or, for a Gitea-only difference, the script) and run python3 scripts/gitea-ci.py")
elif sys.argv[1:]:
    sys.exit("usage: scripts/gitea-ci.py [--check]")
else:
    open(out,'w').write(hdr+s)
