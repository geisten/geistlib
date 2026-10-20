# Changelog fragments

A pull request does not edit `CHANGELOG.md`. It adds one file here per
change, `changes/<slug>.md`; the slug is free-form (the issue number and a
few words, or the branch name: `123-q4k-tail-fix.md`). The release pull
request folds every fragment into `CHANGELOG.md` and deletes them, so two
open pull requests never conflict over the changelog.

The first line names the section, one of `Added`, `Changed`, `Deprecated`,
`Removed`, `Fixed`, `Security`. The rest is the entry exactly as it will
appear in `CHANGELOG.md`: a list item with a bold lead sentence and the issue
reference, then what changed, why, and the evidence.

```markdown
### Fixed
- **cpu_neon kernels no longer leak their scratch with every thread that
  ends (#100).** The q4_K/q6_K/TL1 kernels' `_Thread_local` high-water
  buffers grow through `heap_tl_grow`, which frees them at thread exit; ...
```

One section per fragment; a change that both adds and fixes something gets
two fragments. Edit your fragment freely until the pull request merges.

```sh
python3 scripts/changelog.py check                     # CI runs this
python3 scripts/changelog.py release 0.22.0 --dry-run  # the CHANGELOG diff a release would make
```
