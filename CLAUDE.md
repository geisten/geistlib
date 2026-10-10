See [AGENT.md](AGENT.md) — the coding rules for this repository. Read it before writing or changing a function.

Fastest orientation:

- **Parameter order** — lengths before the arrays they describe, so
  `T arr[static len]` is expressible. AGENT.md §1 has the full order and the
  rule for when `[static len]` would be a lie.
- **Allocation** through `src/base/heap.h`; checked size arithmetic through
  `src/base/checked.h`. AGENT.md §3.
- **Untrusted lengths** are checked by subtraction before the pointer moves.
  AGENT.md §4.
- **Changing an existing API** happens in bounded batches with a disassembly
  and benchmark gate. AGENT.md §6.

Build, test and benchmark commands: `CONTRIBUTING.md`.
Architecture: `docs/ARCHITECTURE.md`. API promises: `docs/API_CONTRACT.md`.

**Where the repository lives.** Development, CI and releases are on
`git.geisten.net/geisten/geistlib` (Gitea, private). `github.com/geisten/geistlib`
is a push mirror of it: anything pushed to GitHub directly is overwritten on
the next mirror sync, so never push there. Push branches to the Gitea remote
and open PRs with `tea pulls create --login git.geisten.net --repo
geisten/geistlib`; merge with a merge commit. CI is `.gitea/workflows/`
(`.github/workflows/` only serves pull requests on the mirror). A PR opened on
GitHub by someone else comes over with `scripts/import-github-pr.sh <n>`
after reading it. Releases: `docs/RELEASING.md`.
