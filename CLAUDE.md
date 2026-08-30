See [AGENT.md](AGENT.md) — the coding rules for this repository, and the file
source comments cite by name. Read it before writing or changing a function.

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
