# Architecture decisions

These records capture existing choices with lasting trade-offs. They explain
why; [ARCHITECTURE.md](../ARCHITECTURE.md) describes the current implementation.

| Decision | Trade-off |
| :-- | :-- |
| [0001: C23 and a C ABI](0001-c23-runtime-and-c-abi.md) | Embedding reach and control versus memory safety |
| [0002: Application-neutral engine](0002-application-neutral-engine.md) | Reusable inference primitives versus application-specific integration |
| [0003: Bind compute paths at load](0003-bind-compute-paths-at-load.md) | Predictable hot paths versus general runtime scheduling |

Add a record only for a choice that is costly to reverse, surprising without
context and based on real alternatives. Use the next four-digit number and a
short explanation of the context, choice and reason. Add a successor record
when changing a decision; retain the original rationale with a link to it.
