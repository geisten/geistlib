# Bind compute paths at model load

geistlib resolves supported compute paths and weight placement at load time
to keep per-token execution simple and avoid repeated dtype discovery. This
trades the flexibility of a general runtime graph scheduler for predictable
execution on small CPUs and explicit backend capability failures.

Offload preserves this choice: runtime code selects prepared paths by chunk
size or selected experts, and backend streaming moves weights through prepared
staging resources. CPU and GPU arithmetic can differ, so changing chunk size
can change numerical results within the existing backend tolerances. See
[Architecture](../ARCHITECTURE.md#zero-dispatch-kernel-binding) and
[Offload](../OFFLOAD.md).
