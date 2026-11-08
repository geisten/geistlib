# Use C23 and a public C ABI

geistlib uses C23 to reach existing platform toolchains and embedding languages
through a direct C ABI, with transparent control over kernel code. This
prioritizes interoperability and a small runtime over the memory safety a
language such as Rust could provide; checked sizes and allocation, strict
compiler warnings and sanitizer tests carry that cost.

This records the existing choice, not a new language migration. See
[Architecture](../ARCHITECTURE.md#why-c), the [API contract](../API_CONTRACT.md)
and [coding rules](../../AGENT.md).
