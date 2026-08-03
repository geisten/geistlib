# Documentation ownership

This repository documents the application-neutral Geist engine:

- `QUICKSTART.md`: C-library entry points
- `PI5_BITNET.md`: user guide for the self-contained Pi 5 BitNet binary
- `MODELS.md`: supported models — Gemma (vision/audio), Llama, ternary BitNet
- `BACKENDS.md`: CPU backends and the experimental Metal/Vulkan GPU paths
- `ARCHITECTURE.md`: engine layers and runtime architecture
- `DEPLOY.md`: building the library and consuming the packaged SDK
- `CI_COVERAGE.md`: what each CI job actually verifies
- `API_CONTRACT.md`: what the stability tags promise across a release boundary
- `../benchmark/`: methodology and results

Out of scope, on purpose: tool-use agents, resident daemons, chat templating,
authorization, product UX, app packaging and domain evaluations. Those belong to
whatever links this engine. Documentation here may use a downstream consumer as
an example, but must not carry its setup instructions, product roadmap,
credentials, policy code or release requirements.
