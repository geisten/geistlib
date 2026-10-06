# Documentation

geistlib documents an application-neutral engine: it loads models and
produces tokens.

| Document | What it covers |
| :-- | :-- |
| [QUICKSTART.md](QUICKSTART.md) | Build, generate text, embed the C library |
| [PI5_BITNET.md](PI5_BITNET.md) | The self-contained Pi 5 BitNet binary: install, speed, errors, model limits |
| [MODELS.md](MODELS.md) | Supported model families, downloads, vision and audio towers |
| [BACKENDS.md](BACKENDS.md) | CPU backends, experimental Metal / Vulkan, memory per backend |
| [VOICE.md](VOICE.md) | Push-to-talk and dictation on a Pi 5 and macOS |
| [DEMOS.md](DEMOS.md) | Recorded demos: vs bitnet.cpp, offline box, writing assistant |
| [DEPLOY.md](DEPLOY.md) | Building the library, the packaged SDK, single-file deployment |
| [API_CONTRACT.md](API_CONTRACT.md) | What the stability tags promise across a release |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Engine layers, load-time kernel binding, KV cache, why C |
| [TUNING.md](TUNING.md) | Forward-only (zeroth-order) fine-tuning with per-tensor gains |
| [BITNET_EMBEDDINGS_PLAN.md](BITNET_EMBEDDINGS_PLAN.md) | BitNet embedding models: conversion, API, verification status |
| [DECISION.md](DECISION.md) | Optional numeric candidate scoring (`DECISION=1`) |
| [DECISION_EVALUATION.md](DECISION_EVALUATION.md) | Offline MMLU evaluation protocol for the decision API |
| [DECISION_ENCODERS.md](DECISION_ENCODERS.md) | Python encoder baselines for that evaluation |
| [CI_COVERAGE.md](CI_COVERAGE.md) | What each CI job verifies |
| [CI_SELF_HOSTED.md](CI_SELF_HOSTED.md) | The self-hosted Vulkan GPU runner |
| [RELEASING.md](RELEASING.md) | The guarded release workflow and recovery |
| [../benchmark/](../benchmark/README.md) | Benchmark methodology, raw runs, per-system results |

Contributor rules live at the repository root:
[CONTRIBUTING.md](../CONTRIBUTING.md) and [AGENT.md](../AGENT.md).

## Scope

Out of scope on purpose: tool-use agents, resident daemons, chat templating,
authorization, product UX, app packaging and domain evaluations. They belong
to whatever links the engine. These documents may use a downstream consumer as
an example, but do not carry its setup instructions, roadmap, credentials,
policy code or release requirements.
