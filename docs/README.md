# Documentation

geistlib is an application-neutral inference engine: it loads models and
provides token generation, embeddings and optional candidate scoring.
Start with the quickstart for usage, the glossary for terminology, or the
architecture decisions for the reasons behind the design.

## Start and integrate

| Document | Read it to… |
| :-- | :-- |
| [Quickstart](QUICKSTART.md) | Build, generate text and embed the C library |
| [Examples](../examples/README.md) | Find runnable C consumers |
| [Foreign-language examples](../examples/ffi/README.md) | Integrate through the C ABI |
| [Deployment](DEPLOY.md) | Link the packaged SDK or ship a single executable |
| [API contract](API_CONTRACT.md) | Understand stability tags and compatibility promises |
| [Pi 5 BitNet](PI5_BITNET.md) | Install the standalone binary and check its limits |

## Models and execution reference

| Document | What it covers |
| :-- | :-- |
| [Models](MODELS.md) | Supported families, downloads, vision and audio towers |
| [Backends](BACKENDS.md) | CPU, Metal and Vulkan support, settings and memory |
| [Offload](OFFLOAD.md) | Current layer planning, prefill routing and expert placement |
| [BitNet embeddings](BITNET_EMBEDDINGS_PLAN.md) | Conversion, pooling, API, verified behavior and remaining limits |
| [Decision scoring](DECISION.md) | Optional numeric candidate scoring (`DECISION=1`) |
| [Tuning](TUNING.md) | Forward-only fine-tuning with per-tensor gains |

The embeddings document keeps its historical filename to preserve existing
links; it documents implemented functionality as well as open work.

## Domain language and architecture decisions

| Document | Purpose |
| :-- | :-- |
| [Glossary](../GLOSSARY.md) | Canonical domain terms without implementation detail |
| [Architecture](ARCHITECTURE.md) | Code map, layers, model families, sessions and caches |
| [Architecture decisions](adr/README.md) | C23/C ABI, engine scope and load-time binding trade-offs |
| [Roadmap](../ROADMAP.md) | Project direction and work still to do |

The glossary defines terms; ADRs explain lasting choices. Usage instructions,
API guarantees and measurements remain in their respective references.

## Development and operations

| Document | What it covers |
| :-- | :-- |
| [Contributing](../CONTRIBUTING.md) | Build, tests, benchmarks and contribution workflow |
| [Coding rules](../AGENT.md) | C contracts, allocation, bounds, errors and checks |
| [Tests](../tests/README.md) | Test tiers, preconditions and exit codes |
| [Test scripts](../tests/scripts/README.md) | Script-based test entry points |
| [CI coverage](CI_COVERAGE.md) | Required and post-merge checks |
| [Hardware CI](HARDWARE_CI.md) | Self-hosted Pi 5 and GPU runner setup |
| [Toolchain probe](TOOLCHAIN_PROBE.md) | C23 features verified on the CI toolchains |
| [Releasing](RELEASING.md) | Release workflow and recovery |
| [Release fragments](../changes/README.md) | Format for unreleased changes |
| [Changelog](../CHANGELOG.md) | Released changes |
| [Security](../SECURITY.md) | Vulnerability reporting |
| [Code of conduct](../CODE_OF_CONDUCT.md) | Community expectations |

## Evaluation, measurements and demonstrations

| Document | What it covers |
| :-- | :-- |
| [Benchmarks](../benchmark/README.md) | Protocols, tools and per-system results |
| [Benchmark methodology](../benchmark/METHODOLOGY.md) | Measurement and parity requirements |
| [Offload experiments](../benchmark/results/OFFLOAD.md) | Recorded phase 0–3, expert-staging and MTP experiments |
| [C23 review](C23_REVIEW.md) | Historical #333 results, rejected approaches and follow-up status |
| [Decision evaluation](DECISION_EVALUATION.md) | Offline MMLU protocol for candidate scoring |
| [Encoder baselines](DECISION_ENCODERS.md) | Optional Python comparison adapters and their limits |
| [Voice](VOICE.md) | Push-to-talk and dictation examples on Pi 5 and macOS |
| [Demos](DEMOS.md) | Recorded application demonstrations |
| [Demo tooling](../tools/demos/README.md) | Scripts for recording demonstrations |

Historical reports retain the measured revisions, configurations and caveats.
They are evidence, not statements of current defaults. Raw benchmark runs and
release fragments remain alongside their reports and release workflow.

## Scope

Tool-use agents, resident daemons, chat templating, authorization and product
UX belong to consumers of the engine. Deployment recipes, demonstrations and
offline model-evaluation harnesses document how to use or verify the library;
they do not add application policy to its runtime. See
[ADR 0002](adr/0002-application-neutral-engine.md).
