# geist Architecture

geist is a C23 inference runtime built around one idea: **decide everything
expensive at load time, so the hot path is branch-light.** This document maps
the codebase and the design rationale referenced from
[`include/geist.h`](../include/geist.h).

## Three layers

```
include/geist.h          public C ABI (the only supported surface)
  │
src/engine/              orchestration: model load, sessions, sampler,
  │                      tokenizers, allocators, backend/arch registries
src/archs/               architectures: how a model's forward pass is wired
  │   transformer/         Gemma 4 family (RoPE, GQA, KV cache, PLE, head)
  │   audio_conformer/     Conformer audio tower (mel → encoder → soft tokens)
  │   vision_siglip/       SigLIP vision tower (image/video → soft tokens)
src/backends/            compute: the kernels that actually run the ops
      common/              shared compute library (GEMM facade, gemma4 kernels, KIVI)
      cpu_neon/            Apple Silicon + ARM64, OpenMP-parallel
      cpu_scalar/          portable reference (correctness oracle)
      cpu_x86/             x86-64 AVX-512 / VNNI (opt-in; runtime hw_probe dispatch)
      metal/               Apple GPU (opt-in, experimental; dlopen'd Metal.framework)
      vulkan/              Linux/NVIDIA GPU (opt-in, experimental; dlopen'd libvulkan)
src/formats/             GGUF + PTQTP quant (de)quantization
src/io/                  GGUF reader, safetensors reader
src/base/                freestanding utilities: heap, error, hw_probe
```

An **architecture** knows the shape of the computation (which ops, in which
order, with which tensors). A **backend** knows how to execute an op on a given
dtype/layout. The engine binds the two and drives sessions. Backends and archs
are listed in compile-gated registries (`src/engine/*_registry.c`, each entry
behind a `GEIST_BACKEND_*` / `GEIST_ARCH_*` guard), so the set compiled in is a
build-time choice (`make BACKENDS="..."`) and the one used is a runtime choice
(`geist_backend_create("auto" | "cpu_neon" | ...)`).

### Dependency rules

The include graph follows the layers strictly — these rules are load-bearing,
not aspirational (the tree conforms today):

- `src/base/` includes nothing above it; everyone may include `src/base/`.
- `src/engine/` includes `base` and the public headers. It never reaches into
  an arch or backend implementation — with exactly **two composition points**:
  `arch_registry.c` and `backend_registry.c`, the only files that name concrete
  archs/backends (each entry compile-gated).
- `src/archs/` never includes a concrete backend (`cpu_*`, `metal`, `vulkan`).
  Ops reach
  compute through the backend vtbl. The one sanctioned shortcut is
  `src/backends/common/` — the shared compute library (GEMM facade, gemma4
  kernels, KIVI) that is always compiled and may be called directly from archs.
- `src/backends/<name>/` includes `base` and `backends/common` only; a backend
  never includes another backend or an arch.
- `src/formats/` and `src/io/` sit beside the backends: `base` only.

A new backend or arch therefore touches its own directory, one `mk/backend-*.mk`
fragment, and one registry entry — nothing else.

## Processing pipeline

End-to-end flow of a request, from loading a model to emitting tokens. Load-time
work (parse, kernel binding) happens once; the decode loop is the per-token hot
path. The dense-fp32 `geist_sgemm` node is where the build-time `GEMM_PROVIDER`
(accelerate / openblas / native) plugs in; quantized weights bypass it entirely
via the kernel bound at load time.

```mermaid
flowchart TD
    subgraph LOAD["Load time — decide everything expensive once"]
        A["geist_model_load(path)<br/>GGUF parse · io/"] --> B["arch_registry_lookup<br/>match general.architecture"]
        B --> C["decoder_ops.state_create<br/>load weights · KV layout"]
        C --> D["resolve_weight per tensor<br/>bind (op,dtype,layout) → kernel ptr<br/>backends/*/weight_resolve.c"]
    end

    D --> E["geist_backend_create<br/>auto | cpu_neon | cpu_scalar"]
    E --> F["geist_session_create<br/>KV cache · sampler · scratch"]

    F --> G{"input modality"}
    G -->|text| H["geist_session_set_prompt<br/>tokenize"]
    G -->|audio / image / video| I["geist_session_attach_*<br/>tower: mel/patch → encoder → projector<br/>archs/audio_conformer · vision_siglip"]
    I --> J["soft tokens (embeddings)"]
    H --> K["prefill: append tokens to KV cache"]
    J --> K

    K --> L{{"decode step (loop)"}}
    L --> M["per layer: RMSNorm → attention<br/>RoPE · GQA · KV append · fused QK^T·softmax·V"]
    M --> N["residual → RMSNorm → FFN<br/>gate/up linear · SiLU/GELU · down linear"]
    N --> O{"linear weight dtype"}
    O -->|quantized| P["NEON W*A8 kernel<br/>cpu_neon/kernels/*"]
    O -->|dense fp32| Q["geist_sgemm<br/>GEMM_PROVIDER: accelerate | openblas | native"]
    P --> R["head → logits"]
    Q --> R
    R --> S["sampler: greedy / top-k / top-p"]
    S --> T["geist_session_token_to_str → emit"]
    T -->|"not EOS and under limit"| L
    T -->|"EOS or limit"| U(["done"])
```

Speculative decode (`geist_session_decode_speculative`) replaces the single
`decode step` with an n-gram draft + one batched `verify_forward`; on a miss it
falls back to the sequential path shown above.

## Above the ABI: CLI & agent

The CLI (`geist` with its `agent` / `chat` subcommands), the memory palace, and
the tool-use agent are **consumers** of the public ABI, not part of the runtime.
They live in `tools/` as header-only modules (`mind.h`, `agent.h`, `agent_*.h`,
`agent_main.h`) so the same code runs in-process on desktop and inside an embedded
host (iOS/Android) with no link step. The agent is the **same process** as the model — `geist_agent_run` is an
in-process call over a resident session. The host, not the model, decides what
runs: a request can act only through an immutable **tool whitelist** — compiled
once for legacy tools or supplied dynamically per request — bounded by a global
call budget. See [agent.md](agent.md).

```mermaid
flowchart TD
    REQ["input + optional dynamic tools"] --> COMPILE["strict parse · copy immutable toolset<br/>compile Schema-v1"]
    COMPILE --> SYS["geist_agent_run<br/>system prompt + offered menu"]
    SYS --> GEN["generate one turn<br/>(decode loop, above)"]
    GEN --> PARSE{"emitted a<br/>tool call?"}
    PARSE -->|no| ANS(["plain text = final answer"])
    PARSE -->|yes| GATE{"name ∈ whitelist?"}
    GATE -->|no| REJ["error observation<br/>(tool never runs)"]
    GATE -->|yes| SCHEMA{"args match selected schema?"}
    SCHEMA -->|no| REJ
    SCHEMA -->|yes, static| RUN["in-process tool.invoke<br/>doc_search · web_fetch · …"]
    SCHEMA -->|yes, dynamic| CALL["emit correlated tool.call"]
    CALL --> HOST["host revalidates policy<br/>and executes"]
    HOST --> RESULT["tool.result · cancel · retryable"]
    RESULT --> OBS
    REJ --> OBS["append observation"]
    RUN --> OBS
    OBS -->|"step < max_steps"| GEN
    OBS -->|"step = max_steps"| ANS

    GEN -.->|"geist_session_set_prompt / decode_step"| ABI["public ABI (include/geist.h)"]
```

The generate node reuses the decode pipeline shown above; only orchestration
(compile → route → parse → gate → validate → call/result → observe) is in the
agent/server layer. `json_schema_v1.h`, `dynamic_tools_v1.h` and
`dynamic_request_v1.h` contain no HA dependency. `dynamic_host_v1.h` adapts the
same callback boundary to newline-framed host round trips. Tools remain
platform-specific, which is why none of this expands `libgeist`'s public ABI.

**Driving tools on small, non-tool-trained models — without touching the
sampler.** A 2 B model rarely emits a clean tool call on its own, and geist's
ABI deliberately exposes no in-engine grammar mask. The agent gets the same
guarantee *from outside* the engine, using only `peek_logits` / `prefill_tokens`
/ `tokenize`:

- **Routing** scores each tool *name* as the answer to a "which tool?" cloze and
  PMI-calibrates it against a content-free baseline (so a frequent token like
  `list` doesn't always win) — picking the tool, not trusting a raw logit.
- **Forcing** prefills the `{"tool":"<name>","args":{…}}` scaffold. Legacy
  one-string tools lift their value from the request. Dynamic tools build typed
  objects from the schema: required/optional fields, numbers, booleans, scalar
  enums and enum arrays. The complete object must validate before RUNNING.
- **Confidence gating** compares the routed winner with the runner-up and the
  direct-reply pseudo-entry. A close race returns a clarification, not a guessed
  action.

This is the architectural reason the agent layer is **above** the ABI, not inside
it: a constrained-decoding capability that would normally live in the sampler is
reconstructed from the public peek/prefill primitives, keeping `libgeist` small
and the grammar logic auditable in `agent.h`. See [agent.md](agent.md#tool-selection--forced-calls).

### Dynamic host boundary

A dynamic request may offer up to 16 tools and 1–16 total calls. The offered
array is copied into request-owned storage; changing sender buffers cannot widen
capability after compilation. Names must be unique. The documented v1 schema
subset supports objects, strings, numbers, integers, booleans, arrays, required
and optional properties, scalar enums, bounds and nested structures.
Unsupported keywords fail compilation.

Dynamic actions never execute inside Geist. The server emits a newline-framed
`tool.call` with a fresh `call_id`; the host rechecks name, schema and application
policy, executes, and returns `tool.result`. One explicitly retryable result may
repeat the call with a new id and consumes the same global budget. A correlated
cancel stops later calls. Home Assistant is one adapter;
`examples/dynamic_tools_host.c` is an independent C adapter and build.

The serving surface also accepts the exact model-free health control frame
`{"type":"health"}` and returns protocol identity plus readiness. Configuration
clients use it to reject the wrong socket before saving state. It does not add a
second transport or invoke the model.

## Zero-dispatch kernel binding

Generic engines walk a layer-dispatch loop every token, switching on dtype and
op. geist instead resolves, **at model-load time**, a direct function pointer
for each tensor's `(op, dtype, layout)` combination
(`src/backends/*/weight_resolve.c`, `kernel_catalog.c`). The decode loop then
calls bound kernels with no vtable walk or format switch. This matters most on
single-core-heavy edge CPUs where dispatch overhead is a real fraction of the
per-token budget.

Capabilities are queryable up front via `geist_backend_supports_op` returning
`NONE` / `EMULATED` / `NATIVE`, so an arch can pick the best available path or
fail cleanly rather than discovering an unsupported combination mid-forward.

## Tensors: dtype vs layout

A `geist_tensor` separates the **logical** dtype (`F32`, `Q4_K`, `TQ2_0`, …)
from the **physical** layout (`DENSE`, `BLOCK_QUANTIZED`, `TERNARY_BITPLANE`,
…). Block-quantized formats carry a `geist_quant_desc` with bits-per-value as an
exact rational (`158/100` for 1.58-bit), block size, and scale/zero offsets.
This is what lets ternary BitNet be a first-class citizen rather than a bolt-on:
the kernel for a `TERNARY` weight does only adds/subtracts, no multiplies.

## Op vocabulary

The backend op set (`enum geist_op`) is deliberately small: `LINEAR`,
`RMSNORM`, `RESIDUAL_ADD`, `SILU_GATE`, `EMBEDDING_LOOKUP`, `ATTENTION`,
`ROPE`, plus reserved SSM ops (`SSM_STEP`/`SSM_SCAN`/`CONV1D`) for a future
Mamba arch. Fused attention (QKᵀ → softmax → V) is one op so the backend can
keep the score matrix in registers/L1.

## Sessions and the KV cache

A `geist_model` is immutable, shared, read-only weights. A `geist_session` owns
the mutable per-conversation state: KV cache, pending logits, sampler config,
stats. Multiple sessions can share one model. The KV cache supports quantized
modes (`INT8`, packed `INT4` — half the INT8 footprint, near-lossless via a
default-on Hadamard rotation, `GEIST_KV_ROT=0` to opt out; and `KIVI` 2-bit)
and prefix pinning
(`geist_session_pin_prefix`) to
amortize a constant system prompt across chat turns. Speculative decode drafts
via an n-gram lookup over history and verifies in one batched forward.

The rotation and packed-INT4 modes store K post-RoPE and rotated; this is only
safe because geist never re-bases cached positions (the sliding window masks,
it does not re-RoPE the cache). A future KV context-shift feature would have to
un-rotate/unpack before re-RoPE-ing — see the note atop `forward/kv_store.c`.

## Multimodal: soft-token prefixes

Instead of a "Whisper → text → LLM" cascade, the audio/vision towers produce
embedding **soft tokens** that are prefixed directly into the LM's KV cache
(`attach_audio` / `attach_image` / `attach_video`). The LM attends to the
modality embeddings directly, cutting latency and preserving information that a
text bottleneck would discard.

## Where to start reading

- The forward pass: `src/archs/transformer/forward/step.c`.
- Kernel binding: `src/backends/cpu_neon/weight_resolve.c`.
- A representative low-bit kernel: `src/backends/cpu_neon/kernels/q4_K.c`.
- The public contract: `include/geist.h` (stability tags per symbol).
- The tool-use SDK interface (`agent.h`, `--serve`, dynamic-tools-v1; concrete
  tools live in consumer repos): [agent.md](agent.md).
- Building self-contained binaries and deploying: [DEPLOY.md](DEPLOY.md).

Per directory, the file to open first:

| Directory | Start here | It owns |
| :-- | :-- | :-- |
| `src/base/` | `heap.h` | arenas/allocators, error codes, hw probe |
| `src/engine/` | `session.c` | load → session → decode loop, tokenizers, registries |
| `src/archs/transformer/` | `forward/step.c` | the per-token forward pass |
| `src/archs/audio_conformer/` | `audio_encoder.c` | mel → Conformer → soft tokens |
| `src/archs/vision_siglip/` | `vision_encoder.c` | image/video → SigLIP → soft tokens |
| `src/backends/common/` | `geist_gemm.c` | shared GEMM facade + gemma4 kernels |
| `src/backends/cpu_neon/` | `weight_resolve.c` | load-time kernel binding, NEON kernels |
| `src/backends/cpu_x86/` | `backend.c` | AVX-512/VNNI kernels, runtime dispatch |
| `src/backends/cpu_scalar/` | `backend.c` | the portable correctness oracle |
| `src/backends/metal/` | `backend.c` | Apple-GPU path (shaders in `metal_shaders.h`) |
| `src/backends/vulkan/` | `backend.c` | Linux/NVIDIA-GPU path (SPIR-V in `shaders/`) |
| `src/formats/gguf/` | `common.c` | per-quant decode (one file per format) |
| `src/io/` | `gguf_reader.c` | GGUF/safetensors file parsing |
| `tools/` | `geist.c` | the demo CLI and the header-only SDK |
