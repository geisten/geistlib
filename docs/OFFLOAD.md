# Offload

Offload chooses between compute paths prepared at model load. Layer splitting,
Vulkan weight streaming and MoE expert placement are implemented; the GPU
backends and split API remain experimental. The design constraint is recorded
in [ADR 0003](adr/0003-bind-compute-paths-at-load.md).

## Dense layer split

Set `geist_session_opts.split_backend` and `split_layers` when loading through
`geist_model_load_with_opts`. The first `split_layers` layers run on the split
backend; the embedding, remaining layers and head stay on the load backend.
Both backends must outlive the model and its sessions. A null split backend or
zero layers disables the split.

`GEIST_SPLIT_AUTO` asks `plan_split_layers` in
`src/archs/transformer/arch_state.c` to choose how many leading layers fit.
It accounts for layer weights, KV storage at `max_seq_len` (4096 if unset),
and a reserve of one sixteenth of device memory plus 512 MiB. Unified memory
can hold all layers in the split stage. This is a dense-layer fit planner,
not a general per-weight, calibrated placement optimizer.

### Prefill offload

For a split model on a discrete GPU, the loader can also bind GPU copies
(**twins**) of the CPU layers. Large prefill chunks then run every layer on
the GPU, while decode and smaller chunks retain the original split.

`GEIST_OFFLOAD_M_MIN` controls the minimum prefill rows:

- Unset: 64 when the split leaves layers on the load backend, the split
  backend is discrete, and the twin matrices fit in available host memory
  with a fifth to spare; otherwise disabled.
- `0`: disable prefill offload.
- Positive integer: override the threshold. Session setup can still fall
  back to the plain split if the twin session does not fit.

The twin cache must be current before a chunk can use it. New K/V rows from
GPU prefill are appended through the home cache's own quantization and
rotation; CPU steps mirror rows into the twin cache where it is current.
The session owns the twin cache and transfer buffers. See
[Architecture: layer split](ARCHITECTURE.md#layer-split-across-two-backends).

## Vulkan weight streaming

Weights that spill from VRAM to host memory can be copied into staging slots
before their GEMM. This is internal to Vulkan and also applies when the whole
model is loaded on Vulkan without a layer split.

| Setting | Default and behavior |
| :-- | :-- |
| `GEIST_VK_STREAM_M_MIN` | 32 rows; `0` disables staging and reads spilled weights over the bus |
| `GEIST_VK_STREAM_ASYNC` | On where a transfer-only queue is available; `0` keeps copies in the compute batch |

The asynchronous path overlaps the next weight's copy with computation using
two staging slots. These thresholds are current implementation defaults, not
per-weight values derived from the calibration driver.

## Mixture of experts

MoE placement keeps hot experts on the GPU and computes cold experts on the
CPU. A profile can choose which experts receive resident GPU copies; the
optional runtime profile cache influences the next model load rather than
migrating experts during decode. On supported GPU paths, attention and routing
also run on the device.

Cold experts with enough prefill rows can instead be staged to the GPU;
`GEIST_MOE_STAGE=0` disables this default behavior. The staging threshold is
measured at load and can be overridden with `GEIST_MOE_STAGE_M_MIN`.
[Backend settings](BACKENDS.md#vulkan-linux-gpu-experimental) document the profile,
cache, memory budget and staging overrides.

The CPU and GPU paths share host resources. Expert coverage measured on real
text can differ substantially from pseudo-random benchmark tokens. MTP is
opt-in: its benefit depends on acceptance and the cost of verifying rows;
the recorded split-MoE experiment did not show a speedup.

## Correctness and limits

Placement and kernels are prepared at load, with runtime routing in shared
forward mechanics. Dense prefill routes per chunk, Vulkan streams per weight,
and MoE routes selected experts. This does not introduce a graph scheduler
or dtype discovery in the hot path.

Results can depend on chunk size: CPU and GPU kernels need not be bit-exact,
and GPU prefill may attend over denser cache rows than a quantized CPU cache.
The existing per-backend parity tolerances apply. Two CPU instances with an
FP32 cache exercise the split plumbing with byte-identical output in
`test_layer_split_int`; `test_layer_split_fault_int` covers backend failures.

Adaptive expert eviction, a dedicated SSD tier and multi-GPU expert splitting
are not implemented. Expert pruning or merging is an offline model
transformation, not a load-time placement decision. Unified-memory devices
have no PCIe weight transfer to hide.

Cold experts are not stored in a lower precision than hot ones: in the
measurement the quality bar held only for a small share of the selections,
no more experts fit in VRAM, and the lower-precision kernels were slower per
token.

## Evidence

[Offload measurements](../benchmark/results/OFFLOAD.md) retain the phase 0–3
experiments, rejected per-linear routing, expert-staging and MTP results with
raw-run references. [Layer-split results](../benchmark/results/LAYER-SPLIT.md)
and [MoE route-tail results](../benchmark/results/MOE_ROUTE_TAIL_2026-10-30.md)
record the split baseline and the quality cost of dropping selected experts;
[cold-expert precision results](../benchmark/results/MOE_COLD_EXPERT_PRECISION_2026-11-09.md)
record the mixed-precision expert measurement.
Those reports describe their measured revisions; use this page and
[BACKENDS.md](BACKENDS.md) for current configuration.
