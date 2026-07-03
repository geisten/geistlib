# Metal backend → main backend contract: staged port plan

Branch `gpu-backends-main-port` (based on `origin/main` @ bca3ce8). Goal: run
the Metal GPU backend on main's authoritative engine, preserving the perf
kernels from `gpu-backends-wip` (mm_sg GEMM, simdgroup flash attention, q6k,
strided PLE). Verify bit-exact against the pre-merge baseline, then benchmark.

## What's done (this checkpoint)
- Port branch created on main. `gpu-backends-wip` untouched (backup tag
  `gpu-backends-wip-premerge-20260703` + branch `gpu-backends-wip-backup`).
- `src/backends/metal/backend.c` brought over (12137 lines; ~10k are MSL
  kernel strings that are contract-neutral and reusable as-is).
- Build wiring: `mk/backend-metal.mk` (+`-ldl`, runtime dlopen of Metal/objc)
  and the `GEIST_BACKEND_METAL` block in `src/engine/backend_registry.c`.
- Include fixed: `gguf_quant.h` → `quant.h`.

## Feasibility — proven, low-risk on the ABI
- `struct geist_tensor` is **byte-identical** main vs wip → every encode
  function that speaks geist_tensor compiles unchanged.
- `struct geist_weight` differs only by our extra `linear_triple_mN`.
- Our mm_sg / q6k kernels read **native GGUF `block_q4_K`/`block_q6_K`**
  directly (block struct is defined inside the MSL string), so weights need
  no device repack — `w->raw` (mmap'd GGUF) can be wrapped no-copy as an
  MTLBuffer on Apple unified memory.
- main's `quant.h` exposes only `const void*` dequant fns (no block struct);
  the C glue does not need the struct (all 6 `block_q4_K` uses are in MSL).

## Exact remaining compile surface (from `-fsyntax-only -ferror-limit=200`)
All 201 errors trace to 6 old-contract types main deleted + the command-
sequence enum. These back the OLD fine-grained vtbl ops; DELETE the op impls,
KEEP their MSL kernels + low-level encode fns for the new ops:
- `geist_backend_greedy_head` / `_batch` (133) → metal greedy-head op impls
- `geist_backend_attention_block` / `_query_block` (42) → old attention ops
- `geist_backend_ffn_geglu_block` (14) → fused FFN op
- `geist_backend_ple_block` (10) → fused PLE op
- `geist_command_sequence_kind` + `GEIST_COMMAND_SEQUENCE_*` → the whole
  command-sequence batching machinery (~20 refs)

## Progress (2026-07-03)
- **Stage 1 DONE** (commit 36ce89b): compiles/links/registers; device creates;
  buffer_create works. Descriptor rewritten to main's 25 fields; old op impls
  kept compilable via metal_legacy_ops.h (backend-private struct copies).
- **Stage 2 DONE** (commit ff9603b): full gemma-E2B model LOADS on metal (717ms).
  metal_buffer_create_aliased (copy into SHARED MTLBuffer) + all buffers made
  host-visible (buffer_map must work for weights/scratch on main's contract).
- **Stage 4 NEXT — design resolved.** Blocker: metal_resolve_weight is a stub,
  so linear reports UNSUPPORTED and prefill returns 0. KEY FACT: every weight's
  `w->raw = buffer_map(buf) + offset` (layer_wiring.c) → w->raw ALWAYS points
  into a metal-created MTLBuffer's contents. x/y in linear come from
  buffer_map of metal scratch buffers. So:
  1. Add a registry to metal_state: {contents_ptr, size, MTLBuffer} filled in
     metal_new_buffer AND metal_buffer_create_aliased, cleared on destroy.
  2. metal_resolve_weight installs metal_linear_m1/mN for Q4_K/Q6_K/F32.
  3. metal_linear_mN(x,w,m,be,y): range-lookup x, w->raw, y in the registry →
     (MTLBuffer, offset); build geist_tensors; reuse the existing GEMM encode
     fns (metal_encode_matmul_q4k / _q6k / _f32) on a standalone command buffer;
     commit + wait. m==1 uses the matvec kernels.
  Gate: matmul parity vs cpu_scalar (buffer_download), then decode checksum
  0x609900994dc13840. Then benchmark + llama compare + README.

## Stages (verify after each)
1. **Compile skeleton.** Delete the 6 old-contract op impls + command-sequence
   machinery. Rewrite the metal descriptor to main's 25-field vtbl. Stub the
   new ops to `GEIST_E_UNSUPPORTED`. Gate: `make BACKENDS="cpu_scalar metal"`
   links; `geist_backend_create("metal")` succeeds.
2. **Buffer ops.** create/destroy/create_aliased/upload/download/map/unmap —
   mostly already present; adapt signatures to main. map must return a stable
   host alias into the MTLBuffer (unified memory).
3. **Level-2 ops** (reuse existing encode fns): rmsnorm, add, mul, gelu_tanh,
   gelu_tanh_mul, rope_apply, embedding_lookup. Gate: per-op parity vs
   cpu_scalar via buffer_download.
4. **GPU linear via resolver.** `metal_resolve_weight` installs `linear_m1`/
   `linear_mN` for q4_K/q6_K/f32 that: wrap x/y (host ptrs from the engine's
   buffer_map) + `w->raw` as no-copy MTLBuffers, encode the mm_sg / q6k GEMM,
   commit, wait. This is the crux — main's linear path is host-pointer based;
   unified memory makes the GPU dispatch correct. Gate: matmul parity.
5. **attention** (main signature: F32 q/k/v, q_offset, sliding_window). Adapt
   the flash-sg kernel or use the plain F32 attention_rows kernel; qnorm/rope
   are done by the engine separately, so metal attention is plain SDPA.
   Gate: decode checksum 0x609900994dc13840 (TG=8 short-ctx).
6. **Perf pass (later).** Implement `transformer_block` to re-batch the layer
   into one submit (recover the command-sequence win) + re-enable flash/q6k
   fast paths. Gate: pp512 within ~5% of the 1072 tok/s baseline.

Baseline to beat: `docs/benchmarks/metal-baseline-premerge-20260703.txt`
(pp512 1072, pp2048 754, tg128 61 tok/s cool; checksum 0x609900994dc13840).
