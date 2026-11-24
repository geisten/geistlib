# The Metal prefill on the Mac mini M6, kernel by kernel, against mlx (2026-11-24)

Follow-up to [`METAL_PROFILE_M6_2026-11-23.md`](METAL_PROFILE_M6_2026-11-23.md),
which found geist's prefill GPU time at 1.45–1.47× that of mlx-lm's batched
pass and named two places without measuring either alone: the chunked
DeltaNet (Qwen3.5) and the quantized GEMM (#590). Both were measured here
one kernel at a time, on the same machine (Apple M6 Mac mini, Mac18,5, 32 GB,
macOS 27.2; runner `macmini-m6`), with mlx's own kernel on the same shapes
beside them.

**Result.**

- **The gap is the quantized GEMM, on both models.** In a prefill batch
  geist's tensor-API kernel runs at 8–15.5 TFLOP/s depending on the shape;
  mlx's `quantized_matmul` runs every 512-row shape at 17–19. Weighted by a
  model's layer, the GEMMs of 512 tokens take 7.6 ms per Llama 3.2 3B layer
  against mlx's 5.5 (1.38×) and 1.85 ms per Qwen3.5 0.8B DeltaNet layer
  against 1.26 (1.47×). 28 layers × 7.6 ms is 213 of the 250 ms of Llama's
  prefill; the profile had booked 216 under the GEMMs.
- **The DeltaNet is not it.** geist's mixer takes 0.41 ms per layer and
  256-row pass, mlx-lm's recurrence kernel alone 0.21 ms for the same rows.
  The 4.2 ms that the profile gave mlx-lm's DeltaNet came from splitting a
  command buffer's time by dispatch counts; measured, the kernel is 7 ms per
  512 tokens over the 18 layers, before its convolution and norms. Two scan
  kernels written for this note were no faster than the chunked recipe.
- **A GEMM tiled as mlx's, with the same output bit for bit, takes 6–10 %
  off the prefill** (A/B against `main`: −6.1 % on Qwen3.5 0.8B, −9.7 % on
  Llama 3.2 3B, −7.8 % on Qwen3.5 4B, −6.7 % on Gemma 4 E2B; decode and
  the logits unchanged). The kernel of #314 hands `matmul2d`
  tensors over memory and lets eight simdgroups cooperate; the new one
  gives each simdgroup 16 × 32 × 16 fragments in its registers. What
  decided its speed was not the multiply: looking up where an operand
  element goes inside the K loop cost about two thirds of the kernel, and
  a per-thread branch between vector and element-wise loads another
  seventh.
- **What remains to mlx, 1.1–1.25× per GEMM, is its 16-bit activations
  and cheaper weights.** With the matrix units the multiply is nearly free and
  a GEMM is the loads around it; mlx reads half as many activation bytes
  and dequantizes affine groups. Reading the activations as half from
  memory takes another 7 % off the fragment kernel (a bound: wrong values).
- **A finding on precision, not acted on.** With `relaxed_precision`, which
  geist's kernels and mlx's set (without it a GEMM takes three times as
  long), the matrix units take FP32 activations as FP16 truncated toward
  zero. A copy of the activations rounded to half before the GEMM is
  closer to the scalar backend (2.7e-4 against 4.3e-4 of the largest
  output of one GEMM). With the cooperative kernel it was 0–2.5 % slower
  in the models, and it moves the logits by up to 4.5e-3 of the largest
  one. It is not in this change; see
  [Activations](#activations-what-the-matrix-units-do-with-fp32).

## How it was measured

- **One GEMM**: `tests/bench_metal_gemm` builds one weight of a block
  format with random codes and scales, applies it to 256 or 512 rows of
  random activations through the weight's `linear_mN` (the call the prefill
  makes), 20 times back to back inside one prefill batch, and reports the
  median of seven such samples per GEMM. It also prints a hash of the
  output and its largest difference to `cpu_scalar` on thirteen rows.
  `tools/probes/metal_gemm_kernels_mlx.py` does the same with
  `mx.quantized_matmul` (affine groups of 64, activations `float16` for 4
  and 6 bits, `bfloat16` for 8, as mlx-lm's models hold them), 20 calls in
  one `mx.eval`. `tools/probes/metal_gemm_kernels.sh` runs both, three
  cycles; runs 4149 to 4153 are of that kind.
- **A GEMM outside a batch costs 0.11–0.23 ms more**, whatever its shape
  (the cooperative kernel with neither dequant nor multiply after the
  first K step, run 4145). Runs 4145 and 4147 ran every GEMM as a sequence
  of its own; they are quoted below only for differences between variants
  of the same shape.
- **Kernel variants** were compiled into the tensor unit on a scratch
  branch by a preprocessor prefix from the environment, one process per
  variant; they are not in the tree. The tables name what each one did.
- **DeltaNet**: `tests/bench_metal_deltanet`, 16 mixer calls in a prefill
  batch, and `tools/probes/metal_dn_kernels_mlx.py` with mlx-lm's
  `gated_delta_kernel`, 16 calls chained through the state in one
  `mx.eval`.
- **In the models**: `tools/probes/metal_prefill_gemm.sh`, the workload of
  `MLX_VS_GEIST_M6_2026-11-22.md` (512 prompt tokens, 128 greedy tokens),
  `origin/main` against this change, this change with the old kernel, and
  mlx-lm, interleaved; `tests/bench_nll`'s logits hash; and
  `tools/bench_revision_ab.py`.

"Model-weighted" below is the sum over the GEMMs of one layer for 512
tokens, from the shapes measured: Llama 3.2 3B `2 × 3072→3072 + 2 ×
3072→1024 + 2 × 3072→8192 + 8192→3072` as Q4_K; a Qwen3.5 0.8B DeltaNet
layer `1024→6144 + 2 × 2048→1024 + 2 × 1024→3584 + 3584→1024` as Q8_0 (the
2048→1024 stands for the gate and the output projection); Qwen3.5 4B `2 ×
2560→9216 + 9216→2560 + 2560→8192` as Q4_0. geist prefills in passes of 256
rows, so its figure is twice the 256-row time unless it says "512 rows";
mlx-lm batches the prompt in one pass, so its figure is the 512-row time.

## One GEMM

Run 4153, the two kernels of this change and mlx on the same shapes; ms
per GEMM in a prefill batch, median of three cycles; TFLOP/s = 2 × rows ×
n_in × n_out / time. "Same output": the two kernels' output hash. The last
column is the largest difference to `cpu_scalar` over the largest output,
on thirteen rows.

| format n_in x n_out, rows | cooperative ms | fragments ms | change | same output | cooperative TFLOP/s | fragments TFLOP/s | mlx ms | mlx TFLOP/s | difference to cpu_scalar (coop, fragments) |
|:--|--:|--:|--:|:--|--:|--:|--:|--:|:--|
| q4_0 2560 x 8192, 256 | 0.724 | 0.645 | -11 % | yes | 14.8 | 16.6 | 0.598 | 18.0 | 4.25e-04, 4.25e-04 |
| q4_0 2560 x 8192, 512 | 1.414 | 1.257 | -11 % | yes | 15.2 | 17.1 | 1.178 | 18.2 | 4.05e-04, 4.05e-04 |
| q4_0 2560 x 9216, 256 | 0.793 | 0.718 | -9 % | yes | 15.2 | 16.8 | 0.672 | 18.0 | 3.91e-04, 3.91e-04 |
| q4_0 2560 x 9216, 512 | 1.558 | 1.400 | -10 % | yes | 15.5 | 17.3 | 1.325 | 18.2 | 3.91e-04, 3.91e-04 |
| q4_0 9216 x 2560, 256 | 0.965 | 0.793 | -18 % | yes | 12.5 | 15.2 | 0.668 | 18.1 | 4.39e-04, 4.39e-04 |
| q4_0 9216 x 2560, 512 | 2.146 | 1.815 | -15 % | yes | 11.3 | 13.3 | 1.336 | 18.1 | 4.45e-04, 4.45e-04 |
| q4_k 3072 x 1024, 256 | 0.135 | 0.129 | -4 % | yes | 11.9 | 12.5 | 0.389 | 4.1 | 4.36e-04, 4.36e-04 |
| q4_k 3072 x 1024, 512 | 0.381 | 0.281 | -26 % | yes | 8.5 | 11.5 | 0.180 | 17.9 | 4.83e-04, 4.83e-04 |
| q4_k 3072 x 3072, 256 | 0.355 | 0.329 | -7 % | yes | 13.6 | 14.7 | 0.266 | 18.2 | 4.44e-04, 4.44e-04 |
| q4_k 3072 x 3072, 512 | 0.756 | 0.653 | -14 % | yes | 12.8 | 14.8 | 0.516 | 18.7 | 4.44e-04, 4.44e-04 |
| q4_k 3072 x 8192, 256 | 0.914 | 0.808 | -12 % | yes | 14.1 | 15.9 | 0.693 | 18.6 | 4.30e-04, 4.30e-04 |
| q4_k 3072 x 8192, 512 | 1.818 | 1.554 | -15 % | yes | 14.2 | 16.6 | 1.369 | 18.8 | 4.30e-04, 4.30e-04 |
| q4_k 8192 x 3072, 256 | 0.997 | 0.841 | -16 % | yes | 12.9 | 15.3 | 0.699 | 18.4 | 3.66e-04, 3.66e-04 |
| q4_k 8192 x 3072, 512 | 2.325 | 1.804 | -22 % | yes | 11.1 | 14.3 | 1.396 | 18.5 | 3.66e-04, 3.66e-04 |
| q6_k 3072 x 1024, 256 | 0.142 | 0.131 | -8 % | yes | 11.3 | 12.3 | 0.393 | 4.1 | 5.09e-04, 5.09e-04 |
| q6_k 3072 x 1024, 512 | 0.402 | 0.286 | -29 % | yes | 8.0 | 11.3 | 0.192 | 16.8 | 4.77e-04, 4.77e-04 |
| q6_k 8192 x 3072, 256 | 1.232 | 1.016 | -18 % | yes | 10.5 | 12.7 | 0.770 | 16.7 | 4.27e-04, 4.27e-04 |
| q6_k 8192 x 3072, 512 | 2.328 | 2.779 | +19 % | yes | 11.1 | 9.3 | 1.566 | 16.5 | 4.27e-04, 4.27e-04 |
| q8_0 1024 x 3584, 256 | 0.165 | 0.144 | -13 % | yes | 11.4 | 13.0 | 0.112 | 16.8 | 4.26e-04, 4.26e-04 |
| q8_0 1024 x 3584, 512 | 0.305 | 0.264 | -13 % | yes | 12.3 | 14.2 | 0.214 | 17.6 | 4.26e-04, 4.26e-04 |
| q8_0 1024 x 6144, 256 | 0.235 | 0.209 | -11 % | yes | 13.7 | 15.4 | 0.185 | 17.4 | 3.84e-04, 3.84e-04 |
| q8_0 1024 x 6144, 512 | 0.452 | 0.400 | -12 % | yes | 14.3 | 16.1 | 0.360 | 17.9 | 3.84e-04, 3.84e-04 |
| q8_0 2048 x 1024, 256 | 0.097 | 0.080 | -18 % | yes | 11.1 | 13.4 | 0.268 | 4.0 | 4.01e-04, 4.01e-04 |
| q8_0 2048 x 1024, 512 | 0.235 | 0.187 | -20 % | yes | 9.1 | 11.5 | 0.128 | 16.8 | 3.81e-04, 3.81e-04 |
| q8_0 3584 x 1024, 256 | 0.165 | 0.135 | -18 % | yes | 11.4 | 13.9 | 0.446 | 4.2 | 4.29e-04, 4.29e-04 |
| q8_0 3584 x 1024, 512 | 0.461 | 0.376 | -18 % | yes | 8.2 | 10.0 | 0.221 | 17.0 | 4.29e-04, 4.29e-04 |

Model-weighted, 512 tokens, ms:

| | Llama 3.2 3B layer | Qwen3.5 0.8B DeltaNet layer | Qwen3.5 4B |
| :-- | --: | --: | --: |
| cooperative kernel (#314, `main`), two passes of 256 rows | 7.61 | 1.85 | 6.55 |
| fragments (this change), two passes of 256 rows | 6.75 | 1.58 | 5.75 |
| change | −11 % | −15 % | −12 % |
| mlx, one pass of 512 rows | 5.53 | 1.26 | 5.16 |
| fragments over mlx | 1.22 | 1.25 | 1.11 |
| cooperative kernel, one pass of 512 rows | 8.23 | 1.99 | 6.68 |
| fragments, one pass of 512 rows | 6.78 | 1.68 | 5.87 |

One shape is slower with the fragments: Q6_K 8192→3072 at 512 rows in one
call, 2.78 against 2.33 ms (+19 %; the same in run 4151). At 256 rows, which
is what a prefill pass is unless `GEIST_M_MAX` says otherwise, it is 1.02
against 1.23 ms (−18 %).

mlx's 256-row figures on the 1024-output shapes (4 TFLOP/s) are its own:
it is slower there at 256 rows than at 512 in absolute time. mlx-lm does not
run those; its prefill is one 511-row pass.

### Where the cooperative kernel's time goes

Run 4151, model-weighted ms (two passes of 256 rows):

| cooperative kernel (#314, `main`) | Llama 3.2 3B layer | Qwen3.5 0.8B DeltaNet layer | Qwen3.5 4B |
| :-- | --: | --: | --: |
| as it is | 7.62 | 1.78 | 6.57 |
| weights dequantized in the first K step only | 6.69 | 1.75 | 5.88 |
| `matmul2d` in the first K step only | 2.57 | 1.48 | 2.13 |
| neither | 1.61 | 1.26 | 0.87 |
| mlx, one pass of 512 rows | 5.53 | 1.27 | 5.17 |

On the large shapes two thirds of the kernel are the `matmul2d` call,
which loads its operands itself from the tensors it is given (the weight
tile in threadgroup memory, the activations in device memory), and a
tenth is the dequant. Static extents for whole tiles, which the header of
MetalPerformancePrimitives recommends, changed nothing (run 4147: 1.913
against 1.929 ms at 3072→8192, 512 rows); Q4_K's dequant by 32-bit words
instead of bytes changed nothing either (run 4145).

## Fragments

mlx's `affine_qmm_t_nax` gives a simdgroup a 32-row × 32-output tile and
multiplies it in fragments of 16 × 32 × 16 through a `matmul2d` of
`execution_simdgroup` scope, with both operands and the result in
cooperative tensors, which are registers. geist's port keeps FP32
activations and its own block dequant. Model-weighted ms, two passes of
256 rows, then one of 512 (runs 4149 and 4151; every row but the bounds
has the cooperative kernel's output hash on all 26 shape and row
combinations):

| fragment kernel | Llama 3.2 3B layer | Qwen3.5 0.8B DeltaNet layer | Qwen3.5 4B |
| :-- | --: | --: | --: |
| cooperative kernel, for scale | 7.62, 8.34 | 1.78, 2.05 | 6.57, 6.81 |
| element offsets taken before the K loop, loaded one by one; 64-row threadgroups | 12.14, 11.32 | 2.69, 2.59 | 10.42, 9.85 |
| the same with a branch per thread to vector loads | 9.32, 8.79 | 2.18, 2.03 | 7.38, 6.95 |
| vector loads fixed at compile time | 8.08, 7.60 | 1.89, 1.75 | 6.56, 6.18 |
| 32-row threadgroups (64 threads) | 15.74, 15.13 | 3.08, 2.99 | 10.78, 10.12 |
| 128-row threadgroups (256 threads), vector loads | 7.02, 6.86 | 1.70, 1.75 | 6.16, 5.95 |
| 256-row threadgroups (512 threads), vector loads | 6.93, 7.27 | 1.88, 1.92 | 6.21, 6.76 |
| 128 rows, weight rows 72 apart in threadgroup memory | 6.94, 6.73 | 1.67, 1.67 | 6.02, 5.83 |
| 128 rows, output tiles along the grid's width | 6.77, 11.06 | 1.70, 2.79 | 5.83, 10.07 |
| mlx, one pass of 512 rows | 5.53 | 1.27 | 5.17 |

- **Asking the cooperative tensor where an element goes is expensive.**
  The first port called `is_valid_element` and
  `get_multidimensional_index` for every operand element of every
  fragment. It took 5.67 ms for 3072→8192 at 512 rows where the
  cooperative kernel took 1.93 (run 4147, per-GEMM sequences), and twice
  the cooperative kernel's time on an M1 Max's ALUs. The offsets do not
  change during a GEMM; taken once, before the loop, the same kernel takes
  2.05 ms (run 4149), and on the M1 Max a tenth more than the cooperative
  kernel.
- **A branch per thread between vector and element-wise loads costs a
  seventh**: the two paths are compiled into one loop body. On the matrix
  units a thread holds its 8 left and 16 right elements as runs of four
  along K (the layout mlx's kernel hard-codes); the ALUs' `matmul2d` lays
  them out otherwise. The kernel in the tree is therefore compiled for one
  or the other: `tc_frag_layout`, one simdgroup, reports the layout when
  the unit is first built, and a device without the runs gets the unit
  compiled again for element-wise loads.
- **128-row threadgroups**: a weight tile is dequantized once per
  threadgroup and K step, so two pairs of simdgroups per threadgroup
  dequantize twice as often as four. Eight pairs (512 threads) are no
  better. Without any dequant the 128-row kernel would take 5.86 ms per
  Llama layer.
- **Weight rows 72 halves apart instead of 64** (mlx pads them by 16
  bytes): 1–2 %.
- **The grid's order matters, and not in one direction.** With the output
  tiles along the grid's width and two row tiles (a 256-row pass), a GEMM
  takes 3.5 % less; with four row tiles 60 % more. The cooperative kernel
  gains nothing at two (7.85 against 7.62 ms per Llama layer) and loses
  37 % at four (11.45 against 8.34). The kernel in the tree puts the
  output tiles along the width when a call has at most two row tiles, and
  tells by the grid's shape which way it was laid out. One shape goes the
  other way at four row tiles: Q6_K 8192→3072 takes 2.20 ms with the
  output tiles along the width and 2.76 without.
- **No effect** (128 rows, vector loads, 7.02 ms per Llama layer):
  accumulators in plain registers with the cooperative tensor filled per
  multiply, as mlx's code does it (7.03); the K loop not unrolled (7.11);
  aligned instead of packed vector loads (7.04).
- **Worse**, both measured with the per-thread branch: a K step of 128
  instead of 64 (8.25 against 7.98 at 128 rows) or of 32 (9.61 against
  9.32 at 64 rows).
- **Not possible**: more than one simdgroup per `matmul2d` with the
  operands in registers. MetalPerformancePrimitives refuses it ("Input
  cooperative tensors require a single SIMD group"), so operands shared by
  several simdgroups have to be tensors over memory, which is the
  cooperative kernel.
- **`relaxed_precision` off**: 23.0 ms per Llama layer for the cooperative
  kernel and 24.1 for the fragments, three times the time, and another
  output.

### What is left to mlx

| 128-row fragment kernel, vector loads | Llama 3.2 3B layer | Qwen3.5 0.8B DeltaNet layer | Qwen3.5 4B |
| :-- | --: | --: | --: |
| as measured | 7.02 | 1.70 | 6.16 |
| no dequant after the first K step (bound) | 5.86 | 1.62 | 5.36 |
| the second threadgroup barrier of a K step left out (bound, races) | 6.67 | 1.64 | 5.84 |
| activations read as half from device memory (bound, wrong values) | 6.51 | 1.71 | 5.54 |
| no multiply after the first K step | 8.14 | 1.98 | 7.11 |
| mlx, one pass of 512 rows | 5.53 | 1.27 | 5.17 |

Leaving out the multiply does not make the kernel faster. On the matrix
units a quantized GEMM is the work around the multiply: loading operands
into registers, dequantizing, and the two barriers per K step. mlx loads
half as many activation bytes, since its activations are 16-bit throughout
the model, and its dequant is one scale and one bias per 64 weights where
Q4_K has two 6-bit values per 32 and Q8_0 a scale per 32. How much of the
remaining 1.1–1.25× each of these is was not separated further; none of it
is reachable with FP32 activations and GGUF's block formats by retiling.

## Activations: what the matrix units do with FP32

`bench_metal_gemm`'s difference to `cpu_scalar`, largest over thirteen rows,
over the largest output; Q8_0, 1024→3584, 256 rows:

| path | difference |
| :-- | --: |
| simdgroup kernels (M1 Max): half weights, activations rounded to half | 2.74e-4 |
| tensor kernels on an M1 Max's ALUs (`GEIST_METAL_TENSOR=1`): half weights, FP32 activations | 1.89e-4 |
| tensor kernels on the M6's matrix units, both kernels | 4.26e-4 |
| the same with the activations converted to half in the kernel or copied as half first | 2.74e-4 |

A model of the GEMM in double precision on the same random data gives
1.89e-4 for half weights with FP32 activations, 2.74e-4 with the
activations rounded to half, and 4.26e-4 with the activations truncated to
half toward zero; rounding to bfloat16 would give 1.3e-3. The three
measured figures are the model's to the last digit (on this shape; the
others were not modelled, their differences have the same sizes). That is
an inference from one number per path, and a tight one: with
`relaxed_precision` the matrix units take an FP32 operand as its half
truncated toward zero. The kernel's comment in #314 ("read as float
straight from device memory") describes what is passed, not what is
multiplied.

The cooperative kernel fed a copy of the activations rounded to half (the
existing `cvt_f32_f16_rows` into the scratch the PQ2_0 simdgroup kernel
uses, one conversion per GEMM) was measured as a scratch variant:

| | Llama 3.2 3B layer | Qwen3.5 0.8B DeltaNet layer | Qwen3.5 4B |
| :-- | --: | --: | --: |
| cooperative kernel | 7.62, 8.34 | 1.78, 2.05 | 6.57, 6.81 |
| with a rounded half copy | 7.25, 7.13 | 1.81, 1.87 | 6.23, 6.00 |
| 128-row fragments, vector loads | 7.02, 6.86 | 1.70, 1.75 | 6.16, 5.95 |
| with a rounded half copy | 6.90, 6.62 | 1.83, 1.73 | 5.84, 5.59 |

In the models (run 4151, two cycles; cooperative kernel with the copy
against `main`): prefill −2.3 % on Qwen3.5 0.8B, +0.1 % on Llama 3.2 3B,
−2.5 % on Qwen3.5 4B, −2.2 % on Gemma 4 E2B, since a conversion per GEMM
is a dispatch per GEMM. The prefill-path logits of `bench_nll` move by at
most 4.1e-4 of the largest logit on Qwen3.5 0.8B, 1.2e-3 on Llama 3.2 3B,
2.0e-3 on Qwen3.5 4B and 4.5e-3 on Gemma 4 E2B (0.136 at a largest logit of
30.0); the greedy tokens of the 128-token workload stay the same on all
four, and one of 127 scored positions changes its top token on Qwen3.5
0.8B. That is a different rounding of the activations, closer to the
reference per GEMM, and more than a reordering of sums. It is described
here and left out of the change.

## DeltaNet

ms per call, 16 calls in a prefill batch (run 4149; geist's call is the
whole mixer of one layer, mlx-lm's the recurrence alone, its convolution
and norms are other mlx ops):

| | 16 value heads, 256 rows | 512 rows | 32 value heads, 256 rows | 512 rows |
| :-- | --: | --: | --: | --: |
| geist: conv, norms, chunked delta rule, gated norm (7 kernels) | 0.405 | 0.946 | 0.807 | 1.721 |
| mlx-lm `gated_delta_kernel` | 0.205 | 0.394 | 0.377 | 0.720 |
| scan prototype: 8 threads per value column, 16 state rows each | 0.465 | 1.078 | 0.807 | 1.716 |
| scan prototype: 4 threads per column, 32 rows each (mlx-lm's layout) | 0.417 | 0.957 | 0.839 | 1.796 |

Each of the seven kernels dispatched alone, 16 value heads and 256 rows
(run 4147; a kernel alone still pays its dispatch, so the seven add up
to more than the mixer): `dnf_state` 0.148,
`dnf_out` 0.097, `dnf_chunk` 0.073, `dnf_conv` 0.061, `dnf_norm` 0.041,
`dnf_cst_copy` and `dnf_roll` 0.009 each. Leaving one out of the seven:
−0.148 (state), −0.110 (out), −0.073 (chunk), −0.073 (conv), −0.024 (norm),
0 for the two copies.

On Qwen3.5 0.8B's 18 DeltaNet layers that is 2 × 0.405 × 18 = 14.6 ms per
512 tokens for geist (the profile's one-kernel-per-buffer run had 15.3)
against 0.394 × 18 = 7.1 ms for mlx-lm's recurrence plus its convolution
and norms, which were not measured alone. The recurrence itself, written
as a scan with the state in registers the way mlx-lm does it, was not
faster in geist than the chunked recipe: 0.42–0.47 ms with the prep and
the norm it still needs, against 0.405. mlx-lm's kernel reads q and k as
bfloat16. The scan's results are also not those of the chunked recipe (it
is closer to the double-precision oracle of
`tests/test_backend_metal_deltanet_chunk_unit.c`, worst 3.4e-6 against
5.4e-6, and independent of how a prompt is cut into calls). It was
dropped: it does not measure.

## In the models

Run 4153, this change at `b9ec2ea4` against `origin/main` at `16c7df30`.
512 prompt tokens, 128 greedy tokens. Cells: three interleaved cycles of
three repeats each, median (minimum–maximum), every cell behind the gate of
`mlx_vs_geist.py`; "coop" is this change with `GEIST_METAL_TC_COOP=1`.

| model | arm | prefill tok/s | against main | decode tok/s | against main |
| :-- | :-- | --: | --: | --: | --: |
| Qwen3.5 0.8B, 8-bit | main | 7431 (7391–7456) | | 162.0 (161.1–162.3) | |
| | this change | 7929 (7884–7947) | +6.7 % | 162.1 (162.0–162.4) | +0.1 % |
| | this change, coop | 7442 (7401–7455) | +0.1 % | 162.2 (162.0–162.2) | +0.1 % |
| | mlx-lm | 7256 (7167–7377) | −2.4 % | 160.1 (158.7–160.1) | −1.2 % |
| Llama 3.2 3B, 4-bit | main | 2057 (2054–2058) | | 68.4 (68.4–68.5) | |
| | this change | 2278 (2276–2283) | +10.8 % | 68.4 (68.3–68.5) | 0.0 % |
| | this change, coop | 2055 (2052–2062) | −0.1 % | 68.4 (68.4–68.4) | 0.0 % |
| | mlx-lm | 2636 (2631–2639) | +28.1 % | 77.1 (76.8–77.2) | +12.6 % |
| Qwen3.5 4B, 4-bit | main | 1568 (1541–1575) | | 49.6 (49.6–49.7) | |
| | this change | 1705 (1669–1711) | +8.8 % | 49.6 (49.6–49.7) | 0.0 % |
| | this change, coop | 1568 (1541–1572) | 0.0 % | 49.7 (49.6–49.7) | +0.1 % |
| | mlx-lm | 1811 (1799–1812) | +15.5 % | 57.1 (56.8–57.2) | +15.0 % |
| Gemma 4 E2B, 4-bit | main | 7147 (7106–7173) | | 87.8 (87.8–87.9) | |
| | this change | 7649 (7566–7690) | +7.0 % | 87.8 (87.6–87.9) | 0.0 % |
| | this change, coop | 7150 (7115–7178) | 0.0 % | 87.8 (87.5–87.9) | 0.0 % |
| | mlx-lm | 7667 (7626–7720) | +7.3 % | 96.7 (96.6–96.8) | +10.1 % |

mlx-lm's prefill is now 15.7 % ahead on Llama 3.2 3B (was 28.1 %), 6.2 %
on Qwen3.5 4B (15.5 %), level on Gemma 4 E2B (7.3 %), and 8.5 % behind on
Qwen3.5 0.8B, where its end-to-end time carries host work between its
batched pass and its first token (see the profile).

### A/B against main

`tools/bench_revision_ab.py`, each revision built from scratch, 10
interleaved cycles with a control copy of the baseline's binary, every run
from a cool chip on AC power:

| model | row | main ms | this change ms | change | 95 % interval | faster | control |
| :-- | :-- | --: | --: | --: | --: | --: | --: |
| Qwen3.5 0.8B Q8_0 | prefill @ 512 | 69.2 | 65.0 | −6.1 % | [−6.4 %, −5.8 %] | 10/10 | 0.0 % [−0.2 %, +0.2 %] |
| | decode, 128 tokens | 789.5 | 789.9 | +0.1 % | [0.0 %, +0.4 %] | 2/10 | 0.0 % [−0.1 %, +0.1 %] |
| | first token | 69.5 | 65.3 | −6.1 % | [−6.3 %, −5.8 %] | 10/10 | 0.0 % [−0.2 %, +0.3 %] |
| Llama 3.2 3B Q4_K_M | prefill @ 512 | 248.6 | 224.6 | −9.7 % | [−9.9 %, −9.5 %] | 10/10 | +0.1 % [0.0 %, +0.3 %] |
| | decode, 128 tokens | 1871.3 | 1870.5 | −0.1 % | [−0.1 %, 0.0 %] | 7/10 | 0.0 % [0.0 %, +0.2 %] |
| | first token | 248.8 | 224.8 | −9.7 % | [−9.8 %, −9.5 %] | 10/10 | +0.1 % [0.0 %, +0.3 %] |
| Qwen3.5 4B Q4_0 | prefill @ 512 | 332.2 | 306.4 | −7.8 % | [−7.9 %, −7.7 %] | 10/10 | 0.0 % [−0.1 %, +0.1 %] |
| | decode, 128 tokens | 2577.9 | 2578.9 | 0.0 % | [−0.1 %, +0.1 %] | 5/10 | 0.0 % [−0.2 %, +0.1 %] |
| | first token | 332.6 | 306.7 | −7.8 % | [−7.9 %, −7.7 %] | 10/10 | 0.0 % [−0.1 %, +0.1 %] |
| Gemma 4 E2B Q4_K_M | prefill @ 512 | 72.0 | 67.1 | −6.7 % | [−6.9 %, −6.6 %] | 10/10 | 0.0 % [−0.2 %, +0.1 %] |
| | decode, 128 tokens | 1457.7 | 1457.8 | 0.0 % | [−0.3 %, +0.1 %] | 5/10 | +0.1 % [−0.3 %, +0.1 %] |
| | first token | 72.0 | 67.1 | −6.7 % | [−6.9 %, −6.6 %] | 10/10 | 0.0 % [−0.2 %, +0.1 %] |

The control's intervals reach ±0.3 % (±0.2 % on Qwen3.5 4B). Decode is
within noise on all four; Qwen3.5 0.8B's +0.1 % has an interval that
touches zero and the control's width.

### The same output

- `tests/bench_nll`'s hash over every scored logits vector, decode path (95
  tokens one at a time) and prefill path (511 tokens in one call, then 127
  scored steps): equal for `main`'s build, this change and this change
  with the cooperative kernel, on all four models (the hashes are in the
  run's cells).
- The 128 greedy tokens of every cell are `main`'s, on all four models.
- `bench_metal_gemm`'s output hash: equal for the two kernels on all 26
  combinations above.
- `tests/test_backend_metal_parity_unit.c` compares the two kernels bit
  for bit on six formats and nine shapes. It passed on the M6 with the
  fragments' vector loads against the cooperative kernel, and on an M1 Max
  (`GEIST_METAL_TENSOR=1`, the ALUs) with their element-wise loads. The
  other Metal unit tests passed on the M6 with either kernel
  (`test_backend_metal_deltanet_unit`, `_deltanet_chunk_unit`,
  `test_metal_budget_unit`, `_concurrent_unit`, `_pq2_lowp_unit`,
  `test_decode_ahead_unit`; `test_metal_autorelease_unit` and
  `_resources_unit` skip without their model).

### What it costs

- Process start to the first token of a one-token prompt, ten alternating
  runs each, median: 66.4 against 66.5 ms (Qwen3.5 0.8B), 104.3 against
  104.8 (Llama 3.2 3B), 105.4 against 105.6 (Qwen3.5 4B), 83.8 against 84.2
  (Gemma 4 E2B): 0.1–0.6 ms for `tc_frag_layout` and its compile.
- 1 KiB more threadgroup memory per GEMM threadgroup (9216 bytes).
- The unit's source holds both kernels of every format; one is compiled.
- At 512 rows in one call, Q6_K 8192→3072 is 19 % slower than before (see
  One GEMM).
- On a GPU without matrix units nothing changes: `GEIST_METAL_TENSOR=1`
  there still runs the cooperative kernel.

## Limits

- One machine and one day. The M5 has matrix units too and was not
  measured; the fragment kernel is its default as well.
- The per-GEMM figures are wall times of 20 GEMMs in a batch on an
  otherwise idle GPU, not GPU times of a GEMM inside a forward. They
  reproduce the profile's GEMM total for Llama 3.2 3B (214 against 216 ms).
- mlx's kernel and geist's do not compute the same thing: other weight
  formats, 16-bit against 32-bit activations. The comparison is of time
  for the same shapes.
- What the remaining 1.1–1.25× per GEMM is made of was bounded, not
  separated.
- The rule for the grid's order was measured at two and at four row tiles.
- On an M1 Max the change is inert by construction (no tensor unit); an
  A/B there was not taken: the host stayed above the benchmark's cool
  gate under other load.

## Raw data

`benchmark/results/raw/metal-prefill-gemm-m6-2026-11-24/`: the job
summaries of runs 4145 and 4147 (per-GEMM sequences: variants of the
cooperative kernel, the first fragment kernel, the DeltaNet kernels alone),
4149 and 4151 (in a batch: the variant tables, with the scratch branch's
arms in the models), and 4153 (this change: unit tests on the M6,
the two kernels and mlx, the models, the A/B reports), with the cells of
the model runs (`*-cells.jsonl`: every sample, the gate readings, the
tokens, the logits hashes).
