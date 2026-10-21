### Added
- **`geist_weight.linear_triple_m1`: one call for three same-input M=1
  projections (#16).** The transformer's q/k/v dispatcher calls it when
  all three weights carry the same function. cpu_neon installs it on its
  Q4_K and Q6_K weights and runs q, k and v of a decode step in one
  parallel region over their joint rows, the activation quantized once
  (Llama 3.2's v is Q6_K in some layers, Q4_K in the others). Logits
  byte-identical.
