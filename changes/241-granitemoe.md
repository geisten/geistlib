### Added
- **Granite-MoE loads (#241).** `general.architecture` `granitemoe`: Granite
  3.0's A400M and A800M. Mixtral's stack with four multipliers from the file.
  Only the attention scale reaches the forward path; the embedding, residual
  and logit scales become load-time constants, because every use of the
  hidden state is an RMSNorm, so no backend changes. Perplexity of Granite 3.0
  1B A400M Q4_K_M on a 255-token text: 4.84 on Metal and cpu_scalar. The
  comparison with `llama-perplexity` is still open.
