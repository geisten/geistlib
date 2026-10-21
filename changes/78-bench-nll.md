### Added
- **`bench_nll`: the perplexity of a text, fed one token at a time
  (#78).** Compares two placements of the same model on quality; for a
  mixture of experts a sharper bar than greedy tokens, which routing turns
  into other continuations at tiny numeric differences. `moe-baseline.yml`
  runs it for the CPU, the hot-expert split and the model on the GPU.
