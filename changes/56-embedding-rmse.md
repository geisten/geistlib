### Fixed
- **The published-embedding test states and gates its current RMSE (#56).**
  `test_published_embedding_e2e` reproduces Microsoft's BitNet-embedding
  vector to RMSE 1.01e-3 on cpu_neon since the online FP32-KV softmax
  (f78ce0f1), not the documented 8.5e-4: the reordered sum, amplified by the
  int8 activation quantization of the I2_S linears, moves single components
  by up to 1.1e-3. The test and `docs/BITNET_EMBEDDINGS_PLAN.md` say so, and
  the bound tightens from 5e-3 to 1.5e-3.
