### Changed
- **Metal: tensor-API prefill GEMMs retiled for the M6, prefill another
  7-28 % faster on M5/M6-class Macs (#314).** The `matmul2d` GEMMs now step
  K by 64 instead of 32 and run on 8 simdgroups instead of 4 (tile still
  64 weight rows x 128 activation rows), from a sweep of tile shapes on the
  Apple M6. M1-M4 are unchanged (no tensor path there).
