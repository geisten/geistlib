### Changed
- **A mixture of experts on the GPU keeps no CPU copy of its hot experts
  (#164).** With the model on a GPU backend, the experts that get a copy
  there are no longer also resolved (and on cpu_x86 repacked) on the
  state's CPU backend: the placement now comes first and only the experts
  left on the CPU are resolved there, and the hot experts' file pages are
  dropped once the GPU has copied them. A session that cannot get its GPU
  buffers is refused with an error naming the experts that have no CPU
  copy, where it used to run every expert on the CPU.
