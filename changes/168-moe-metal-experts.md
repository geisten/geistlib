### Changed
- **A mixture of experts on a unified-memory GPU runs its experts there
  (#168).** With the model loaded on Metal, every expert used to stay on
  the state's CPU backend (no GPU copy on unified memory unless
  `GEIST_MOE_HOT` forced one), so each layer handed its rows to cpu_neon
  and waited for them. Now every expert that fits the GPU's working set gets
  a GPU twin, as on a discrete card; on unified memory the twin aliases the
  mapping, so nothing is copied. A CPU model with a unified-memory GPU as
  split backend keeps its experts on the CPU as before.
