### Changed
- **A mixture of experts on the GPU decodes on a busy host (#161).** With
  the model on the GPU, a decode step's cold experts ran in an OpenMP
  region, whose end waits for every thread of the team, including one the
  OS has not scheduled yet; on a host whose cores were busy a layer's cold
  experts then took 6-7 ms instead of 0.2 and decode fell to 2.5 tok/s.
  They now run on helper threads of the model's own (`moe_pool.h`, up to
  7, `GEIST_MOE_POOL=<n>`, 0 turns them off) that claim the experts one by
  one; the calling thread claims too and waits only for experts a helper
  took. Kernel calls inside such a task stay on its thread
  (`geist_par_set_inline`).
