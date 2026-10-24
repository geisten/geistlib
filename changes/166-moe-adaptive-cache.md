### Added
- **An adaptive expert cache for a mixture of experts on a discrete GPU
  (#166, opt-in: `GEIST_MOE_ADAPT=1`).** Per window of decode steps
  (`GEIST_MOE_ADAPT_WINDOW`, default 64) the model counts each layer's expert
  selections and times both branches of a decode layer: the CPU's cold
  experts on every step, the GPU's hot batch on two probe steps. Where the
  CPU branch is the longer one, it swaps the most selected cold experts for
  the least selected hot ones, up to the predicted balance of the two
  branches, at most `GEIST_MOE_ADAPT_SWAPS` (16) per window, one per decode
  step; a layer whose GPU branch is already the longer one keeps its
  placement. A swap trades one expert's GPU copy for another's, so device
  memory stays as placed. Only with one session on the model.
