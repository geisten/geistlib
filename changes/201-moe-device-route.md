### Added
- **A mixture of experts decodes on Metal without the host (#201).** With
  every routed expert of a layer on the GPU (unified memory, #168) a decode
  step routes there too: `fused->moe_route` picks the top-k of the router
  logits on the device (ties to the lower id, as the host's
  `moe_route_topk`; a non-finite logit is reported after the step),
  `fused->linear_t_ids` runs gate, up and down of the chosen experts by
  their ids (Q4_K, Q6_K, Q8_0, F32), and `fused->moe_weighted_sum` adds
  their outputs onto the residual. The layer no longer waits for the host
  twice. New EXPERIMENTAL fused ops in `geist_backend.h`
  (`GEIST_FUSED_MOE_DEVICE`); Metal implements them. `GEIST_MOE_DEVICE=0`
  keeps the host route. Not yet for a model with a shared expert, and not
  at prefill.
