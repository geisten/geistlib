### Fixed
- **qwen35moe: a shared expert is never dropped silently, nor does it keep
  the routed experts off the GPU (#169).** A GGUF with shared-expert
  tensors but no `expert_shared_feed_forward_length` loaded without its
  shared expert (wrong logits, no error); the tensors now decide too, with
  the experts' width. A shared expert the GPU cannot take (its dtype) ended
  the hot placement, and as it ranks first no expert went to the GPU; it
  now stays on the CPU alone. test_qwen35moe_unit also checks the shared
  gate's weight against a host reference, sigmoid(x · g) with a
  non-constant g (a sign error fails it; the dense twin's sigmoid(0) did
  not).
