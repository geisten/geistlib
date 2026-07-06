# Project report — geist on-device inference

Status: the BitNet 2B-4T model runs end-to-end on the Raspberry Pi 5 with
ternary TQ2_0 kernels. Decode is bandwidth-bound; the f16 tied lm_head
accounts for roughly half of each token. Prefill was parallelized across
the attention core, flattening the throughput curve at long sequences.

Next milestones: cut lm_head cost via an i8 sketch top-k, and make the
agent tool loop reliable on the untrained model.
