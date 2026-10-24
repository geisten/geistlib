### Changed
- **A speculative verify costs about one prefill pass of its rows (#192).**
  - `verify_forward` runs in one command batch, as prefill and decode do.
    Metal used to submit and wait on every op.
  - Greedy rows take the device argmax. Mapping k × vocab logits read them
    over the bus on a discrete GPU, ~32 ms per row on an RTX 2080 Ti.
  - A partial accept no longer replays the accepted rows. On Metal and
    Vulkan the DeltaNet kernels keep the state after every verified row
    (`geist_deltanet_mix_args.row_*`), and the rollback copies one slot back.
  - The DeltaNet transaction, row-slot and checkpoint buffers are
    device-local.
  - On Vulkan, `mm_pq2_0_m16_cm` covers 32 rows with 8 k-slices and shares
    one token tile between two weight tiles: a 2..8-row pass of Bonsai 2 27B
    costs 47–55 instead of 58–62 ms of GPU time.
  - Verify + rollback + correction for 6 drafts:
    - M1 Max: 188–194 → 179–184 ms;
    - RTX 2080 Ti: 543 → 98 ms.
