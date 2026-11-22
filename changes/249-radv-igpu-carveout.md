### Fixed
- **Vulkan: Qwen3.6-35B-A3B loads on an integrated Radeon (RADV) instead of
  failing or hanging the GPU (#249).** On the Ryzen 9950X's integrated GPU
  the 22.7 GB model stopped loading with `radv/amdgpu: Not enough memory for
  command submission` and `vulkan: transfer submit/wait failed`; earlier runs
  hung in the kernel (`amdgpu_vm_bo_update`) until a reboot. RADV places its
  device-local memory types in the firmware's VRAM carve-out first (2 GiB
  there). The kernel keeps the process's GPU page tables in the carve-out
  too, 4 KiB per 2 MiB mapped, and does not move the process's own buffers
  out for them. Each weight is a buffer of its own, so once no hole of a
  weight's size was left, the page tables used up the holes: run 3998 shows
  the free carve-out falling by 20 KiB per 16 weights of 576-704 KiB, none
  of which landed there, from 8.5 MiB to 128 KiB at 6.3 GiB allocated.
  Qwen3-30B-A3B ended its load with about 5 MiB left. On RADV with an
  integrated GPU the backend now reads the carve-out's size and use from
  the kernel driver and allocates from the device-local types only up to
  what is free less a reserve for the page tables (an eighth, at least
  128 MiB); past that, from a memory type outside the device-local heap
  (`vk_alloc_mem_type`): the same system memory, never in the carve-out.
  Limits and placement still count against the device-local heap.
  `GEIST_VK_CARVEOUT=<bytes>` sets the budget; other drivers and discrete
  GPUs are untouched. The 35B then loads with 9741 of 10280 experts on the
  GPU and 231 MiB of the carve-out free (run 4007), prefills 512 tokens at
  16.2 tok/s and decodes at 9.10 tok/s (llama.cpp Vulkan on the same GPU:
  38.1 / 8.84), perplexity 18.10 against 18.23 on cpu_x86 (run 4009). OLMoE-1B-7B and
  Qwen3-30B-A3B decode as before (27.9 and 10.05 tok/s against 27.9 and
  10.05 with the old placement); with every allocation kept out of the
  carve-out they lost 3.9 % and 3.3 % (run 4005).
  `test_backend_vulkan_vram_budget_unit` checks the choice of type on that
  GPU's memory types without a device.
