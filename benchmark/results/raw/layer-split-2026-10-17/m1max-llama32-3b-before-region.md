model `llama-3.2-3b-Q4_K_M.gguf`, prompt 512, decode 64, 3 cycles x 3 repeats; median (min-max) over cycles

| configuration | prefill tok/s | decode tok/s | peak RSS MB | peak VRAM MB |
| :-- | --: | --: | --: | --: |
| geistlib cpu (cpu_neon) | 107.01 (101.38-108.58) | 35.05 (34.47-36.28) | 5694 | - |
| geistlib gpu (metal) | 876.40 (857.34-890.12) | 81.67 (81.17-82.94) | 614 | - |
| geistlib split 7 | 116.82 (116.46-119.82) | 17.91 (17.88-18.00) | 4660 | - |
| geistlib split 14 | 134.38 (132.04-135.82) | 12.11 (12.04-12.15) | 3556 | - |
| geistlib split 21 | 151.39 (150.42-155.78) | 9.05 (8.95-9.08) | 2447 | - |
