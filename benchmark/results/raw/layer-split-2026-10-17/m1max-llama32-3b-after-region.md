model `llama-3.2-3b-Q4_K_M.gguf`, prompt 512, decode 64, 3 cycles x 3 repeats; median (min-max) over cycles

| configuration | prefill tok/s | decode tok/s | peak RSS MB | peak VRAM MB |
| :-- | --: | --: | --: | --: |
| geistlib cpu (cpu_neon) | 104.45 (104.42-107.30) | 35.66 (35.57-36.09) | 5694 | - |
| geistlib gpu (metal) | 879.93 (879.57-917.91) | 81.67 (81.19-83.75) | 615 | - |
| geistlib split 7 | 131.71 (130.56-133.28) | 37.30 (36.75-37.32) | 4660 | - |
| geistlib split 14 | 182.75 (182.44-183.20) | 39.88 (39.51-39.95) | 3555 | - |
| geistlib split 21 | 293.07 (292.99-294.62) | 47.37 (46.92-47.60) | 2447 | - |
