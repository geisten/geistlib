model `Qwen3-14B-Q8_0.gguf`, prompt 512, decode 64, 3 cycles x 3 repeats; median (min-max) over cycles
| configuration | prefill tok/s | decode tok/s | peak RSS MB | peak VRAM MB |
| :-- | --: | --: | --: | --: |
| geistlib cpu (cpu_x86) | 88.06 (81.87-88.33) | 5.18 (5.17-5.18) | 14401 | 365 |
| geistlib split 20 | 133.47 (132.26-133.67) | 7.80 (7.80-7.82) | 7895 | 7409 |
| geistlib split 26 | 167.97 (163.75-167.99) | 9.62 (9.61-9.63) | 5888 | 9432 |
| llama.cpp -ngl 0 | 245.82 (245.75-246.60) | 4.12 (4.12-4.12) | 15369 | 1518 |
| llama.cpp -ngl 20 | 392.12 (392.00-392.69) | 7.20 (7.18-7.21) | 15947 | 7918 |
| llama.cpp -ngl 26 | 485.21 (484.93-486.29) | 9.21 (9.18-9.21) | 15947 | 9939 |
| llama.cpp -ngl 0 -nopo 1 | 64.26 (64.25-64.54) | 5.12 (5.12-5.13) | 15585 | 377 |
| llama.cpp -ngl 20 -nopo 1 | 106.49 (106.31-107.82) | 7.38 (7.37-7.38) | 15947 | 7898 |
| llama.cpp -ngl 26 -nopo 1 | 144.52 (142.32-144.84) | 9.51 (9.48-9.53) | 15947 | 9919 |
