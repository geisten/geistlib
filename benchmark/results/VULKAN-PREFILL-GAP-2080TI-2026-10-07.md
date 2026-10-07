# Vulkan-Prefill-Lücke gegenüber llama.cpp — Ursachenanalyse (RTX 2080 Ti)

Folgeuntersuchung zu [MAIN-VS-V011-AMD9950X-2026-10-07.md](MAIN-VS-V011-AMD9950X-2026-10-07.md). Dort lag geist main beim pp512 auf Vulkan um −70 % (Gemma 4 E2B) bis −91 % (Qwen3.5 4B) hinter llama.cpp, bei Bonsai 2 27B nur um −12 %. Dieser Report ordnet die Lücke einzelnen Ursachen zu.

geist main `1b5380a`, llama.cpp `2d8d612e4` (Bonsai: PrismML-Fork `adfffbe41`). Beide Engines nutzen auf der 2080 Ti `KHR_coopmat` mit fp16, ohne int-dot. Die Hardware-Fähigkeiten sind also identisch.

## Ergebnis

Die Lücke hat drei Haupt- und zwei Nebenursachen:
- **Hauptursachen:**
  - Quant-Typen ohne Tensor-Core-GEMM (Q4_0/Q4_1/Q5_K/Q8_0, 5–11× langsamer als llama.cpp bei gleicher Batch-Größe).
  - Die feste Prefill-Chunkgröße von 64 Tokens (llama.cpp gewinnt durch 512er-Batches +57 bis +116 %; geist kann wegen #488 nicht über 128 gehen).
  - Die Effizienz der vorhandenen Q4_K/Q6_K-Tensor-Core-Kernels (1,3–2,4× langsamer).
- **Nebenursachen:**
  - Die Tensor-Core-Attention greift nur bei head_dim 256 ohne Sliding Window.
  - DeltaNet.

Bonsai bestätigt die Methode: Bei gleicher Batch-Größe ist geist dort **schneller** als llama.cpp (Faktor 1,22). Die −12 % im Gesamtvergleich stammen allein von der Chunkgröße.

## Gesamtdurchsatz bei gleicher Batch-Größe (pp512, tok/s, mit Profiler)

| Modell | geist M=64 | geist M=128 | llama ub=64 | llama ub=128 | llama ub=512 | geist/llama @64 | llama-Gewinn 64→512 |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Gemma 4 E2B Q4_K_M | 1328 | 1602 | 2302 | 2924 | 4550 | 0,58 | 2,0× |
| Qwen3.5 4B Q4_0 | 251 | 260 | 1759 | 2200 | 2808 | 0,14 | 1,6× |
| Qwen3 0.6B Q8_0 | 1814 | 1687 | 6697 | 6757 | 14397 | 0,27 | 2,1× |
| Bonsai 2 27B PQ2_0 | 375 | 501 | 307 | 484 | 538 | **1,22** | 1,75× |

geist bleibt bei M ≥ 256 nicht messbar. Bei Gemma fällt der Durchsatz auf ~360 tok/s, weil ein zweiter Scratch-Puffer (99 MiB bei M=256, 166 MiB bei M=512) im Hauptspeicher statt im VRAM landet, auch mit `GEIST_VK_SCRATCH_DEVICE=1`. Siehe Ursache E.

Die Summe der GPU-Zeitstempel (Gemma M=64: 389 ms) deckt sich mit der gemessenen Prefill-Zeit (385 ms). geist ist also vollständig GPU-gebunden; Host-Overhead und Dispatch-Anzahl spielen keine Rolle.

## Rangliste der Ursachen

Erwarteter pp512-Gewinn, wenn **nur** diese Ursache auf llama.cpp-Niveau bei gleicher Batch-Größe käme (Amdahl auf die geist-GPU-Zeit bei M=64). Die Gewinne addieren sich nicht. E misst den Batch-Gewinn mit llama.cpps Kernels; mit geists skalaren Kernels bringt M=128 heute nichts (Qwen-Zeilen oben).

| Rang | Ursache | Gemma 4 E2B | Qwen3.5 4B | Qwen3 0.6B | Bonsai 27B | Aufwand | Tracking |
| :-- | :-- | --: | --: | --: | --: | :-: | :-- |
| 1 | **A** Kein Tensor-Core-GEMM für Q4_0/Q4_1/Q5_K/Q8_0 | — | **+583 %** | **+156 %** | — | L | #467 |
| 2 | **E** Chunkgröße 64 statt 512 (blockiert durch Scratch-Einbruch) | +94 % | +57 % | +116 % | +74 % | M | #488 |
| 3 | **B** Q4_K/Q6_K-Tensor-Core-Kernels 1,3–2,4× langsamer | **+46 %** | — | — | — | M | #658 |
| 4 | **C** Tensor-Core-Attention nur für head_dim 256 ohne Sliding Window | +11 % | +0 % | +15 % | +1 % | M | #475 |
| 5 | **D** DeltaNet-Kernels (`dn_delta` 2,3–2,8× langsamer) | — | +1 % | — | +4 % | S | #467 |
| 6 | **F** f32/bf16-GEMM (`matmul_f32`, Gemma PLE/Projektionen) 3,5–3,8× | +4 % | — | — | — | S | — |

### A — Quant-Typen ohne Tensor-Core-GEMM

`vk_linear_cm_route` (`src/backends/vulkan/ops.c:674`) leitet nur Q4_K, Q6_K und PQ2_0 auf `KHR_coopmat`-Kernels um. Q4_0, Q4_1, Q5_K und Q8_0 laufen über die skalaren `matmul_*`-GEMMs und erreichen ~2 TFLOPS, llama.cpp ~17 TFLOPS bei derselben Form. Bei Qwen3.5 4B sind das 1968 von 2038 ms GPU-Zeit. Ein größerer Chunk hilft diesen Kernels nicht (M=64 → 128: 798 → 797 ms für 9216×2560): Sie sind durch Dequantisierung und Rechenwerk begrenzt, nicht durch das Lesen der Gewichte.

### E — Chunkgröße und Scratch-Platzierung

llama.cpp halbiert seine GPU-Zeit beim Schritt von ub=64 auf ub=512 (Gemma 211 → 109 ms). geist läuft standardmäßig mit M=64. M=128 bringt bei Gemma +21 % und bei Bonsai +34 %, ab M=256 bricht geist ein. Ursache: Bei Gemma fordert ein zweiter Scratch-Puffer keinen reinen VRAM an und landet nach dem 256-MB-BAR-Bereich im Hauptspeicher. `GEIST_VK_SCRATCH_DEVICE=1` (#488) ändert daran nichts, weil die Arch für diesen Puffer `GEIST_MEMORY_DEVICE` nicht setzt (`vk_buffer_create_api`, `src/backends/vulkan/resources.c`). Erst der Fix für #488 macht E nutzbar. Er verstärkt außerdem A und B, weil Tensor-Core-Kacheln mit großem M besser ausgelastet sind.

### B — Effizienz der Q4_K/Q6_K-Tensor-Core-Kernels (Gemma)

Bei gleicher Batch-Größe erreicht geist 4–13 TFLOPS, llama.cpp 10–21. Am schlechtesten stehen die Down-Projektionen da (1536×12288 und 1536×6144, 2,3–2,4×). Dort wählt das Routing wegen `n_out < 4096` die 32×32-Kachel (`mm_q4k_cm32`) bzw. `mm_q6k_cm`. Kachelgröße und Split-K für schmale, tiefe Formen sind der naheliegende Ansatz.

### C — Attention

`attention_f16_cm` greift nur bei `hd == 256 && sliding_window == 0` (`src/backends/vulkan/ops.c:1324`). Gemma (Sliding-Window-Schichten, globale Schichten mit head_dim 512) und Qwen3 (head_dim 128) nutzen deshalb den skalaren `attention_f16`. Gemma: 56,6 ms gegen 18,0 ms bei llama.cpp. Qwen3 0.6B: 46,1 ms gegen 9,7 ms.

## Formabgleich pro GEMM (geist vs. llama.cpp, ms GPU-Zeit für pp512)

N×K = Ausgabezeilen × Eingabebreite. Bei Gemma Q4_K_M teilen sich Q4_K- und Q6_K-Tensoren dieselbe Form, die Typ-Spalte zeigt dann beide. TFLOPS sind auf die GEMM-Arbeit bei M=64 bezogen. Zum Vergleich: Das fp16-Tensor-Core-Maximum der 2080 Ti liegt laut NVIDIA-Datenblatt bei ~107 TFLOPS (fp16-Akkumulation) bzw. ~54 TFLOPS (fp32-Akkumulation).

### gemma4-e2b — pp512, ms GPU-Zeit (geist M=64 Summe 388.9 ms; llama ub64 210.7 ms, ub512 108.9 ms)

| N×K | Typ | geist-Kernel | geist M64 | geist M128 | llama ub64 | llama ub512 | Faktor @64 | TFLOPS geist / llama @64 | Anteil geist |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: |
| 1536×12288 | q4_K/q6_K | mm_q4k_cm32/mm_q6k_cm | 93.5 | 65.9 | 39.1 | 20.2 | 2.39× | 4.1 / 9.9 | 24 % |
| 12288×1536 | q4_K | mm_q4k_cm | 60.6 | 58.7 | 37.1 | 22.5 | 1.63× | 12.8 / 20.8 | 16 % |
| 1536×6144 | q4_K/q6_K | mm_q4k_cm32/mm_q6k_cm | 34.2 | 22.6 | 15.1 | 8.6 | 2.27× | 4.2 / 9.6 | 9 % |
| 6144×1536 | q4_K | mm_q4k_cm | 27.1 | 22.5 | 18.7 | 10.9 | 1.45× | 10.7 / 15.5 | 7 % |
| 256×1536 | f32/q4_K/q6_K | matmul_f32/mm_q4k_cm32/mm_q6k_cm | 20.6 | 15.2 | 25.3 | 7.8 | 0.81× | 1.2 / 0.9 | 5 % |
| 1536×2048 | q4_K | mm_q4k_cm32 | 17.8 | 15.6 | 9.7 | 5.2 | 1.84× | 5.1 / 9.3 | 5 % |
| 2048×1536 | q4_K | mm_q4k_cm32 | 14.9 | 15.4 | 11.7 | 3.8 | 1.27× | 6.0 / 7.7 | 4 % |
| 1536×256 | f32 | matmul_f32 | 12.4 | 11.1 | 3.3 | 1.0 | 3.82× | 1.1 / 4.3 | 3 % |
| 1536×4096 | q4_K | mm_q4k_cm32 | 8.7 | 7.5 | 4.4 | 2.5 | 1.97× | 5.2 / 10.2 | 2 % |
| 8960×1536 | bf16 | matmul_f32 | 8.1 | 7.8 | 2.3 | 1.4 | 3.54× | 1.7 / 6.2 | 2 % |
| 4096×1536 | q4_K | mm_q4k_cm | 4.6 | 3.5 | 3.2 | 1.3 | 1.42× | 9.8 / 13.9 | 1 % |
| 512×1536 | q4_K/q6_K | mm_q4k_cm32/mm_q6k_cm | 2.5 | 1.3 | 2.5 | 0.8 | 1.02× | 1.9 / 2.0 | 1 % |

Nicht-GEMM (geist M64 → llama ub64): attention_f16 56.6; rmsnorm_add 7.2; gelu_mul 4.4; ffn_norm_gu 4.3; rmsnorm 3.6; embed 3.1; qkv_prep 1.4; qkv_prep_f16 1.4; scale 1.1; ple_gate 0.8  ‖  FLASH_ATTN_EXT 18.0; RMS_NORM_MUL 6.0; GLU: 3.7; RMS_NORM_MUL_ROPE 3.1; ADD: 2.9; MUL: 2.2; RMS_NORM_MUL_ROPE_VIEW_SET_ROWS 0.7; GELU: 0.6

### qwen35-4b — pp512, ms GPU-Zeit (geist M=64 Summe 2,038.4 ms; llama ub64 278.7 ms, ub512 178.0 ms)

| N×K | Typ | geist-Kernel | geist M64 | geist M128 | llama ub64 | llama ub512 | Faktor @64 | TFLOPS geist / llama @64 | Anteil geist |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: |
| 9216×2560 | q4_0 | matmul_q4_0 | 797.6 | 797.4 | 88.7 | 51.8 | 9.00× | 1.9 / 17.4 | 39 % |
| 2560×9216 | q4_0/q4_1 | matmul_q4_0/matmul_q4_1 | 410.7 | 394.9 | 43.4 | 33.2 | 9.46× | 1.9 / 17.8 | 20 % |
| 8192×2560 | q4_0 | matmul_q4_0 | 353.1 | 354.8 | 32.7 | 20.1 | 10.79× | 1.9 / 21.0 | 17 % |
| 2560×4096 | q4_0/q5_K | matmul_q4_0/matmul_q5k | 190.9 | 179.8 | 25.4 | 15.5 | 7.52× | 1.8 / 13.5 | 9 % |
| 4096×2560 | q4_0 | matmul_q4_0 | 140.9 | 136.2 | 19.4 | 7.3 | 7.25× | 1.8 / 13.3 | 7 % |
| 32×2560 | q8_0 | matmul_q8_0 | 45.2 | 22.3 | 9.8 | 1.8 | 4.63× | 0.1 / 0.4 | 2 % |
| 1024×2560 | q4_0 | matmul_q4_0 | 25.3 | 24.2 | 4.2 | 1.9 | 6.10× | 1.7 / 10.3 | 1 % |

Nicht-GEMM (geist M64 → llama ub64): dn_delta 38.4; attention_f16_cm 12.5; dn_conv 8.3; rmsnorm 5.6; silu_mul 3.3; add 2.9; embed 1.4; rope 0.5  ‖  GATED_DELTA_NET: 13.5; RMS_NORM_MUL 8.3; L2_NORM: 5.4; FLASH_ATTN_EXT 4.5; SSM_CONV_SILU 3.7; ADD: 3.4; GLU: 3.1; CONCAT: 2.9; GET_ROWS: 2.3; MUL: 2.1; CPY: 1.8; SILU: 1.1; ROPE: 0.7; SIGMOID: 0.7

### qwen3-0.6b — pp512, ms GPU-Zeit (geist M=64 Summe 278.7 ms; llama ub64 72.0 ms, ub512 33.3 ms)

| N×K | Typ | geist-Kernel | geist M64 | geist M128 | llama ub64 | llama ub512 | Faktor @64 | TFLOPS geist / llama @64 | Anteil geist |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: |
| 3072×1024 | q8_0 | matmul_q8_0 | 89.4 | 103.7 | 15.6 | 5.9 | 5.72× | 2.0 / 11.5 | 32 % |
| 1024×3072 | q8_0 | matmul_q8_0 | 37.6 | 44.3 | 6.7 | 3.4 | 5.63× | 2.4 / 13.5 | 13 % |
| 1024×1024 | q8_0 | matmul_q8_0 | 32.8 | 34.6 | 14.5 | 4.5 | 2.26× | 1.8 / 4.1 | 12 % |
| 2048×1024 | q8_0 | matmul_q8_0 | 31.0 | 32.8 | 7.5 | 2.4 | 4.15× | 1.9 / 8.1 | 11 % |
| 1024×2048 | q8_0 | matmul_q8_0 | 28.9 | 29.7 | 5.7 | 2.7 | 5.10× | 2.1 / 10.6 | 10 % |

Nicht-GEMM (geist M64 → llama ub64): attention_f16 46.1; rmsnorm 5.1; rope 1.8; add 1.8; embed 1.3; silu_mul 1.2; kv_append_f16 0.8; scale 0.8  ‖  FLASH_ATTN_EXT 9.7; RMS_NORM_MUL_ROPE 4.1; RMS_NORM_MUL_ROPE_VIEW_SET_ROWS 2.4; RMS_NORM_MUL 1.9; ADD: 1.6; GLU: 1.6; SET_ROWS: 0.7

### bonsai2-27b — pp512, ms GPU-Zeit (geist M=64 Summe 1,402.0 ms; llama ub64 1,644.8 ms, ub512 946.6 ms)

| N×K | Typ | geist-Kernel | geist M64 | geist M128 | llama ub64 | llama ub512 | Faktor @64 | TFLOPS geist / llama @64 | Anteil geist |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: |
| 17408×5120 | pq2_0 | mm_pq2_0_cm64 | 347.2 | 283.9 | 451.7 | 284.6 | 0.77× | 33.6 / 25.9 | 25 % |
| 5120×17408 | pq2_0 | mm_pq2_0_cm64 | 345.4 | 220.0 | 355.0 | 221.5 | 0.97× | 16.9 / 16.5 | 25 % |
| 5120×6144 | pq2_0 | mm_pq2_0_cm64 | 126.0 | 83.2 | 117.9 | 71.4 | 1.07× | 16.4 / 17.5 | 9 % |
| 10240×5120 | pq2_0 | mm_pq2_0_cm64 | 117.0 | 103.7 | 122.7 | 73.2 | 0.95× | 22.0 / 21.0 | 8 % |
| 6144×5120 | pq2_0 | mm_pq2_0_cm64 | 79.0 | 52.7 | 82.6 | 47.0 | 0.96× | 19.6 / 18.7 | 6 % |
| 1024×5120 | pq2_0 | mm_pq2_0_cm64 | 50.6 | 32.3 | 12.3 | 7.2 | 4.10× | 3.4 / 13.9 | 4 % |
| 48×5120 | bf16 | matmul_f32 | 49.2 | 26.1 | 273.1 | 37.4 | 0.18× | 0.5 / 0.1 | 4 % |
| 12288×5120 | pq2_0 | mm_pq2_0_cm64 | 39.5 | 34.2 | 42.8 | 26.8 | 0.92× | 26.1 / 24.1 | 3 % |

llama-GEMMs ohne geist-Gegenstück: f32 1024×1024 (17.3 ms)

Nicht-GEMM (geist M64 → llama ub64): dn_delta 90.6; hadamard 42.3; rmsnorm 29.6; dn_conv 27.1; attention_f16_cm 25.1; silu_mul 17.2; add 7.9; embed 1.7; qgate_split 1.6; attention_f16 1.3; sigmoid_mul 1.2; rope 1.1; scale 0.7; kv_append_f16 0.6  ‖  GATED_DELTA_NET: 39.1; RMS_NORM_MUL 24.0; MUL: 23.7; GLU: 15.9; FLASH_ATTN_EXT 11.6; L2_NORM: 10.3; SSM_CONV_SILU 9.2; ADD: 8.6; CONCAT: 7.3; GET_ROWS: 6.5; CPY: 4.4; CONT: 3.1; ROPE: 1.7; SIGMOID: 1.6; SOFTPLUS: 0.9; SCALE: 0.8; SET_ROWS: 0.8

## Methodik

- **geist:** `bench_perf_sweep` pp512, `--warmup 64`, `GEIST_M_MAX=64/128`, `GEIST_VK_PROFILE=1`. Dazu ein lokaler, nicht committeter Patch ([`profile_shapes.patch`](raw/vulkan-prefill-gap-2026-10-07/profile_shapes.patch)), der pro Dispatch Pipeline, n_in, n_out und rows mit der Zeitstempel-Differenz ausgibt (`GEIST_VK_PROFILE_SHAPES=1`). Der Warmup-Chunk wird pro Form exakt abgezogen, Nicht-GEMM-Ops anteilig (512/576).
- **llama.cpp:** `llama-bench -p 512 -n 0 -r 2 -ub 64/128/512 -ngl 99` mit `GGML_VK_PERF_LOGGER=1`. Ausgewertet werden die letzten 512/ub Graph-Blöcke, also eine gemessene Wiederholung.
- **Zuordnung:** Die GEMMs werden über (N, K) gepaart. Prefill heißt rows > 1; Matvec und Decode sind ausgeschlossen.
- **Fehlerbild:** Einzelläufe auf einem Host ohne Quiet-Gate. Die Faktoren pro Kernel liegen weit über dem Rauschen von ±2 %; für Gesamtdurchsatz-Vergleiche gelten die Protokollzahlen aus dem Vorgänger-Report.
- **Reproduktion:** Die Rohdaten (stderr/JSONL beider Engines, 20 Läufe), das Sammelskript und die Auswertung liegen in [`raw/vulkan-prefill-gap-2026-10-07/`](raw/vulkan-prefill-gap-2026-10-07/). `tar xJf profiles.tar.xz && python3 prof_analyze.py` erzeugt die Tabellen oben neu.
