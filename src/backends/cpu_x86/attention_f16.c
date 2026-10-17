/*
 * src/backends/cpu_x86/attention_f16.c — causal attention over the F16 KV
 * cache (prims->attention with F16 k / v views, #48).
 *
 * Layer: BACKEND (cpu_x86).
 *
 * attention.c's kernel compiled a second time with AF_F16: K and V are IEEE
 * half, widened eight at a time by F16C (vcvtph2ps, in the x86-64-v3
 * floor) as they are loaded; the arithmetic, the work items and the split
 * decode are the FP32 kernel's. One kernel source, two element types.
 */
#define AF_F16 1
#include "attention.c"
