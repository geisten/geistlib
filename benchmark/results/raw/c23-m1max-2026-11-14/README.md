# M1 Max Metal, bonsai2-27b-pq2 at 6642260d (#444), 2026-11-14

The cell that failed the 5 % repeat-spread gate on every earlier try, measured
with a 55 °C start and a 15 % limit for this model on this host
(`model_host_gate` in `benchmark/cross_engine_gpu_protocol.json`).

- `apple_m1max_metal/bonsai2-27b-pq2/` — the recorded run (65 W adapter,
  battery 100 %, host 85–97 % idle). Four llama.cpp measurements were
  discarded at 19.4, 20.1, 15.9 and 20.0 % and measured again.
- `failed-5pct/` — the same cell with the 55 °C start and the unchanged 5 %
  limit: geist's repeats within 0.2 %, llama.cpp's runs at 4.4, 10.0, 10.5,
  11.0, 11.9, 14.6 and 25.4 %; stopped in cycle 2.
- `inv-llama2.log`, `inv-llama3.log` — single llama.cpp calls on a 30 W
  adapter at 7–10 % battery: in 7 of 14 calls at 256 and 512 tokens the GPU
  clock fell from 1293 to 388 MHz a few seconds in and stayed there at
  61–65 °C (prefill 33–45 instead of 107–148 tok/s).
- `inv-llama4.log` — the same calls on 65 W at 100 %: no such drop in 12
  calls, with Spotlight and media analysis daemons busy on the host.
- `trace_probe.py` — the script behind `inv-llama4.log` (macmon trace; paths
  are those of the measuring machine).
