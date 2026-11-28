### Added
- **`fused->kv_append_q8` (`struct geist_kv_append_q8_args`): the quantizing
  append of the INT8 and INT4 KV caches as a backend kernel (#609).** For a
  backend whose `attention_kv_int8` / `attention_kv_int4` read the cache
  where the host cannot write it cheaply; the architecture calls it on the
  layers whose plan binds one of those two. Vulkan implements it; a backend
  that leaves it null keeps the architecture's host loop.
