### Fixed
- **Vulkan refuses a device without arithmetic subgroup operations instead
  of crashing (#92).** The reduction shaders use `subgroupAdd` /
  `subgroupMax`; Mesa's V3DV (Raspberry Pi 5) does not offer them, and its
  compiler aborted the whole process when the pipelines were built, so
  every program of a Pi build with `BACKENDS="vulkan ..."` died at backend
  creation, `auto` included. Backend creation now fails with
  `GEIST_E_UNSUPPORTED` ("... has no arithmetic subgroup operations in
  compute shaders") and `auto` falls back to cpu_neon.
