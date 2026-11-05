### Added
- **`GEIST_IMAGE_MAX_DIM` in `<geist_arch.h>` (#441).** The largest image
  height or width a vision entry point accepts (`INT_MAX / 3`, where the
  resizer's int row stride stops being representable). The engine checks it
  at the public entry without including a vision-tower header; the tower's
  own `IMAGE_PIPELINE_MAX_DIM` is now defined from it, same value.
