### Changed
- **A Metal handle of the wrong type no longer compiles (#347).** The
  backend's Objective-C objects (device, queue, libraries, functions,
  pipelines, MTLBuffers, command buffers, encoders, residency set, pools)
  are typed opaque struct pointers instead of `void *`, so binding a buffer
  as a pipeline or a command buffer as an encoder is a compile error, and the
  pipeline tables check each row's field type. `make check-handles` (part of
  `make check-static`) proves five such mistakes fail. Machine code is
  unchanged: every Metal object file is byte-identical to the one before.
