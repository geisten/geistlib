### Changed
- **`<geist_arch.h>` handles have types of their own (#347).** The decoder
  vtable takes `struct geist_arch_state *` (the model) and
  `struct geist_arch_session *` (one stream), the decision rows a
  `struct geist_arch_readout *`, and the audio and vision encoder vtables
  `struct geist_audio_encoder_state *` and
  `struct geist_vision_encoder_state *`, in place of `void *`. The structs are
  declared and never defined; an architecture converts them to its own state
  types. Passing a session where the model belongs, or the audio encoder's
  state to the vision tower, is now a compile error, which
  `make check-handles` proves. The header is `EXPERIMENTAL`: the vtable
  layout and the ABI are unchanged, but an out-of-tree architecture has to
  update its function signatures (a mismatch is an incompatible function
  pointer at its descriptor). Machine code of the library is byte-identical.
