### Fixed
- **`geist_session_get_stats` no longer counts image and video encoding as
  audio.** `attach_image` and `attach_video` added their encoder time to
  `total_audio_encode_ns`; it now goes to the new
  `total_vision_encode_ns`, appended at the end of `struct
  geist_session_stats` (EXPERIMENTAL) so the fields before it keep their
  offsets. `total_audio_encode_ns` counts audio only.
