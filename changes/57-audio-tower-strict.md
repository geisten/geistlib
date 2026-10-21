### Fixed
- **Every tower-gated audio test names what it missed, and fails where CI
  fetched the tower (#57).** The session-driven audio tests
  (`test_audio_attach_int`, `test_chat_audio_int`,
  `test_audio_stream_lifecycle_int`, `test_audio_stream_wav_int`, the
  "cannot hear" branch of `test_audio_stream_session_int`) turned "no
  encoder" into a plain SKIP, also with a dead `audio_bench/` link. They now
  take the dead-link check and skip naming the expected path (or the present
  tower and the session's error). `GEIST_STRICT_FIXTURES` takes an `audio`
  class that turns such a skip into a failure; the coverage, audio-smoke and
  x86_64 audio-parity CI legs, which fetch the tower, set it.
