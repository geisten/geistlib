### Fixed
- **Audio-tower tests fail on a configured but unreadable tower (#57).**
  `GEIST_AUDIO_MODEL_PATH` pointing at a missing file, or the default path
  being a dangling symlink, made the tower parity tests SKIP, so a parity
  run reported "identical" without running. They now fail and say why; no
  tower configured still skips.
