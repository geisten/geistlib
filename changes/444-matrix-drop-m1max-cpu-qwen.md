### Changed
- **The nightly benchmark matrix no longer runs qwen35-0.8b-q8 on the M1 Max CPU column (#444).**
  The cell failed the 5 % repeat-spread gate on the shared desktop in every
  try and is recorded as not measurable; the Mac mini M6 column measures the
  model on an Apple CPU.
