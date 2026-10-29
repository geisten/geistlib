### Fixed
- **`tools/eval_decision_prefix.py` finalizes when consecutive subjects share an empty
  cut (#26).** The driver prepares one empty prefix for both subjects, so the
  second has no prepare record. The report then raised a KeyError. Such subjects
  now count a prepare of 0 ms and are marked `prefix_reused_from_previous_subject`.
