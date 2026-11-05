### Fixed
- **SentencePiece vocabularies split malformed UTF-8 as llama.cpp does
  (#454).** llama.cpp's SentencePiece tokenizer does not decode the text:
  after replacing spaces with U+2581 it cuts it into symbols by the lead
  byte's high nibble alone (`unicode_len_utf8`, 8642541b2: 0x0..0xB one
  byte, 0xC..0xD two, 0xE three, 0xF four, clipped to the end), so a
  malformed sequence swallows the bytes after it, U+2581's included, and a
  symbol that forms no token becomes its `<0xXX>` byte tokens. geistlib's
  unigram and SPM paths split by decoded codepoints instead, a malformed
  byte as a symbol of its own. They now size symbols as llama.cpp does;
  on valid UTF-8 the split is the same (ids bit-identical on 1 MB of
  mixed-script text for llama-spm, phi-3, baichuan, Gemma 3, qwen2,
  gemma-4 and llama-bpe). On the 17 malformed texts of
  `test_utf8_malformed_unit` the ids now equal llama-tokenize 0.8.0's on
  llama.cpp's llama-spm, phi-3 and baichuan vocabularies and on Gemma 3
  270M (40 of 68 cases differed before); the test gains the SentencePiece
  split from a harness built on llama.cpp's own `unicode_len_utf8`, for the
  unigram and the merge path. tokenizer.bin keeps its own split (#263).
