### Fixed
- **Malformed UTF-8 tokenizes as llama.cpp tokenizes it (#427).** A byte
  that starts no valid sequence (a stray continuation byte, a truncated,
  overlong or past-U+10FFFF sequence, 0xF8..0xFF) is now U+FFFD and takes
  one byte, as in llama.cpp's `unicode_cpts_from_utf8` (8642541b2); before,
  the byte-level BPE scanners read it as codepoint 0xFFFFFFFF ("other"
  instead of llama.cpp's symbol, so deepseek-v3 split "\xFFabc" as one piece
  where llama.cpp splits it in two) and the encoder merged the raw byte
  where llama.cpp merges U+FFFD's three bytes, so every byte-BPE vocabulary
  and Gemma 4 gave other ids for such text. SentencePiece ("llama") and
  unigram vocabularies keep the raw bytes, as llama.cpp does. On 17
  malformed texts the ids now equal llama-tokenize 0.8.0's on llama.cpp's
  gpt-2, qwen2, qwen35, llama-bpe, deepseek-llm, starcoder, refact,
  command-r, mpt and gemma-4 vocabularies and on gpt-oss-20b, Llama 4 Scout
  and DeepSeek-V2-Lite (208 of these 221 cases differed before); falcon and
  deepseek-coder, which run the simple splitter, differ only where they do
  on valid text. `test_utf8_malformed_unit` pins the decoder, the splits of
  every scanner against llama.cpp's `unicode_regex_split` and the encoded
  bytes.
