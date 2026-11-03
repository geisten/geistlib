### Fixed
- **SmolLM and the other GPT-2-regex byte-BPE GGUFs tokenize as llama.cpp
  does (#403).** A `tokenizer.ggml.pre` of `smollm` (SmolLM2, geist's CI
  fixture) went through geist's simplified GPT-2 splitter: one byte per
  punctuation chunk (`...` was three tokens, one in llama.cpp), the whole
  leading whitespace run glued to the next word and no contractions. The
  wikitext-2 test text left `llama-tokenize` at token 1203; 2920 of 5106
  parity records were identical. These files now run llama.cpp's GPT-2
  regex (`unicode_regex_split_custom_gpt2`), after its `\p{N}` split for
  the SMOLLM group (`smollm`, `starcoder`, `refact`, `command-r`,
  `codeshell`, `exaone`, `minerva-7b`, `mellum2`) and without it for the
  GPT2 group (`gpt-2`, `phi-2`, `mpt`, `olmo`, `jais`, `exaone4`,
  `trillion`, `granite-docling`, ...). geist and llama-tokenize now agree on
  the whole wikitext-2 test text, 2000 of its lines, 40103 edge-case and
  random strings and all 48821 normal vocab entries of SmolLM2. The prompts
  of the SmolLM2 goldens (`test_logit_golden_int`, "The capital of France
  is") tokenized the same before and after, so no golden changed.
  Tokenizing got faster on the way (single-codepoint chunks skip the BPE
  scratch, ASCII skips the UTF-8 decode, the special-token match is skipped
  until the next special): wikitext-2 test text on an M1 Max, SmolLM2 -7 %,
  Llama 3.2 -40 %, Qwen3 -15 %, Qwen3.5 -19 %.
