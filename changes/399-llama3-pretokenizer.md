### Fixed
- **Llama 3 byte-BPE GGUFs tokenize as llama.cpp does (#399).** A
  `tokenizer.ggml.pre` of `llama-bpe` (also `llama3`, `llama-v3`,
  `falcon3`, `falcon-h1`, `pixtral`, `midm-2.0`, `lfm2`, `jina-v5-nano`)
  went through geist's simplified GPT-2 splitter: whole digit runs
  (" 2000" was one chunk, tokens " " "2" "000", instead of llama.cpp's
  " " "200" "0"), one byte per
  punctuation chunk (" ..." in three pieces) and the whole leading
  whitespace run glued to the next word. The wikitext-2 test text left
  `llama-tokenize` at token 39 on Llama 3.2 3B Instruct. These files now
  use the llama3 regex (`\p{N}{1,3}`, the qwen2 scanner's other rules) and
  llama.cpp's `ignore_merges` (a chunk that is a vocab entry is that one
  token; 588 of Llama 3's own vocab entries BPE-encode differently
  without it). The same scanner, shared with `qwen2` / `qwen35`, now uses
  llama.cpp's letter, number and whitespace classes instead of an
  approximation (² and ½ are numbers, U+0085 is whitespace, a combining
  mark is a letter only for `qwen35`), takes only an ASCII space as the
  lead of a punctuation run, and ends a whitespace run at the next special
  token. geist and llama-tokenize now agree on the whole wikitext-2 test
  text plus 2000 of its lines and 40000 random unicode / code / number /
  whitespace / contraction strings for Llama 3.2, Qwen3 (0.6B, 30B-A3B,
  Huihui MoE) and Qwen3.5 / 3.8 (before: 2371 of 5106 records identical for
  Llama 3.2, 4819 for Qwen3, 4861 for Qwen3.5); gpt-4o, llama4 and
  deepseek-llm were already identical and stay so.
