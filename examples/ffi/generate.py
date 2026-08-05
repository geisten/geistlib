#!/usr/bin/env python3
"""examples/ffi/generate.py — geistlib from Python, no bindings package.

The header is the ABI: this is the whole integration, ~30 lines of ctypes.
Build the shared library once (see README.md in this directory), then:

    python3 generate.py ../../libgeist.so model.gguf "The capital of France is"
"""
import ctypes, sys

lib, model, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
g = ctypes.CDLL(lib)
g.geist_version_string.restype = ctypes.c_char_p
g.geist_session_token_to_str.restype = ctypes.c_char_p
print(f"geistlib {g.geist_version_string().decode()}", file=sys.stderr)

be, m, s = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
assert g.geist_backend_create(b"auto", None, None, ctypes.byref(be)) == 0
assert g.geist_model_load(model.encode(), be, ctypes.byref(m)) == 0
assert g.geist_session_create(m, be, None, ctypes.byref(s)) == 0  # NULL opts = greedy
assert g.geist_session_set_prompt(s, prompt.encode()) == 0

eos = g.geist_model_eos_token(m)
print(prompt, end="", flush=True)
tok = ctypes.c_int32()
for _ in range(256):
    if g.geist_session_decode_step(s, ctypes.byref(tok)) != 0 or tok.value == eos:
        break
    piece = g.geist_session_token_to_str(s, tok)
    if piece is None or (piece.startswith(b"<") and piece.endswith(b">")):
        break
    sys.stdout.buffer.write(piece); sys.stdout.flush()
print()
g.geist_session_destroy(s); g.geist_model_destroy(m); g.geist_backend_destroy(be)
