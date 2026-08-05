// examples/ffi/generate.js — geistlib from JavaScript via Bun's built-in FFI.
// No native addon, no node-gyp: the header is the ABI, dlopen is the binding.
//
// Build the shared library once (see README.md), then:
//
//     bun generate.js ../../libgeist.so model.gguf "The capital of France is"
import { dlopen, FFIType, ptr, CString } from "bun:ffi";

const [libPath, model, prompt] = Bun.argv.slice(2);
const { symbols: g } = dlopen(libPath, {
  geist_version_string:       { args: [], returns: FFIType.cstring },
  geist_backend_create:       { args: ["cstring", "ptr", "ptr", "ptr"], returns: FFIType.i32 },
  geist_model_load:           { args: ["cstring", "ptr", "ptr"], returns: FFIType.i32 },
  geist_session_create:       { args: ["ptr", "ptr", "ptr", "ptr"], returns: FFIType.i32 },
  geist_session_set_prompt:   { args: ["ptr", "cstring"], returns: FFIType.i32 },
  geist_session_decode_step:  { args: ["ptr", "ptr"], returns: FFIType.i32 },
  geist_session_token_to_str: { args: ["ptr", "i32"], returns: FFIType.cstring },
  geist_model_eos_token:      { args: ["ptr"], returns: FFIType.i32 },
  geist_session_destroy:      { args: ["ptr"], returns: FFIType.void },
  geist_model_destroy:        { args: ["ptr"], returns: FFIType.void },
  geist_backend_destroy:      { args: ["ptr"], returns: FFIType.void },
});

const cstr = (s) => Buffer.from(s + "\0", "utf8");
const out = new BigUint64Array(1);
const deref = () => Number(out[0]);

console.error(`geistlib ${g.geist_version_string()}`);
if (g.geist_backend_create(cstr("auto"), null, null, ptr(out)) !== 0) throw "backend";
const be = deref();
if (g.geist_model_load(cstr(model), be, ptr(out)) !== 0) throw "model";
const m = deref();
if (g.geist_session_create(m, be, null, ptr(out)) !== 0) throw "session"; // null opts = greedy
const s = deref();
if (g.geist_session_set_prompt(s, cstr(prompt)) !== 0) throw "prompt";

const eos = g.geist_model_eos_token(m);
process.stdout.write(prompt);
const tok = new Int32Array(1);
for (let i = 0; i < 256; i++) {
  if (g.geist_session_decode_step(s, ptr(tok)) !== 0 || tok[0] === eos) break;
  const piece = g.geist_session_token_to_str(s, tok[0]);
  if (piece == null) break;
  const str = piece.toString();
  if (str.startsWith("<") && str.endsWith(">")) break;
  process.stdout.write(str);
}
process.stdout.write("\n");
g.geist_session_destroy(s); g.geist_model_destroy(m); g.geist_backend_destroy(be);
