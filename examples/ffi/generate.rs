//! examples/ffi/generate.rs — geistlib from Rust, no bindgen, no crate.
//!
//! Hand-written extern "C" declarations for the STABLE core — the header is
//! the ABI. Build (after building the shared library, see README.md):
//!
//!     rustc -O generate.rs -L ../.. -lgeist -o generate_rs
//!     ./generate_rs model.gguf "The capital of France is"
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};

#[link(name = "geist")]
extern "C" {
    fn geist_backend_create(name: *const c_char, opts: *const c_void, alloc: *const c_void, out: *mut *mut c_void) -> c_int;
    fn geist_model_load(path: *const c_char, be: *mut c_void, out: *mut *mut c_void) -> c_int;
    fn geist_session_create(m: *mut c_void, be: *mut c_void, opts: *const c_void, out: *mut *mut c_void) -> c_int;
    fn geist_session_set_prompt(s: *mut c_void, prompt: *const c_char) -> c_int;
    fn geist_session_decode_step(s: *mut c_void, tok: *mut i32) -> c_int;
    fn geist_session_token_to_str(s: *mut c_void, tok: i32) -> *const c_char;
    fn geist_model_eos_token(m: *const c_void) -> i32;
    fn geist_session_destroy(s: *mut c_void);
    fn geist_model_destroy(m: *mut c_void);
    fn geist_backend_destroy(be: *mut c_void);
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model = CString::new(args.next().expect("usage: generate_rs <model.gguf> <prompt>")).unwrap();
    let prompt = CString::new(args.next().expect("missing prompt")).unwrap();

    let auto = CString::new("auto").unwrap();
    unsafe {
        let (mut be, mut m, mut s) = (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
        assert_eq!(geist_backend_create(auto.as_ptr(), std::ptr::null(), std::ptr::null(), &mut be), 0);
        assert_eq!(geist_model_load(model.as_ptr(), be, &mut m), 0);
        assert_eq!(geist_session_create(m, be, std::ptr::null(), &mut s), 0); // NULL opts = greedy
        assert_eq!(geist_session_set_prompt(s, prompt.as_ptr()), 0);

        let eos = geist_model_eos_token(m);
        print!("{}", prompt.to_str().unwrap());
        let mut tok: i32 = 0;
        for _ in 0..256 {
            if geist_session_decode_step(s, &mut tok) != 0 || tok == eos { break; }
            let p = geist_session_token_to_str(s, tok);
            if p.is_null() { break; }
            let piece = CStr::from_ptr(p).to_string_lossy();
            if piece.starts_with('<') && piece.ends_with('>') { break; }
            print!("{piece}");
            use std::io::Write; std::io::stdout().flush().unwrap();
        }
        println!();
        geist_session_destroy(s); geist_model_destroy(m); geist_backend_destroy(be);
    }
}
