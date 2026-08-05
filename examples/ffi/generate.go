// examples/ffi/generate.go — geistlib from Go via cgo. The header is the ABI:
// cgo reads it directly, nothing is generated or wrapped.
//
// Build (after building the shared library at the repo root, see README.md):
//
//	LD_LIBRARY_PATH=../.. go run generate.go model.gguf "prompt"
package main

/*
#cgo CFLAGS: -I${SRCDIR}/../../include
#cgo LDFLAGS: -L${SRCDIR}/../.. -lgeist
#include <geist.h>
#include <geist_util.h>
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"os"
	"unsafe"
)

func main() {
	model, prompt := C.CString(os.Args[1]), C.CString(os.Args[2])
	defer C.free(unsafe.Pointer(model))
	defer C.free(unsafe.Pointer(prompt))

	var be *C.struct_geist_backend
	var m *C.struct_geist_model
	var s *C.struct_geist_session
	if C.geist_backend_create(C.CString("auto"), nil, nil, &be) != 0 {
		panic("backend")
	}
	if C.geist_model_load(model, be, &m) != 0 {
		panic("model")
	}
	if C.geist_session_create(m, be, nil, &s) != 0 { // nil opts = greedy
		panic("session")
	}
	if C.geist_session_set_prompt(s, prompt) != 0 {
		panic("prompt")
	}

	eos := C.geist_model_eos_token(m)
	fmt.Print(os.Args[2])
	var tok C.geist_token_t
	for i := 0; i < 256; i++ {
		if C.geist_session_decode_step(s, &tok) != 0 || tok == eos {
			break
		}
		p := C.geist_session_token_to_str(s, tok)
		if p == nil {
			break
		}
		piece := C.GoString(p)
		if len(piece) >= 2 && piece[0] == '<' && piece[len(piece)-1] == '>' {
			break
		}
		fmt.Print(piece)
	}
	fmt.Println()
	C.geist_session_destroy(s)
	C.geist_model_destroy(m)
	C.geist_backend_destroy(be)
}
