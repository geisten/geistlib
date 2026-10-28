/* tools/tool_common.h — setup and teardown shared by the in-tree tools
 * (dump_geist_*, eval_geist, bench_decision): one backend, one model, one
 * session, and reading a whole file. Header-only; each tool is one TU. */
#ifndef GEIST_TOOL_COMMON_H
#define GEIST_TOOL_COMMON_H

#include <geist.h>
#include <geist_backend.h>
#include <geist_util.h>

#include <stdio.h>
#include <stdlib.h>

struct tool_engine {
    struct geist_backend *be;
    struct geist_model   *model;
    struct geist_session *sess;
};

/* Create backend `backend_name` (backend_opts may be nullptr), load `gguf`
 * with `load_opts` (nullptr = geist_model_load) and create a session with
 * `session_opts`. On failure prints which step failed and returns false;
 * *e holds what was created and tool_close releases it either way. */
[[nodiscard]] static inline bool tool_open(const char                      *backend_name,
                                           const struct geist_backend_opts *backend_opts,
                                           const char                      *gguf,
                                           const struct geist_session_opts *load_opts,
                                           const struct geist_session_opts *session_opts,
                                           struct tool_engine              *e) {
    *e = (struct tool_engine) {0};
    if (geist_backend_create(backend_name, backend_opts, nullptr, &e->be) != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return false;
    }
    if (geist_model_load_with_opts(gguf, e->be, load_opts, &e->model) != GEIST_OK) {
        fprintf(stderr, "model load failed: %s\n", geist_last_create_error());
        return false;
    }
    if (geist_session_create(e->model, e->be, session_opts, &e->sess) != GEIST_OK) {
        fprintf(stderr, "session create failed: %s\n", geist_last_create_error());
        return false;
    }
    return true;
}

static inline void tool_close(struct tool_engine *e) {
    geist_session_destroy(e->sess);
    geist_model_destroy(e->model);
    geist_backend_destroy(e->be);
    *e = (struct tool_engine) {0};
}

/* The whole file, NUL-terminated (the NUL is not counted in *size);
 * nullptr when it cannot be read. Free with free(). */
static inline char *tool_read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (f == nullptr)
        return nullptr;
    long end = -1;
    if (fseek(f, 0, SEEK_END) != 0 || (end = ftell(f)) < 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return nullptr;
    }
    const size_t bytes = (size_t) end;
    char        *data  = malloc(bytes + 1u);
    if (data == nullptr || fread(data, 1, bytes, f) != bytes) {
        free(data);
        fclose(f);
        return nullptr;
    }
    fclose(f);
    data[bytes] = '\0';
    *size       = bytes;
    return data;
}

#endif /* GEIST_TOOL_COMMON_H */
