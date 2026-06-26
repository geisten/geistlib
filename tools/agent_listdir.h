/*
 * agent_listdir.h — a list_dir geist_tool: list the entries of a directory.
 * args = {"path": "..."} (default "."). The safe equivalent of `ls`:
 *
 *   - implemented with opendir/readdir, NOT by spawning a shell or `ls`, so
 *     there is NO command-injection surface (a path like ".;rm -rf ~" is just a
 *     directory name that fails to open — it is never interpreted);
 *   - identical on Linux + macOS (POSIX dirent), no platform branching;
 *   - the model's ONLY input is the path, used solely as an opendir() argument:
 *     it can read a directory the host process can already read — no exec, no
 *     writes, no shell expansion.
 *
 * This is the agent security model in miniature: the host exposes a fixed,
 * narrow capability ("list a directory"), the model can only invoke it with a
 * validated string — it never gets to run an arbitrary command.
 *
 * ponytail: lists any path the process can read; to confine it to a sandbox,
 * pass a root dir as ctx and reject paths that escape it (realpath + prefix
 * check). Skips dotfiles like plain `ls`.
 */
#ifndef GEIST_AGENT_LISTDIR_H
#define GEIST_AGENT_LISTDIR_H

#include <geist.h>

#include "agent.h"

#include <dirent.h>
#include <stdio.h>
#include <string.h>

static inline enum geist_status listdir_invoke(void      *ctx,
                                               size_t     args_len,
                                               const char args[static args_len],
                                               size_t     out_cap,
                                               char       out[static out_cap],
                                               size_t    *out_len) {
    (void) ctx;
    (void) args_len;
    char path[1024];
    if (!agent_json_str(args, "path", sizeof path, path) || path[0] == '\0') {
        snprintf(path, sizeof path, ".");
    }
    DIR *d = opendir(path);
    if (d == nullptr) {
        size_t n = (size_t) snprintf(out, out_cap, "error: cannot open directory \"%s\"", path);
        if (out_len) {
            *out_len = n;
        }
        return GEIST_OK;
    }
    size_t         w = 0;
    struct dirent *e;
    while ((e = readdir(d)) != nullptr) {
        if (e->d_name[0] == '.') {
            continue; /* skip hidden entries, like `ls` without -a */
        }
        size_t nl = strlen(e->d_name);
        if (w + nl + 2 > out_cap) {
            break; /* leave room for '\n' + '\0' */
        }
        memcpy(out + w, e->d_name, nl);
        w += nl;
        out[w++] = '\n';
    }
    closedir(d);
    if (w == 0) {
        w = (size_t) snprintf(out, out_cap, "(empty directory)");
    } else {
        out[w] = '\0';
    }
    if (out_len) {
        *out_len = w;
    }
    return GEIST_OK;
}

static inline struct geist_tool listdir_tool(void) {
    return (struct geist_tool) {
            .name        = "list_dir",
            .args_schema = "{\"path\": string}",
            .invoke      = listdir_invoke,
            .ctx         = nullptr,
    };
}

#endif /* GEIST_AGENT_LISTDIR_H */
