/*
 * agent_memory.h — remember / recall geist_tools over the file-based memory
 * palace (mind.h). They give the agent and the chat loop model-callable access
 * to the same notes the /remember and /recall slash commands write by hand.
 *
 *   remember: args {"text": "..."}   write a note; the title is derived from the
 *             first line of the text. SINGLE arg on purpose — a forced call (see
 *             agent_force_call) can only lift one key from the request, so a
 *             two-key remember(title,text) couldn't be forced. mind_remember
 *             slugifies the title and builds the index hook from the body.
 *   recall:   args {"slug": "..."}   load note <slug>.md back into context.
 *
 * "Search my notes about X" is NOT a tool here — point the existing doc_search
 * at $GEIST_MIND_DIR for that (local RAG over the same markdown), no new code.
 *
 * ponytail: title = the note's first line, capped — good enough for a slug; pass
 * an explicit title via the /remember slash command when you want control.
 */
#ifndef GEIST_AGENT_MEMORY_H
#define GEIST_AGENT_MEMORY_H

#include <geist.h>

#include "agent.h"
#include "mind.h"

#include <stdio.h>
#include <string.h>

enum { MEMORY_ARG_CAP = 1 << 14 };

static inline enum geist_status memory_remember_invoke(void      *ctx,
                                                       size_t     args_len,
                                                       const char args[static args_len],
                                                       size_t     out_cap,
                                                       char       out[static out_cap],
                                                       size_t    *out_len) {
    (void) ctx;
    (void) args_len;
    static char text[MEMORY_ARG_CAP];
    if (!agent_json_str(args, "text", sizeof text, text) || text[0] == '\0') {
        size_t n = (size_t) snprintf(out, out_cap, "error: remember needs a non-empty \"text\"");
        if (out_len) {
            *out_len = n;
        }
        return GEIST_OK;
    }
    /* title = first line of the text, capped — mind slugifies it for the filename
     * and builds the index hook from the body. */
    char   title[80];
    size_t t = 0;
    for (const char *p = text; *p && *p != '\n' && t + 1 < sizeof title; p++) {
        title[t++] = *p;
    }
    title[t] = '\0';

    char slug[256];
    mind_slugify(title, slug, sizeof slug);
    size_t n = (mind_remember(title, text) == 0)
                       ? (size_t) snprintf(out, out_cap, "remembered as \"%s\" (%s.md)", title, slug)
                       : (size_t) snprintf(out, out_cap, "error: could not write the note");
    if (out_len) {
        *out_len = n;
    }
    return GEIST_OK;
}

static inline enum geist_status memory_recall_invoke(void      *ctx,
                                                     size_t     args_len,
                                                     const char args[static args_len],
                                                     size_t     out_cap,
                                                     char       out[static out_cap],
                                                     size_t    *out_len) {
    (void) ctx;
    (void) args_len;
    char slug[256];
    if (!agent_json_str(args, "slug", sizeof slug, slug) || slug[0] == '\0') {
        size_t n = (size_t) snprintf(out, out_cap, "error: recall needs a \"slug\"");
        if (out_len) {
            *out_len = n;
        }
        return GEIST_OK;
    }
    long got = mind_recall(slug, out, out_cap);
    if (got < 0) {
        size_t n = (size_t) snprintf(out, out_cap, "error: no note \"%s\"", slug);
        if (out_len) {
            *out_len = n;
        }
        return GEIST_OK;
    }
    if (out_len) {
        *out_len = (size_t) got;
    }
    return GEIST_OK;
}

static inline struct geist_tool remember_tool(void) {
    return (struct geist_tool) {
            .name        = "remember",
            .description = "eine Notiz im Gedächtnis speichern",
            .args_schema = "{\"text\": string}",
            .invoke      = memory_remember_invoke,
            .ctx         = nullptr,
    };
}

static inline struct geist_tool recall_tool(void) {
    return (struct geist_tool) {
            .name        = "recall",
            .description = "eine gespeicherte Notiz laden",
            .args_schema = "{\"slug\": string}",
            .invoke      = memory_recall_invoke,
            .ctx         = nullptr,
    };
}

#endif /* GEIST_AGENT_MEMORY_H */
