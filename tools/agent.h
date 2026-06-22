/*
 * agent.h — a bounded, whitelist-gated tool-use loop over a resident session.
 *
 * Header-only (static inline) so the desktop socket daemon and an iOS/Android
 * app share one copy with no libgeist surface change and no build wiring. The
 * agent is the SAME process as the model: geist_agent_run() is an in-process
 * call, so "resident" just means the session/agent outlive the request.
 *
 * Security model — the host, not the model, decides what runs:
 *   - the model may only act by emitting one call:  {"tool":"<name>","args":{...}}
 *   - geist_agent_run validates <name> against the caller's whitelist; an
 *     unknown/forbidden tool NEVER runs (it gets an error observation instead),
 *   - max_steps bounds how many tool calls one request can trigger (runaway +
 *     cost guard on constrained hardware).
 * A small model jailbreaks easily as free chat; here it can only DO what the
 * tool table allows. (Upgrade: grammar-constrained sampling so the model
 * cannot even emit an off-grammar token — see the ponytail note in run().)
 *
 * No assert(): all checks are explicit and return enum geist_status. Buffers
 * are caller-provided or fixed in the struct (no hidden heap).
 */
#ifndef GEIST_AGENT_H
#define GEIST_AGENT_H

#include <geist.h>
#include <geist_util.h>

#include <stdio.h>
#include <string.h>

enum {
    GEIST_AGENT_TRANSCRIPT_CAP = 1 << 15,
    GEIST_AGENT_TURN_CAP       = 4096,
    GEIST_AGENT_NAME_CAP       = 64,
    GEIST_AGENT_ARGS_CAP       = 1024,
    GEIST_AGENT_OBS_CAP        = 4096,
    GEIST_AGENT_MAX_DECODE     = 512,
};

/* A host action. The agent never runs anything not in the whitelist passed to
 * geist_agent_init; the callback receives only validated args. ctx is the
 * host's state (doc index, HTTP client, home bridge) and is host-owned. */
struct geist_tool {
    const char *name;        /* whitelist key, must match the emitted "tool" */
    const char *args_schema; /* shown to the model, e.g. {"query": string} */
    enum geist_status (*invoke)(void      *ctx,
                                size_t     args_len,
                                const char args[static args_len],
                                size_t     out_cap,
                                char       out[static out_cap],
                                size_t    *out_len);
    void *ctx;
};

struct geist_agent {
    struct geist_model      *model;
    struct geist_session    *session;
    const struct geist_tool *tools; /* borrowed — caller keeps it alive */
    size_t                   n_tools;
    size_t                   max_steps;
    geist_token_t            eos, eot;
    char                     transcript[GEIST_AGENT_TRANSCRIPT_CAP];
    size_t                   tlen;
};

/* Caller provides storage for *a (it is large — put it in static/heap, not a
 * deep stack). tools[] must outlive the agent. max_steps 0 -> default 8. */
static inline void geist_agent_init(struct geist_agent     *a,
                                    struct geist_model     *model,
                                    struct geist_session   *session,
                                    size_t                  n_tools,
                                    const struct geist_tool tools[static n_tools],
                                    size_t                  max_steps) {
    a->model     = model;
    a->session   = session;
    a->tools     = tools;
    a->n_tools   = n_tools;
    a->max_steps = max_steps ? max_steps : 8;
    a->eos       = geist_model_eos_token(model);
    a->eot       = geist_model_token_by_text(model, "<end_of_turn>");
    a->tlen      = 0;
}

/* Find the first {"tool":"NAME","args":{...}} in raw. Returns 1 and fills name
 * + args (the brace-balanced {...}, or "{}" if absent) when found, else 0.
 * ponytail: naive brace balance, not string-aware — a '}' inside an arg string
 * value would mis-balance. Fine for flat args; for nested/quoted args move to
 * grammar-constrained sampling so only valid calls can be emitted. */
static inline int agent_parse_call(size_t     raw_len,
                                   const char raw[static raw_len],
                                   size_t     name_cap,
                                   char       name[static name_cap],
                                   size_t     args_cap,
                                   char       args[static args_cap]) {
    name[0] = '\0';
    snprintf(args, args_cap, "{}");
    const char *p = strstr(raw, "\"tool\"");
    if (!p) {
        return 0;
    }
    const char *end = raw + raw_len;
    p += 6;
    while (p < end && *p != '"') { /* skip ':' and spaces to the value quote */
        p++;
    }
    if (p >= end) {
        return 0;
    }
    p++; /* past opening quote */
    size_t w = 0;
    while (p < end && *p != '"' && w + 1 < name_cap) {
        name[w++] = *p++;
    }
    name[w] = '\0';
    if (w == 0) {
        return 0;
    }

    const char *ap = strstr(raw, "\"args\"");
    if (ap) {
        const char *b = ap + 6;
        while (b < end && *b != '{') {
            b++;
        }
        if (b < end) {
            int    depth = 0;
            size_t aw    = 0;
            for (const char *q = b; q < end && aw + 1 < args_cap; q++) {
                args[aw++] = *q;
                if (*q == '{') {
                    depth++;
                } else if (*q == '}' && --depth == 0) {
                    break;
                }
            }
            args[aw] = '\0';
        }
    }
    return 1;
}

/* Extract a JSON string field value:  "<key>":"<value>"  into out (NUL-term).
 * Returns 1 if a non-empty value was found. Handles a backslash escape so a
 * quoted value with \" survives. ponytail: flat fields only, no nested objects
 * or \uXXXX — move to grammar-constrained args if tools need rich inputs. */
static inline int
agent_json_str(const char *json, const char *key, size_t cap, char out[static cap]) {
    out[0] = '\0';
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) {
        return 0;
    }
    p += strlen(pat);
    while (*p && *p != '"') { /* skip ':' + spaces to the opening quote */
        p++;
    }
    if (*p != '"') {
        return 0;
    }
    p++;
    size_t w = 0;
    while (*p && *p != '"' && w + 1 < cap) {
        if (*p == '\\' && p[1]) {
            p++; /* take the escaped char literally */
        }
        out[w++] = *p++;
    }
    out[w] = '\0';
    return w > 0;
}

static inline const struct geist_tool *agent_find(const struct geist_agent *a, const char *name) {
    for (size_t i = 0; i < a->n_tools; i++) {
        if (strcmp(a->tools[i].name, name) == 0) {
            return &a->tools[i];
        }
    }
    return nullptr; /* not in the whitelist -> will not run */
}

/* Decode one assistant turn into out (greedy, stops on EOS/<end_of_turn>).
 * Returns bytes written. */
static inline size_t agent_generate_turn(struct geist_agent *a, size_t cap, char out[static cap]) {
    size_t w = 0;
    for (int i = 0; i < GEIST_AGENT_MAX_DECODE; i++) {
        geist_token_t tok = 0;
        if (geist_session_decode_step(a->session, &tok) != GEIST_OK) {
            break;
        }
        if (tok == a->eos || tok == a->eot) {
            break;
        }
        const char *piece = geist_session_token_to_str(a->session, tok);
        if (!piece) {
            break;
        }
        size_t pl = strlen(piece);
        if (pl >= 2 && piece[0] == '<' && piece[pl - 1] == '>') {
            break; /* control marker */
        }
        if (w + pl + 1 >= cap) {
            break;
        }
        memcpy(out + w, piece, pl);
        w += pl;
    }
    out[w] = '\0';
    /* A turn marker emitted as multiple BPE pieces (e.g. </start_of_turn>)
     * slips past the single-token break above; cut the turn at the earliest
     * marker literal so it never leaks into the call/answer. */
    const char *cut     = nullptr;
    const char *marks[] = {"<start_of_turn", "</start_of_turn", "<end_of_turn", "</end_of_turn"};
    for (size_t m = 0; m < sizeof marks / sizeof marks[0]; m++) {
        const char *hit = strstr(out, marks[m]);
        if (hit && (!cut || hit < cut)) {
            cut = hit;
        }
    }
    if (cut) {
        w = (size_t) (cut - out);
        while (w > 0 && (out[w - 1] == '\n' || out[w - 1] == ' ')) {
            w--;
        }
        out[w] = '\0';
    }
    return w;
}

/* Build the fixed system prompt (scope definition): role + the tool whitelist
 * + the required output shape. Returns bytes written. */
static inline size_t
agent_system_prompt(const struct geist_agent *a, size_t cap, char out[static cap]) {
    size_t w =
            (size_t) snprintf(out,
                              cap,
                              "<start_of_turn>user\n"
                              "You are a task agent. To act, reply with EXACTLY one line of JSON:\n"
                              "{\"tool\":\"<name>\",\"args\":{...}}\n"
                              "Available tools (you may use no other):\n");
    for (size_t i = 0; i < a->n_tools && w < cap; i++) {
        w += (size_t) snprintf(
                out + w, cap - w, "- %s: args %s\n", a->tools[i].name, a->tools[i].args_schema);
    }
    if (w < cap) {
        w += (size_t) snprintf(out + w,
                               cap - w,
                               "When you can answer, reply with the final answer as plain "
                               "text (no JSON).\n");
    }
    return w;
}

/* Copy src into resp (cap), truncating safely; returns the written length.
 * Used instead of snprintf(resp, cap, "%s", turn) so gcc -Wformat-truncation
 * (under _FORTIFY_SOURCE) can't flag copying a larger fixed buffer into resp. */
static inline size_t agent_copy(size_t cap, char resp[static cap], const char *src) {
    if (cap == 0) {
        return 0;
    }
    size_t n = strlen(src);
    if (n >= cap) {
        n = cap - 1;
    }
    memcpy(resp, src, n);
    resp[n] = '\0';
    return n;
}

/* Run one request to completion: loop generate -> parse -> (whitelist) dispatch
 * -> observe, until the model answers in plain text or max_steps is hit.
 * On any failure resp is left well-defined ("" / *resp_len 0). */
[[nodiscard]] static inline enum geist_status geist_agent_run(struct geist_agent *a,
                                                              size_t              req_len,
                                                              const char req[static req_len],
                                                              size_t     resp_cap,
                                                              char       resp[static resp_cap],
                                                              size_t    *resp_len) {
    if (resp_cap == 0) {
        return GEIST_E_INVALID_ARG;
    }
    resp[0] = '\0';
    if (resp_len) {
        *resp_len = 0;
    }

    a->tlen = agent_system_prompt(a, sizeof a->transcript, a->transcript);
    /* close the system turn, open the user's request, open the model turn */
    a->tlen += (size_t) snprintf(a->transcript + a->tlen,
                                 sizeof a->transcript - a->tlen,
                                 "%.*s<end_of_turn>\n<start_of_turn>model\n",
                                 (int) req_len,
                                 req);

    char turn[GEIST_AGENT_TURN_CAP];
    char name[GEIST_AGENT_NAME_CAP];
    char args[GEIST_AGENT_ARGS_CAP];
    char obs[GEIST_AGENT_OBS_CAP];

    for (size_t step = 0; step < a->max_steps; step++) {
        if (geist_session_reset(a->session) != GEIST_OK ||
            geist_session_set_prompt(a->session, a->transcript) != GEIST_OK) {
            return GEIST_E_INVALID_STATE;
        }
        /* ponytail: full-transcript reprefill each step — O(n^2) over the
         * request. Switch to incremental prefill_tokens if requests get long. */
        size_t tn = agent_generate_turn(a, sizeof turn, turn);

        if (!agent_parse_call(tn, turn, sizeof name, name, sizeof args, args)) {
            /* no tool call -> this turn is the final answer */
            size_t n = agent_copy(resp_cap, resp, turn);
            if (resp_len) {
                *resp_len = n;
            }
            return GEIST_OK;
        }

        const struct geist_tool *t  = agent_find(a, name);
        size_t                   on = 0;
        if (!t) {
            on = (size_t) snprintf(obs, sizeof obs, "error: tool \"%s\" is not allowed", name);
        } else if (t->invoke(t->ctx, strlen(args), args, sizeof obs, obs, &on) != GEIST_OK) {
            on = (size_t) snprintf(obs, sizeof obs, "error: tool \"%s\" failed", name);
        }

        /* keep a best-effort answer in resp in case we hit max_steps */
        size_t bn = agent_copy(resp_cap, resp, turn);
        if (resp_len) {
            *resp_len = bn;
        }

        /* append the model's call + the observation, reopen the model turn */
        int w = snprintf(a->transcript + a->tlen,
                         sizeof a->transcript - a->tlen,
                         "%s<end_of_turn>\n<start_of_turn>user\nobservation: %.*s"
                         "<end_of_turn>\n<start_of_turn>model\n",
                         turn,
                         (int) on,
                         obs);
        if (w < 0 || (size_t) w >= sizeof a->transcript - a->tlen) {
            return GEIST_E_INVALID_STATE; /* context full */
        }
        a->tlen += (size_t) w;
    }
    return GEIST_OK; /* max_steps hit; resp holds the last turn (best effort) */
}

#endif /* GEIST_AGENT_H */
