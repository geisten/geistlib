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
 * tool table allows. Grammar-constraint, two slices: (1) an off-whitelist tool
 * NAME is re-picked by agent_decode_name_constrained, which decodes the name
 * constrained to the whitelist (a near-miss recovers to the model's intended
 * tool, not an error step); (2) the args object is re-keyed to the tool's
 * args_schema by agent_args_normalize (small models mis-key flat string args).
 * Full upgrade: per-token logit masking in the sampler so the model cannot even
 * emit an off-grammar token, plus a constrained key-decode for multi-key args.
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

/* Model-specific chat framing. The agent loop, whitelist gate, and grammar
 * constraint are all model-AGNOSTIC; only the turn markers + the assistant-turn
 * terminator differ per model family. Splitting that out here is what lets the
 * same agent drive Gemma, Llama, or BitNet — feeding one model another's turn
 * tokens pushes it off-distribution and wrecks instruction-following. The agent
 * auto-detects the template from the model's special tokens in geist_agent_init;
 * a caller may override `a->tmpl` after init (e.g. to match a GGUF's own
 * tokenizer.chat_template). */
struct geist_chat_template {
    const char *name;
    const char *user_open;  /* opens a user / observation turn */
    const char *turn_close; /* closes any turn */
    const char *model_open; /* opens the assistant turn (the model generates after this) */
    const char *stop;       /* assistant-turn terminator token text; "" -> stop on EOS only */
    const char *leak[5];    /* nullptr-terminated marker literals to cut if they leak into a turn */
};

/* Gemma 3/4: <start_of_turn>{role}\n … <end_of_turn>. */
static const struct geist_chat_template GEIST_CHAT_GEMMA = {
    .name       = "gemma",
    .user_open  = "<start_of_turn>user\n",
    .turn_close = "<end_of_turn>\n",
    .model_open = "<start_of_turn>model\n",
    .stop       = "<end_of_turn>",
    .leak       = {"<start_of_turn", "</start_of_turn", "<end_of_turn", "</end_of_turn", nullptr},
};

/* Generic instruct fallback for families without Gemma's turn tokens (Llama 3,
 * BitNet b1.58, …): plain "User:/Assistant:" framing; the assistant turn ends at
 * the model's EOS. ponytail: a widely-understood format, not any one model's
 * exact template — refine per family (or render the GGUF's tokenizer.chat_template)
 * if a model needs its native framing for good tool-calling. */
static const struct geist_chat_template GEIST_CHAT_GENERIC = {
    .name       = "generic",
    .user_open  = "User: ",
    .turn_close = "\n",
    .model_open = "Assistant:",
    .stop       = "",
    .leak       = {"\nUser:", "\nAssistant:", nullptr, nullptr, nullptr},
};

/* Pick a chat template from the model's special tokens. Gemma is identified by
 * its <end_of_turn> turn marker; everything else gets the generic fallback. */
static inline struct geist_chat_template geist_chat_template_for_model(struct geist_model *m) {
    if (geist_model_token_by_text(m, "<end_of_turn>") != GEIST_TOKEN_NONE) {
        return GEIST_CHAT_GEMMA;
    }
    return GEIST_CHAT_GENERIC;
}

struct geist_agent {
    struct geist_model        *model;
    struct geist_session      *session;
    const struct geist_tool   *tools; /* borrowed — caller keeps it alive */
    size_t                     n_tools;
    size_t                     max_steps;
    const char                *system_prompt; /* borrowed; nullptr -> default role */
    struct geist_chat_template tmpl;          /* model-specific framing (auto-detected in init) */
    geist_token_t              eos, eot;
    char                       transcript[GEIST_AGENT_TRANSCRIPT_CAP];
    size_t                     tlen;
};

/* Caller provides storage for *a (it is large — put it in static/heap, not a
 * deep stack). tools[] + system_prompt must outlive the agent. max_steps 0 ->
 * default 8. system_prompt nullptr -> a generic default role line. */
static inline void geist_agent_init(struct geist_agent     *a,
                                    struct geist_model     *model,
                                    struct geist_session   *session,
                                    size_t                  n_tools,
                                    const struct geist_tool tools[static n_tools],
                                    size_t                  max_steps,
                                    const char             *system_prompt) {
    a->model         = model;
    a->session       = session;
    a->tools         = tools;
    a->n_tools       = n_tools;
    a->max_steps     = max_steps ? max_steps : 8;
    a->system_prompt = system_prompt;
    a->tmpl          = geist_chat_template_for_model(model);
    a->eos           = geist_model_eos_token(model);
    a->eot = a->tmpl.stop[0] ? geist_model_token_by_text(model, a->tmpl.stop) : GEIST_TOKEN_NONE;
    a->tlen = 0;
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
    char pat[GEIST_AGENT_NAME_CAP + 4]; /* "<key>" + NUL; sized so a NAME_CAP key can't truncate */
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

/* ---- args-schema enforcement (item 3, second slice) ------------------------
 * The whitelist constrains the tool NAME; the args object still arrives free,
 * and small models routinely mis-KEY it (e.g. {"contents":...} for a tool that
 * wants {"query":...}), so the dispatch then fails on a missing field. The args
 * VALUE is genuinely free (a query, a URL) — nothing to constrain there — so the
 * only schema-constrainable part is the KEY. For the common single-key string
 * tool the key has no real choice: it is a rename, not a decode. So we ENFORCE
 * the schema by re-keying (deterministic, no model pass), not by a constrained
 * decode. Multi-key schemas are left untouched (re-keying is ambiguous there);
 * the upgrade for those is a constrained key-decode — same technique as
 * agent_decode_name_constrained. */

/* First JSON string VALUE in obj (the value of the first "k":"v" pair) -> out.
 * Returns 1 if a non-empty string value was found. Flat string values only. */
static inline int agent_first_str_value(const char *obj, size_t cap, char out[static cap]) {
    out[0]        = '\0';
    const char *p = strchr(obj, ':');
    if (!p) {
        return 0;
    }
    p++;
    while (*p == ' ' || *p == '\t') {
        p++;
    }
    if (*p != '"') {
        return 0; /* non-string value -> not handled */
    }
    p++;
    size_t w = 0;
    while (*p && *p != '"' && w + 1 < cap) {
        if (*p == '\\' && p[1]) {
            p++;
        }
        out[w++] = *p++;
    }
    out[w] = '\0';
    return w > 0;
}

/* Keys declared in a tool args_schema like {"query": string, "limit": int}:
 * a quoted token immediately followed (after ws) by ':' is a key. Writes up to
 * max keys (NUL-term) and returns the count. ponytail: flat schema only. */
static inline size_t
agent_schema_keys(const char *schema, size_t max, char keys[static max][GEIST_AGENT_NAME_CAP]) {
    size_t      n = 0;
    const char *p = schema;
    while (*p && n < max) {
        if (*p != '"') {
            p++;
            continue;
        }
        const char *q = p + 1;
        size_t      w = 0;
        char        tmp[GEIST_AGENT_NAME_CAP];
        while (*q && *q != '"' && w + 1 < sizeof tmp) {
            tmp[w++] = *q++;
        }
        tmp[w] = '\0';
        if (*q != '"') {
            p = q; /* unterminated quote */
            continue;
        }
        const char *r = q + 1;
        while (*r == ' ' || *r == '\t') {
            r++;
        }
        if (*r == ':') {
            memcpy(keys[n++], tmp, w + 1);
        }
        p = q + 1;
    }
    return n;
}

/* Enforce a single-key string schema on the model's args in place: if the
 * schema declares exactly one key and the model used a different one, re-key its
 * first string value under the schema key. Returns 1 if args carry the schema
 * key afterward. Pure (no model). No schema keys -> nothing to enforce (1);
 * multi-key -> left untouched (0). */
static inline int
agent_args_normalize(const char *schema, size_t args_cap, char args[static args_cap]) {
    char   keys[4][GEIST_AGENT_NAME_CAP];
    size_t nk = agent_schema_keys(schema, 4, keys);
    if (nk == 0) {
        return 1; /* schema declares no keys */
    }
    if (nk > 1) {
        return 0; /* re-keying is ambiguous; constrained key-decode is the upgrade */
    }
    char present[GEIST_AGENT_ARGS_CAP];
    if (agent_json_str(args, keys[0], sizeof present, present)) {
        return 1; /* model already used the schema key (happy path) */
    }
    char val[GEIST_AGENT_ARGS_CAP];
    if (!agent_first_str_value(args, sizeof val, val)) {
        return 0; /* no string value to re-key */
    }
    int k = snprintf(args, args_cap, "{\"%s\":\"", keys[0]);
    if (k < 0 || (size_t) k >= args_cap) {
        return 0;
    }
    size_t w = (size_t) k;
    for (const char *v = val; *v && w + 3 < args_cap; v++) {
        if (*v == '"' || *v == '\\') {
            args[w++] = '\\';
        }
        args[w++] = *v;
    }
    args[w++] = '"';
    args[w++] = '}';
    args[w]   = '\0';
    return 1;
}

static inline const struct geist_tool *agent_find(const struct geist_agent *a, const char *name) {
    for (size_t i = 0; i < a->n_tools; i++) {
        if (strcmp(a->tools[i].name, name) == 0) {
            return &a->tools[i];
        }
    }
    return nullptr; /* not in the whitelist -> will not run */
}

/* ---- grammar-constrained tool-name selection (item 3, first slice) ---------
 * The whitelist gate already stops an off-list tool from RUNNING, but a
 * near-miss name (typo, wrong case, a hallucinated tool) just burns a step on an
 * error observation. On that branch we instead pick the tool by *constrained
 * decoding*: walk the name token-by-token but only ever along a real whitelist
 * name, letting the model's own logits choose WHICH whitelisted tool to extend
 * toward. The result is a whitelist member by construction.
 *
 * Driven entirely over the public peek_logits/prefill_tokens/tokenize API — no
 * in-engine sampler change, so the decode hot path (and the perf gate) is
 * untouched; the cost is a short bounded pass that runs only on the off-list
 * branch. Upgrade path (next rungs): per-token logit masking inside the sampler
 * for a true "cannot emit an off-grammar token" guarantee, plus an args-schema
 * grammar so the args object is constrained too. */

/* partial is a prefix of (or equal to) some whitelist name. PURE (no model). */
static inline int agent_name_is_prefix(const struct geist_agent *a, const char *partial) {
    size_t pl = strlen(partial);
    for (size_t i = 0; i < a->n_tools; i++) {
        if (strncmp(a->tools[i].name, partial, pl) == 0) {
            return 1;
        }
    }
    return 0;
}

/* index of the tool whose name == partial exactly, else -1. PURE (no model). */
static inline int agent_name_complete(const struct geist_agent *a, const char *partial) {
    for (size_t i = 0; i < a->n_tools; i++) {
        if (strcmp(a->tools[i].name, partial) == 0) {
            return (int) i;
        }
    }
    return -1;
}

/* Greedily decode the tool name constrained to the whitelist. Precondition:
 * logits are pending at the name's first position (caller has prefilled the
 * transcript + the opening `{"tool":"`). At each step, among the whitelist names
 * that still match the chars emitted so far, force the next token of whichever
 * has the highest logit, so the model picks the tool but can only ever spell a
 * real one. Returns the chosen tool index, or -1 if the whitelist is empty.
 * ponytail: assumes no whitelist name is a strict prefix of another (true for
 * distinct tool names) — else greedy never stops at the shorter one. */
static inline int agent_decode_name_constrained(struct geist_agent *a) {
    char   partial[GEIST_AGENT_NAME_CAP];
    size_t pl  = 0;
    partial[0] = '\0';
    for (int step = 0; step < GEIST_AGENT_NAME_CAP; step++) {
        size_t       n_logits = 0;
        const float *logits   = geist_session_peek_logits(a->session, &n_logits);
        if (!logits || n_logits == 0) {
            break;
        }
        /* pick the next token of the highest-logit still-matching name */
        int           have = 0;
        float         best_logit = 0;
        geist_token_t best_tok   = 0;
        for (size_t i = 0; i < a->n_tools; i++) {
            const char *name = a->tools[i].name;
            if (strlen(name) <= pl || strncmp(name, partial, pl) != 0) {
                continue; /* not a still-matching name with a remaining suffix */
            }
            geist_token_t ids[8];
            size_t        nid = 0;
            if (geist_session_tokenize(a->session, name + pl, 8, ids, &nid) != GEIST_OK || nid == 0 ||
                ids[0] >= (geist_token_t) n_logits) {
                continue;
            }
            if (!have || logits[ids[0]] > best_logit) {
                have = 1, best_logit = logits[ids[0]], best_tok = ids[0];
            }
        }
        if (!have) {
            break; /* partial is complete; nothing extends it */
        }
        const char *piece = geist_session_token_to_str(a->session, best_tok);
        size_t      plen  = piece ? strlen(piece) : 0;
        if (!plen || pl + plen + 1 >= sizeof partial ||
            geist_session_prefill_tokens(a->session, 1, &best_tok) != GEIST_OK) {
            break;
        }
        memcpy(partial + pl, piece, plen);
        pl += plen;
        partial[pl] = '\0';
    }
    return agent_name_complete(a, partial);
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
     * template leak marker so it never leaks into the call/answer. */
    const char *cut = nullptr;
    for (size_t m = 0; a->tmpl.leak[m] != nullptr; m++) {
        const char *hit = strstr(out, a->tmpl.leak[m]);
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
    const char *role = a->system_prompt ? a->system_prompt : "You are a task agent.";
    size_t      w    = (size_t) snprintf(out,
                                         cap,
                                         "%s%s\n"
                                         "To act, reply with EXACTLY one line of JSON:\n"
                                         "{\"tool\":\"<name>\",\"args\":{...}}\n"
                                         "Available tools (you may use no other):\n",
                                         a->tmpl.user_open,
                                         role);
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
    /* append the user's request, close the turn, open the model turn */
    a->tlen += (size_t) snprintf(a->transcript + a->tlen,
                                 sizeof a->transcript - a->tlen,
                                 "%.*s%s%s",
                                 (int) req_len,
                                 req,
                                 a->tmpl.turn_close,
                                 a->tmpl.model_open);

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
            /* off-list name -> recover via constrained decode: prefill the
             * transcript + the opening `{"tool":"` and let the model pick a
             * whitelisted tool, spelled along the whitelist by construction. */
            geist_token_t ids[GEIST_AGENT_NAME_CAP];
            size_t        nid = 0;
            int           idx = -1;
            if (geist_session_reset(a->session) == GEIST_OK &&
                geist_session_set_prompt(a->session, a->transcript) == GEIST_OK &&
                geist_session_tokenize(a->session, "{\"tool\":\"", sizeof ids / sizeof *ids, ids,
                                       &nid) == GEIST_OK &&
                geist_session_prefill_tokens(a->session, nid, ids) == GEIST_OK) {
                idx = agent_decode_name_constrained(a);
            }
            if (idx >= 0) {
                t = &a->tools[idx];
            }
        }
        if (!t) {
            on = (size_t) snprintf(obs, sizeof obs, "error: tool \"%s\" is not allowed", name);
        } else {
            agent_args_normalize(t->args_schema, sizeof args, args); /* re-key to the schema */
            if (t->invoke(t->ctx, strlen(args), args, sizeof obs, obs, &on) != GEIST_OK) {
                on = (size_t) snprintf(obs, sizeof obs, "error: tool \"%s\" failed", name);
            }
        }

        /* keep a best-effort answer in resp in case we hit max_steps */
        size_t bn = agent_copy(resp_cap, resp, turn);
        if (resp_len) {
            *resp_len = bn;
        }

        /* append the model's call + the observation, reopen the model turn */
        int w = snprintf(a->transcript + a->tlen,
                         sizeof a->transcript - a->tlen,
                         "%s%s%sobservation: %.*s%s%s",
                         turn,
                         a->tmpl.turn_close,
                         a->tmpl.user_open,
                         (int) on,
                         obs,
                         a->tmpl.turn_close,
                         a->tmpl.model_open);
        if (w < 0 || (size_t) w >= sizeof a->transcript - a->tlen) {
            return GEIST_E_INVALID_STATE; /* context full */
        }
        a->tlen += (size_t) w;
    }
    return GEIST_OK; /* max_steps hit; resp holds the last turn (best effort) */
}

#endif /* GEIST_AGENT_H */
