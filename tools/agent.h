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
 * tool table allows. Grammar-constraint, three slices: (1) an off-whitelist
 * tool NAME is re-picked by agent_decode_name_constrained, which decodes the
 * name constrained to the whitelist (a near-miss recovers to the model's
 * intended tool, not an error step); (2) the args object is re-keyed to the
 * tool's args_schema by agent_args_normalize (small models mis-key flat string
 * args); (3) a free turn that opens a call is decoded ALONG the call grammar
 * (agent_generate_call_masked): name and key constrained per-token over the
 * public peek/prefill API, only the value free — an off-grammar call cannot be
 * emitted. Remaining upgrade: full multi-key argument grammars.
 *
 * No assert(): all checks are explicit and return enum geist_status. Buffers
 * are caller-provided or fixed in the struct (no hidden heap).
 */
#ifndef GEIST_AGENT_H
#define GEIST_AGENT_H

#include <geist.h>
#include <geist_util.h>

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <strings.h> /* strncasecmp (POSIX; present on all geist hosts) */

enum {
    GEIST_AGENT_TRANSCRIPT_CAP = 1 << 15,
    GEIST_AGENT_TURN_CAP       = 4096,
    GEIST_AGENT_NAME_CAP       = 64,
    GEIST_AGENT_ARGS_CAP       = 1024,
    GEIST_AGENT_OBS_CAP        = 4096,
    GEIST_AGENT_MAX_DECODE     = 512,
    GEIST_AGENT_LOOP_PMAX      = 128, /* longest repeating block the anti-loop cap detects */
    /* Chat context bound (agent_compact): once the transcript passes BUDGET,
     * evict the oldest turns down to TARGET. Bounds per-turn prefill (so a long
     * chat stays O(n), not O(n^2)) and avoids the hard "context full" stop.
     * ponytail: bytes as a token proxy; lower both for snappier edge chat. */
    GEIST_AGENT_CTX_BUDGET = (GEIST_AGENT_TRANSCRIPT_CAP * 3) / 4,
    GEIST_AGENT_CTX_TARGET = GEIST_AGENT_TRANSCRIPT_CAP / 2,
};

/* Generic anti-degeneration: greedy decoding on a chatty model can fall into a
 * loop, emitting the same phrase over and over without ever hitting EOS (the
 * 2B-4T's "[File Name Used: …]" x40; Gemma's "is is is"). If the tail of `out`
 * is 3 identical consecutive blocks of some length P in [3, PMAX], return P (the
 * run length); else 0. Model-INDEPENDENT — it keys on repetition, not on any
 * model's markers — and a valid one-line tool call / a real answer don't
 * triple-repeat a >=3-byte chunk, so it never fires on good output.
 * ponytail: misses loops whose period exceeds PMAX bytes; raise PMAX if needed. */
static inline size_t agent_tail_loop(const char *out, size_t w) {
    for (size_t p = 3; p <= GEIST_AGENT_LOOP_PMAX; p++) {
        if (w < 3 * p) {
            break; /* p only grows from here, so nothing longer fits either */
        }
        if (memcmp(out + w - 3 * p, out + w - 2 * p, p) == 0 &&
            memcmp(out + w - 2 * p, out + w - p, p) == 0) {
            return p;
        }
    }
    return 0;
}

/* Greedy-decode the open assistant turn into out[0..cap): stop on EOS / eot, a
 * single-token control marker (<...>), the buffer cap, or a degeneration loop
 * (keep one copy of the looped block). Then cut at the EARLIEST leaked turn
 * marker — one emitted as several BPE pieces slips past the single-token break —
 * and trim trailing whitespace. Returns bytes written. The session must already
 * be prefilled to where the model generates. Shared by the agent loop and the
 * summarizer sub-session (the only two greedy generators). */
static inline size_t geist_generate_greedy(struct geist_session     *s,
                                           geist_token_t              eos,
                                           geist_token_t              eot,
                                           const char *const          leak[],
                                           int                        max_tokens,
                                           size_t                     cap,
                                           char                       out[static cap]) {
    size_t w = 0;
    for (int i = 0; i < max_tokens; i++) {
        geist_token_t tok = 0;
        if (geist_session_decode_step(s, &tok) != GEIST_OK) {
            break;
        }
        if (tok == eos || (eot != GEIST_TOKEN_NONE && tok == eot)) {
            break;
        }
        const char *piece = geist_session_token_to_str(s, tok);
        size_t      pl    = piece ? strlen(piece) : 0;
        if (pl == 0 || (pl >= 2 && piece[0] == '<' && piece[pl - 1] == '>')) {
            break; /* control marker / empty piece */
        }
        if (w + pl + 1 >= cap) {
            break;
        }
        memcpy(out + w, piece, pl);
        w += pl;
        size_t loop_p = agent_tail_loop(out, w);
        if (loop_p > 0) {
            w -= 2 * loop_p; /* drop the repeats, keep one copy */
            break;
        }
    }
    out[w] = '\0';
    const char *cut = nullptr;
    for (size_t m = 0; leak && leak[m] != nullptr; m++) {
        const char *hit = strstr(out, leak[m]);
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

/* The "write a message, set *out_len, return OK" tail every tool shares, for the
 * error/short-result paths. Tools that fill `out` incrementally use agent_ret. */
static inline enum geist_status
agent_obs(size_t out_cap, char out[static out_cap], size_t *out_len, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(out, out_cap, fmt, ap);
    va_end(ap);
    if (out_len) {
        *out_len = (n < 0) ? 0 : (size_t) n;
    }
    return GEIST_OK;
}

/* Set *out_len (out is already filled) and return OK. */
static inline enum geist_status agent_ret(size_t *out_len, size_t w) {
    if (out_len) {
        *out_len = w;
    }
    return GEIST_OK;
}

/* A host action. The agent never runs anything not in the whitelist passed to
 * geist_agent_init; the callback receives only validated args. ctx is the
 * host's state (doc index, HTTP client, home bridge) and is host-owned. */
struct geist_tool {
    const char *name;        /* whitelist key, must match the emitted "tool" */
    const char *args_schema; /* shown to the model, e.g. {"query": string} */
    const char *description;  /* one line of intent for routing; nullptr -> use name */
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

/* Llama-3 family: <|start_header_id|>role<|end_header_id|>\n\n … <|eot_id|>. The
 * model ends a turn with <|eot_id|>, so stop on it. Microsoft's BitNet b1.58
 * 2B-4T shares the Llama-3 128k tokenizer AND its training format: its GGUF
 * tokenizer.chat_template ships a simplified "Human:/BITNETAssistant:" string
 * the model does NOT actually follow — feeding it Llama-3 framing instead makes
 * it coherent and stop cleanly ("Paris", not a 512-token ramble). (It still
 * won't emit structured tool calls — it isn't tool-trained; that needs a
 * tool-trained model like Llama-3.1 / Qwen2.5.) */
static const struct geist_chat_template GEIST_CHAT_LLAMA3 = {
    .name       = "llama3",
    .user_open  = "<|start_header_id|>user<|end_header_id|>\n\n",
    .turn_close = "<|eot_id|>",
    .model_open = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    .stop       = "<|eot_id|>",
    .leak       = {"<|start_header_id|>", "<|eot_id|>", "<|end_header_id|>", "<|begin_of_text|>",
                   nullptr},
};

/* Pick a chat template from the model's turn-end special token. NB: this keys on
 * the GGUF-EMBEDDED tokenizer (token_by_text returns NONE for a model loaded via
 * an external SentencePiece, e.g. Gemma 4) — which is deliberate, not a gap: the
 * agent builds the transcript as a STRING and set_prompt re-tokenizes it, and an
 * external tokenizer does NOT map "<start_of_turn>" text to Gemma's control
 * tokens (105/106), so feeding Gemma its own marker text off-distributes it
 * (it answers with a hallucinated "<finish_of_turn>"). Gemma works BETTER on the
 * plain generic User:/Assistant: framing here, so letting it fall through is
 * correct. <end_of_turn> token -> Gemma (GGUF-embedded gemma); <|eot_id|> ->
 * Llama-3 (incl. BitNet 2B-4T); else generic. */
static inline struct geist_chat_template geist_chat_template_for_model(struct geist_model *m) {
    if (geist_model_token_by_text(m, "<end_of_turn>") != GEIST_TOKEN_NONE) {
        return GEIST_CHAT_GEMMA;
    }
    if (geist_model_token_by_text(m, "<|eot_id|>") != GEIST_TOKEN_NONE) {
        return GEIST_CHAT_LLAMA3;
    }
    return GEIST_CHAT_GENERIC;
}

enum { AGENT_MAX_ROUTED = 26 }; /* tools the router ranks (A..Z worth of names) */

/* The agent's current step, surfaced to an optional on_event callback so a host
 * (CLI, UI, server) can show live progress instead of treating geist_agent_run as
 * a black box that only returns the final answer. One event per phase boundary. */
enum geist_agent_phase {
    GEIST_AGENT_ROUTING,   /* choosing which tool handles the request */
    GEIST_AGENT_CALLING,   /* the tool call has been formed (forced or generated) */
    GEIST_AGENT_RUNNING,   /* dispatching the tool */
    GEIST_AGENT_OBSERVED,  /* the tool returned its observation */
    GEIST_AGENT_ANSWERING, /* generating / returning the final text answer */
};

struct geist_agent_event {
    enum geist_agent_phase phase;
    size_t                 step;   /* 0-based loop index */
    const char            *tool;   /* tool name when relevant, else nullptr */
    const char            *detail; /* args on CALLING, an obs snippet on OBSERVED; may be nullptr.
                                    * Valid ONLY during the callback — copy to retain. */
};

struct geist_agent {
    struct geist_model        *model;
    struct geist_session      *session;
    const struct geist_tool   *tools; /* borrowed — caller keeps it alive */
    size_t                     n_tools;
    size_t                     max_steps;
    const char                *system_prompt; /* borrowed; nullptr -> default role */
    struct geist_chat_template tmpl;          /* model-specific framing (auto-detected in init) */
    bool                       force_call;     /* force turn 0 to be a valid tool call (see run) */
    bool                       conversation;   /* keep the transcript across geist_agent_run calls
                                                * (multi-turn chat) instead of resetting each call */
    geist_token_t              eos, eot;
    char                       transcript[GEIST_AGENT_TRANSCRIPT_CAP];
    size_t                     tlen;
    size_t                     sys_len; /* protected system-prompt prefix [0..sys_len) */
    /* Router PMI baseline (agent_select_tool): the per-tool prior is
     * request-independent, so it is computed once and cached here — every later
     * route then does ONE prefill (the request) instead of two. */
    float  route_base[AGENT_MAX_ROUTED];
    size_t route_base_n; /* tools the cached baseline covers; 0 = not computed */
    /* Optional live-progress hook (nullptr = silent, zero overhead). Set after
     * geist_agent_init. See struct geist_agent_event. */
    void (*on_event)(void *ctx, const struct geist_agent_event *ev);
    void *on_event_ctx;
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
    a->force_call    = false;
    a->conversation  = false;
    a->eos           = geist_model_eos_token(model);
    a->eot = a->tmpl.stop[0] ? geist_model_token_by_text(model, a->tmpl.stop) : GEIST_TOKEN_NONE;
    a->tlen         = 0;
    a->sys_len      = 0;
    a->route_base_n = 0; /* router baseline computed lazily on the first route */
    a->on_event     = nullptr;
    a->on_event_ctx = nullptr;
}

/* Fire the progress hook if the host set one (one nullptr-check when unset). */
static inline void agent_emit(struct geist_agent    *a,
                              enum geist_agent_phase phase,
                              size_t                 step,
                              const char            *tool,
                              const char            *detail) {
    if (a->on_event) {
        struct geist_agent_event ev = {.phase = phase, .step = step, .tool = tool, .detail = detail};
        a->on_event(a->on_event_ctx, &ev);
    }
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

/* Greedily decode ONE string out of opts[] constrained to that set.
 * Precondition: logits are pending at the string's first position. At each
 * step, among the options that still match the chars emitted so far, force the
 * next token of whichever has the highest logit — the model picks the option
 * but can only ever spell a real one. Returns the chosen index, or -1 if
 * nothing completed. ponytail: assumes no option is a strict prefix of another
 * (true for tool names and schema keys) — else greedy never stops early. */
static inline int agent_decode_pick(struct geist_agent *a, size_t n_opts, const char *const *opts) {
    char   partial[GEIST_AGENT_NAME_CAP];
    size_t pl  = 0;
    partial[0] = '\0';
    for (int step = 0; step < GEIST_AGENT_NAME_CAP; step++) {
        size_t       n_logits = 0;
        const float *logits   = geist_session_peek_logits(a->session, &n_logits);
        if (!logits || n_logits == 0) {
            break;
        }
        /* pick the next token of the highest-logit still-matching option */
        int           have       = 0;
        float         best_logit = 0;
        geist_token_t best_tok   = 0;
        for (size_t i = 0; i < n_opts; i++) {
            const char *name = opts[i];
            if (strlen(name) <= pl || strncmp(name, partial, pl) != 0) {
                continue; /* not a still-matching option with a remaining suffix */
            }
            geist_token_t ids[8];
            size_t        nid = 0;
            if (geist_session_tokenize(a->session, name + pl, 8, ids, &nid) != GEIST_OK ||
                nid == 0 || ids[0] >= (geist_token_t) n_logits) {
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
    for (size_t i = 0; i < n_opts; i++) {
        if (strcmp(opts[i], partial) == 0) {
            return (int) i;
        }
    }
    return -1;
}

/* Greedily decode the tool name constrained to the whitelist (see
 * agent_decode_pick). Precondition: caller has prefilled the transcript + the
 * opening `{"tool":"`. Returns the chosen tool index, or -1. */
static inline int agent_decode_name_constrained(struct geist_agent *a) {
    const char *names[AGENT_MAX_ROUTED];
    size_t      n = a->n_tools < AGENT_MAX_ROUTED ? a->n_tools : AGENT_MAX_ROUTED;
    for (size_t i = 0; i < n; i++) {
        names[i] = a->tools[i].name;
    }
    return agent_decode_pick(a, n, names);
}

/* Route a request to the best tool by scoring, instead of trusting the raw
 * `{"tool":"` logit (which, with several tools, picks a valid-but-wrong one — a
 * "list the directory" request forced summarize_file). Frame the request + the
 * tool menu as a question and pick the tool whose NAME the model most wants as
 * the answer (first-token logprob, the SCOREALT pattern). Scoring the name, not
 * an A/B letter, gives a weak model real semantic signal ("list_dir" lines up
 * with "list the directory" where an abstract "A" does not). Returns a tool
 * index; 0 when there is a single tool (no choice). Pure peek/prefill/tokenize —
 * leaves the session reset to the selection prompt (caller re-sets transcript). */

/* Logit of the first token of `text`, or -INF-ish if it doesn't tokenize. */
static inline float agent_first_token_logit(struct geist_agent *a, const char *text,
                                            const float *logits, size_t n_logits) {
    geist_token_t ids[8];
    size_t        nid = 0;
    if (geist_session_tokenize(a->session, text, 8, ids, &nid) != GEIST_OK || nid == 0 ||
        ids[0] >= (geist_token_t) n_logits) {
        return -1e30f;
    }
    return logits[ids[0]];
}

/* The router's pseudo-entry for "no tool": scored like a tool name, so a
 * request no tool handles ("What is 2 plus 2?", "Delete report.md") routes to a
 * direct answer instead of a forced-but-wrong call. Named "reply", not
 * "answer" — the menu instruction itself ends in "Answer with the tool name.",
 * and a name colliding with instruction vocabulary muddies the cloze signal. */
#define AGENT_ANSWER_NAME "reply"
#define AGENT_ANSWER_DESC "die Anfrage direkt beantworten, ohne Werkzeug"

/* Build the selection prompt for `req` into sel; returns its length. The menu is
 * fixed (name: description, plus the answer pseudo-entry); only the Request line
 * varies, so the same builder serves the real request and the content-free
 * baseline used for calibration. */
static inline size_t agent_select_prompt(struct geist_agent *a,
                                         size_t              n,
                                         size_t              req_len,
                                         const char         *req,
                                         size_t              cap,
                                         char                sel[static cap]) {
    size_t w = (size_t) snprintf(sel,
                                 cap,
                                 "%sWhich tool best handles this request?\nRequest: %.*s\nTools:\n",
                                 a->tmpl.user_open,
                                 (int) req_len,
                                 req);
    for (size_t i = 0; i < n && w < cap; i++) {
        const char *d = a->tools[i].description ? a->tools[i].description : "";
        w += (size_t) snprintf(sel + w, cap - w, "- %s: %s\n", a->tools[i].name, d);
    }
    w += (size_t) snprintf(sel + w, cap - w, "- %s: %s\n", AGENT_ANSWER_NAME, AGENT_ANSWER_DESC);
    w += (size_t) snprintf(sel + w,
                           cap - w,
                           "Answer with the tool name.%s%s",
                           a->tmpl.turn_close,
                           a->tmpl.model_open);
    return w;
}

/* Score the first token of n names at the current decode position into out[n]:
 * the tool names, then (as entry a->n_tools, when n covers it) the answer
 * pseudo-entry. The first token may be bare ("list_dir") or space-prefixed
 * (" list_dir") per the template — take the max of both. Returns 0 on a peek
 * failure. */
static inline int
agent_score_names(struct geist_agent *a, size_t n, const char *prompt, float out[static n]) {
    if (geist_session_reset(a->session) != GEIST_OK ||
        geist_session_set_prompt(a->session, prompt) != GEIST_OK) {
        return 0;
    }
    size_t       n_logits = 0;
    const float *logits   = geist_session_peek_logits(a->session, &n_logits);
    if (logits == nullptr || n_logits == 0) {
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        const char *nm = i < a->n_tools ? a->tools[i].name : AGENT_ANSWER_NAME;
        char        spaced[GEIST_AGENT_NAME_CAP + 1];
        snprintf(spaced, sizeof spaced, " %s", nm);
        float v0 = agent_first_token_logit(a, nm, logits, n_logits);
        float v1 = agent_first_token_logit(a, spaced, logits, n_logits);
        out[i]   = v0 > v1 ? v0 : v1;
    }
    return 1;
}

/* True if a request word looks like a named file with an extension (note.txt,
 * report.md): a dot, not leading, then 1-5 alnum to the word end. Distinguishes
 * a named file from a bare directory path. ponytail: "2.0" also matches — a rare
 * false positive, only consulted as a close-race tie-breaker, so harmless. */
static inline int agent_request_names_file(size_t req_len, const char *req) {
    for (size_t i = 0; i < req_len;) {
        while (i < req_len && (req[i] == ' ' || req[i] == '\t' || req[i] == '\n')) {
            i++;
        }
        size_t s = i, dot = (size_t) -1;
        while (i < req_len && req[i] != ' ' && req[i] != '\t' && req[i] != '\n') {
            if (req[i] == '.') {
                dot = i;
            }
            i++;
        }
        if (dot != (size_t) -1 && dot > s && i - dot >= 2 && i - dot <= 6) {
            int alnum = 1;
            for (size_t j = dot + 1; j < i; j++) {
                char c = req[j];
                if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))) {
                    alnum = 0;
                    break;
                }
            }
            if (alnum) {
                return 1;
            }
        }
    }
    return 0;
}

/* True if a tool description denotes a DIRECTORY tool — so a request naming a
 * specific file should prefer a non-dir tool. Bilingual, case-robust by matching
 * the stable inner substring (Verzeichnis/Ordner/directory/folder). */
static inline int agent_desc_is_dir(const char *d) {
    return d && (strstr(d, "erzeichnis") || strstr(d, "rdner") || strstr(d, "directory") ||
                 strstr(d, "folder"));
}

/* True if the request contains a literal http(s):// URL (bounded scan — req is
 * not NUL-terminated here). */
static inline int agent_request_has_url(size_t req_len, const char *req) {
    for (size_t i = 0; i + 7 <= req_len; i++) {
        if (memcmp(req + i, "http", 4) != 0) {
            continue;
        }
        size_t j = i + 4;
        if (j < req_len && req[j] == 's') {
            j++;
        }
        if (j + 3 <= req_len && memcmp(req + j, "://", 3) == 0) {
            return 1;
        }
    }
    return 0;
}

/* True if a tool takes a URL argument — its args schema names a "url" key. */
static inline int agent_schema_wants_url(const char *s) {
    return s && strstr(s, "\"url\"");
}

/* True if a tool takes a note-slug argument — its schema names a "slug" key. */
static inline int agent_schema_wants_slug(const char *s) {
    return s && strstr(s, "\"slug\"");
}

/* True if a tool takes a filesystem-path argument — its schema names a
 * path/dir/file-ish key. */
static inline int agent_schema_wants_path(const char *s) {
    return s && (strstr(s, "\"path\"") || strstr(s, "\"dir\"") || strstr(s, "\"directory\"") ||
                 strstr(s, "\"filepath\"") || strstr(s, "\"file\""));
}

/* True if the request contains a slash-path word ("tests/data/x") that is not
 * a URL (the URL rule owns those). Bounded scan. */
static inline int agent_request_has_pathword(size_t req_len, const char *req) {
    for (size_t i = 0; i < req_len;) {
        while (i < req_len && (req[i] == ' ' || req[i] == '\t' || req[i] == '\n')) {
            i++;
        }
        size_t s     = i;
        int    slash = 0;
        while (i < req_len && req[i] != ' ' && req[i] != '\t' && req[i] != '\n') {
            slash |= req[i] == '/';
            i++;
        }
        if (i > s && slash && !agent_request_has_url(i - s, req + s)) {
            return 1;
        }
    }
    return 0;
}

/* True if the request talks about notes/memory — the evidence a memory tool
 * needs (Notiz, note, merke, remember, recall, erinnere, gespeichert). */
static inline int agent_request_mentions_memory(size_t req_len, const char *req) {
    static const char *const w[] = {"notiz",
                                    "note",
                                    "merk",
                                    "remember",
                                    "recall",
                                    "erinner",
                                    "gedächt",
                                    "gespeichert",
                                    "speicher"};
    for (size_t v = 0; v < sizeof w / sizeof *w; v++) {
        size_t wl = strlen(w[v]);
        for (size_t i = 0; i + wl <= req_len; i++) {
            if ((i == 0 || req[i - 1] == ' ' || req[i - 1] == '\t' || req[i - 1] == '\n') &&
                strncasecmp(req + i, w[v], wl) == 0) {
                return 1;
            }
        }
    }
    return 0;
}

/* True if a tool is a memory-palace tool (remember/recall). */
static inline int agent_tool_is_memory(const struct geist_tool *t) {
    return (t->name && (strstr(t->name, "remember") || strstr(t->name, "recall"))) ||
           (t->description && (strstr(t->description, "otiz") || strstr(t->description, "edächt")));
}

/* True if the request contains a slug-shaped word: all lowercase alnum with an
 * inner dash ("pi-serial" — how mind.h names notes). Bounded scan. */
static inline int agent_request_has_slug(size_t req_len, const char *req) {
    for (size_t i = 0; i < req_len;) {
        while (i < req_len && (req[i] == ' ' || req[i] == '\t' || req[i] == '\n')) {
            i++;
        }
        size_t s    = i;
        int    dash = 0, sluggy = 1;
        while (i < req_len && req[i] != ' ' && req[i] != '\t' && req[i] != '\n') {
            char c = req[i];
            if (c == '-') {
                dash = i > s && i + 1 < req_len && req[i + 1] != ' ';
            } else if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))) {
                sluggy = 0;
            }
            i++;
        }
        if (i > s && sluggy && dash) {
            return 1;
        }
    }
    return 0;
}

/* True if the request talks about the documentation corpus: a word starting
 * doc/Dok (docs, documentation, den Dokumenten). Bilingual and prefix-based
 * like agent_desc_is_dir; "docker"/"doctor" also match — harmless, only
 * consulted as a close-race tie-breaker. Bounded scan (req not NUL-terminated). */
static inline int agent_request_mentions_docs(size_t req_len, const char *req) {
    for (size_t i = 0; i + 3 <= req_len; i++) {
        if (i > 0 && req[i - 1] != ' ' && req[i - 1] != '\t' && req[i - 1] != '\n') {
            continue; /* word starts only */
        }
        if ((req[i] == 'd' || req[i] == 'D') && req[i + 1] == 'o' &&
            (req[i + 2] == 'c' || req[i + 2] == 'k')) {
            return 1;
        }
    }
    return 0;
}

/* True if a tool is the documentation tool — its name or description names
 * documents (doc_search, "die lokalen Dokumente", "the documents"). */
static inline int agent_tool_is_docs(const struct geist_tool *t) {
    return (t->name && strstr(t->name, "doc")) ||
           (t->description &&
            (strstr(t->description, "okument") || strstr(t->description, "ocument")));
}

/* True if the request opens with a destructive verb (delete / remove / lösche /
 * rm …), optionally after one politeness word (please / bitte). First-word only:
 * an imperative names the action, while a destructive word later in the request
 * is usually content ("search for how to remove noise"). ponytail: "Please
 * could you delete …" slips past the two-word scan — falls back to PMI. */
static inline int agent_request_is_destructive(size_t req_len, const char *req) {
    static const char *const verbs[] = {"delet",
                                        "remov",
                                        "eras",
                                        "wipe",
                                        "destroy",
                                        "kill",
                                        "drop",
                                        "rm",
                                        "lösch",
                                        "entfern",
                                        "zerstör",
                                        "vernicht"};
    size_t                   i       = 0;
    for (int word = 0; word < 2; word++) {
        while (i < req_len && (req[i] == ' ' || req[i] == '\t' || req[i] == '\n')) {
            i++;
        }
        size_t s = i;
        while (i < req_len && req[i] != ' ' && req[i] != '\t' && req[i] != '\n') {
            i++;
        }
        size_t wl = i - s;
        if (wl == 0) {
            return 0;
        }
        if ((wl == 6 && strncasecmp(req + s, "please", 6) == 0) ||
            (wl == 5 && strncasecmp(req + s, "bitte", 5) == 0)) {
            continue; /* politeness prefix — the verb is the next word */
        }
        for (size_t v = 0; v < sizeof verbs / sizeof *verbs; v++) {
            size_t vl = strlen(verbs[v]);
            if (wl >= vl && strncasecmp(req + s, verbs[v], vl) == 0) {
                return 1;
            }
        }
        return 0;
    }
    return 0;
}

/* True if some whitelisted tool actually covers destruction (its name or
 * description names such an action) — then a destructive request is a normal
 * routing case, not a refusal. */
static inline int agent_tools_cover_destruction(const struct geist_agent *a) {
    for (size_t i = 0; i < a->n_tools; i++) {
        const char *n = a->tools[i].name, *d = a->tools[i].description;
        if ((n && (strstr(n, "delet") || strstr(n, "remov") || strstr(n, "lösch"))) ||
            (d && (strstr(d, "delet") || strstr(d, "remov") || strstr(d, "lösch") ||
                   strstr(d, "entfern")))) {
            return 1;
        }
    }
    return 0;
}

/* Returns the routed tool's index, or -1 when the answer pseudo-entry wins —
 * the request is best served by replying directly, with no tool call at all. */
static inline int agent_select_tool(struct geist_agent *a, size_t req_len, const char *req) {
    /* Destructive-verb guard: an imperative the toolset cannot do routes
     * straight to the reply pseudo-entry — a refusal beats a forced-but-wrong
     * call ("Delete report.md" pulled doc_search up via the named file). Runs
     * before scoring: no model pass at all. */
    if (agent_request_is_destructive(req_len, req) && !agent_tools_cover_destruction(a)) {
        return -1;
    }
    if (a->n_tools <= 1) {
        return 0; /* no menu to rank; a single-tool agent always forces its tool */
    }
    const size_t n  = a->n_tools < AGENT_MAX_ROUTED - 1 ? a->n_tools : AGENT_MAX_ROUTED - 1;
    const size_t nn = n + 1; /* + the answer pseudo-entry, scored like a name */
    static char  sel[GEIST_AGENT_TRANSCRIPT_CAP];
    float        score[AGENT_MAX_ROUTED] = {0};

    /* Raw name logits have a token-frequency bias (a small model favours
     * "list_dir" regardless of the request). Calibrate: subtract the prior the
     * model assigns each name given the SAME menu but a content-free request, so
     * only the request-driven signal (PMI) decides. The baseline depends only on
     * the tool menu, not the request, so it is computed ONCE and cached on the
     * agent — every later route then does a single prefill (the request) instead
     * of two, halving routing latency (the Pi's light-task floor). */
    if (a->route_base_n != nn) {
        agent_select_prompt(a, n, strlen("(unspecified)"), "(unspecified)", sizeof sel, sel);
        a->route_base_n = agent_score_names(a, nn, sel, a->route_base) ? nn : 0;
    }
    int have_base = a->route_base_n == nn;

    agent_select_prompt(a, n, req_len, req, sizeof sel, sel);
    if (!agent_score_names(a, nn, sel, score)) {
        return 0;
    }

    float cal[AGENT_MAX_ROUTED] = {0};
    int   best                  = 0;
    float best_v                = -1e30f;
    for (size_t i = 0; i < nn; i++) { /* nn: the answer pseudo-entry competes too */
        cal[i] = have_base ? score[i] - a->route_base[i] : score[i];
        if (cal[i] > best_v) {
            best_v = cal[i], best = (int) i;
        }
    }

    /* Gate: a memory tool needs memory evidence. "recall" winning "What is
     * 2 plus 2?" is a calibration artifact — a rare name has a low prior, so
     * any raw bump inflates its PMI. Without a note/memory word the winner
     * re-routes to the best non-memory candidate (reply competes). Hard, no
     * window: a note is an explicitly named thing. */
    if (best < (int) n && agent_tool_is_memory(&a->tools[best]) &&
        !agent_request_mentions_memory(req_len, req)) {
        int   alt   = (int) n; /* the reply pseudo-entry */
        float alt_v = cal[n];
        for (size_t i = 0; i < n; i++) {
            if (!agent_tool_is_memory(&a->tools[i]) && cal[i] > alt_v) {
                alt = (int) i, alt_v = cal[i];
            }
        }
        best = alt, best_v = alt_v;
    }

    /* Tie-breaker: a literal http(s):// URL in the request is strong evidence
     * for the tool that takes a "url" arg (fetch), yet the name score often
     * favours the search tool. Same bounded close-race window as the file rule
     * below. Runs first: a URL also ends in a file-ish extension, so without
     * this the file rule would see ".html" and reason about the wrong axis. */
    if (agent_request_has_url(req_len, req) &&
        (best == (int) n || !agent_schema_wants_url(a->tools[best].args_schema))) {
        int   alt   = -1;
        float alt_v = -1e30f;
        for (size_t i = 0; i < n; i++) {
            if (agent_schema_wants_url(a->tools[i].args_schema) && cal[i] > alt_v) {
                alt = (int) i, alt_v = cal[i];
            }
        }
        if (alt >= 0 && alt_v > best_v - 1.5f) {
            best = alt, best_v = alt_v;
        }
    }

    /* Tie-breaker: a slash-path word (tests/data/x) is evidence for the tool
     * that takes a path-like arg — the URL rule's filesystem sibling. Skipped
     * when the slash word is a URL (the rule above owns that). */
    if (!agent_request_has_url(req_len, req) && agent_request_has_pathword(req_len, req) &&
        (best == (int) n || !agent_schema_wants_path(a->tools[best].args_schema))) {
        int   alt   = -1;
        float alt_v = -1e30f;
        for (size_t i = 0; i < n; i++) {
            if (agent_schema_wants_path(a->tools[i].args_schema) && cal[i] > alt_v) {
                alt = (int) i, alt_v = cal[i];
            }
        }
        if (alt >= 0 && alt_v > best_v - 1.5f) {
            best = alt, best_v = alt_v;
        }
    }

    /* Tie-breaker: a slug-shaped word (pi-serial — how mind.h names notes) is
     * strong evidence for the tool that takes a "slug" arg (recall), yet the
     * name score favours "remember" for either memory verb. Same bounded
     * close-race window as the URL rule. */
    if (agent_request_has_slug(req_len, req) &&
        (best == (int) n || !agent_schema_wants_slug(a->tools[best].args_schema))) {
        int   alt   = -1;
        float alt_v = -1e30f;
        for (size_t i = 0; i < n; i++) {
            if (agent_schema_wants_slug(a->tools[i].args_schema) && cal[i] > alt_v) {
                alt = (int) i, alt_v = cal[i];
            }
        }
        if (alt >= 0 && alt_v > best_v - 1.5f) {
            best = alt, best_v = alt_v;
        }
    }

    /* Tie-breaker: the request mentions the documentation corpus (docs/Dokumente)
     * but the winner is not the doc tool — first-token name scoring gives
     * doc_search only a weak margin over list_dir ("doc" vs the answer-ish
     * "list"), observed ~0.3-0.9 short. Skipped when the request names a
     * specific file: a named file is more specific evidence than the corpus
     * noun, and the file rule below owns that case. */
    if (!agent_request_names_file(req_len, req) && agent_request_mentions_docs(req_len, req) &&
        (best == (int) n || !agent_tool_is_docs(&a->tools[best]))) {
        int   alt   = -1;
        float alt_v = -1e30f;
        for (size_t i = 0; i < n; i++) {
            if (agent_tool_is_docs(&a->tools[i]) && cal[i] > alt_v) {
                alt = (int) i, alt_v = cal[i];
            }
        }
        if (alt >= 0 && alt_v > best_v - 1.5f) {
            best = alt, best_v = alt_v;
        }
    }

    /* Tie-breaker: if the winner is a directory tool but the request names a
     * specific file (note.txt), prefer the best non-dir tool within a small
     * window. The German separable verb "Fasse … zusammen" splits the verb so the
     * scored first token is "Fasse" and the name score barely favours list_dir; a
     * named file is strong evidence for a file tool. Bounded — only on a real
     * file extension (a bare directory path never fires) and only a close race. */
    if (best < (int) n && agent_desc_is_dir(a->tools[best].description) &&
        agent_request_names_file(req_len, req)) {
        int   alt   = -1;
        float alt_v = -1e30f;
        for (size_t i = 0; i < n; i++) {
            if (!agent_desc_is_dir(a->tools[i].description) && cal[i] > alt_v) {
                alt = (int) i, alt_v = cal[i];
            }
        }
        if (alt >= 0 && alt_v > best_v - 1.5f) { /* 1.5: the observed residual is ~0.5 */
            best = alt;
        }
    }
    /* An answer win deliberately survives the FILE rule (a named file in
     * "Delete report.md" is not a reason to force a tool), but not the URL/docs
     * rules above — those carry action evidence, not just an object. */
    return best == (int) n ? -1 : best;
}

/* A locator arg (a path/file/url) is something the user almost always wrote out
 * in the request — so for these keys we don't free-decode (a small model mangles
 * "/tmp/note.txt" into "/. tmp/ note. txt" via the tokenizer's leading-space
 * tokens) but LIFT the literal from the request. Recognises path/file/url-ish key
 * names. ponytail: a fixed name list, extend if a tool adds a new locator key. */
static inline int agent_key_is_locator(const char *key) {
    static const char *const loc[] = {"path", "file",      "filename", "dir",
                                       "url",  "directory", "filepath", "slug"};
    for (size_t i = 0; i < sizeof loc / sizeof *loc; i++) {
        if (strcmp(key, loc[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

/* Find the first whitespace-delimited word in the request that looks like a
 * locator — contains '/' (a path/URL: "/tmp/x", "http://host/p"), ends in a
 * file extension (".txt"/".md": "note.txt"), OR is slug-shaped (all lowercase
 * alnum with an inner dash: "pi-serial" — how mind.h names notes) — and copy
 * it to out. Returns its length, or 0 if the request has no such word.
 * ponytail: the slug shape also matches lowercase compounds ("on-device");
 * harmless — it is only consulted when the arg KEY is a locator (slug/path). */
static inline size_t
agent_extract_locator(size_t req_len, const char *req, size_t cap, char out[static cap]) {
    /* pass 0: slash/extension words (a real path/URL/file always wins);
     * pass 1: slug-shaped words — so "the on-device report.md" lifts report.md */
    for (int pass = 0; pass < 2; pass++) {
        for (size_t i = 0; i < req_len;) {
            while (i < req_len && (req[i] == ' ' || req[i] == '\t' || req[i] == '\n')) {
                i++;
            }
            size_t s = i, dot = (size_t) -1;
            int    slash = 0, dash = 0, sluggy = 1;
            while (i < req_len && req[i] != ' ' && req[i] != '\t' && req[i] != '\n') {
                char c = req[i];
                if (c == '/') {
                    slash = 1;
                } else if (c == '.') {
                    dot = i;
                }
                if (c == '-') {
                    dash = i > s && i + 1 < req_len && req[i + 1] != ' '; /* inner dash only */
                } else if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))) {
                    sluggy = 0;
                }
                i++;
            }
            int ext = 0; /* a dot, not leading, then 1-5 alnum to the word end */
            if (dot != (size_t) -1 && dot > s && i - dot >= 2 && i - dot <= 6) {
                ext = 1;
                for (size_t j = dot + 1; j < i; j++) {
                    char c = req[j];
                    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                          (c >= '0' && c <= '9'))) {
                        ext = 0;
                        break;
                    }
                }
            }
            int hit = pass == 0 ? (slash || ext) : (sluggy && dash);
            if (hit && i > s) {
                size_t n = i - s;
                if (n + 1 > cap) {
                    n = cap - 1;
                }
                memcpy(out, req + s, n);
                out[n] = '\0';
                return n;
            }
        }
    }
    return 0;
}

/* ---- recipe chains ----------------------------------------------------------
 * force_call is a single-shot tool-runner, so a two-step request ("search the
 * web AND READ the top result") used to die after step 0. A recipe extends
 * forcing to a known 2-step chain with the same philosophy as the routing
 * tie-breakers: deterministic, bounded, lexical. The router picks step 0 as
 * usual; the recipe fires only when (a) the routed tool is the recipe's first
 * step, (b) its second tool is on the whitelist, and (c) the request carries a
 * cue word asking for the second step ("… and read it" / "… fasse es
 * zusammen"). Step 1's locator arg is lifted from step 0's OBSERVATION — the
 * model decides nothing, and step 1 costs zero model time (a direct invoke).
 * ponytail: two fixed steps over the shipped tool names, top-hit extraction; the
 * upgrade is host-registered recipes + a cloze pick over the candidate lines. */
struct agent_recipe {
    const char *step0, *step1;
    const char *cues; /* space-separated word-start prefixes, lowercase */
};

static const struct agent_recipe AGENT_RECIPES[] = {
        {"web_search",
         "web_fetch",
         "read lies lese fetch hole open visit besuch page seite say summar fasse zusammen"},
        {"doc_search", "summarize_file", "summar fasse zusammen gist"},
        {"list_dir", "summarize_file", "summar fasse zusammen gist"},
};

/* Case-insensitive bounded needle search (req/obs are not NUL-terminated). */
static inline int agent_ci_find(size_t hlen, const char *hay, size_t nlen, const char *needle) {
    if (nlen == 0 || nlen > hlen) {
        return 0;
    }
    for (size_t i = 0; i + nlen <= hlen; i++) {
        if (strncasecmp(hay + i, needle, nlen) == 0) {
            return 1;
        }
    }
    return 0;
}

/* If a recipe continues routed tool `idx` — second tool whitelisted and a cue
 * word (word-start, case-insensitive prefix) present in the request — return
 * the second tool's index, else -1. */
static inline int
agent_recipe_next(struct geist_agent *a, int idx, size_t req_len, const char *req) {
    if (idx < 0 || (size_t) idx >= a->n_tools) {
        return -1;
    }
    for (size_t r = 0; r < sizeof AGENT_RECIPES / sizeof *AGENT_RECIPES; r++) {
        const struct agent_recipe *rc = &AGENT_RECIPES[r];
        if (strcmp(a->tools[idx].name, rc->step0) != 0) {
            continue;
        }
        int nxt = -1;
        for (size_t i = 0; i < a->n_tools; i++) {
            if (strcmp(a->tools[i].name, rc->step1) == 0) {
                nxt = (int) i;
                break;
            }
        }
        if (nxt < 0) {
            continue;
        }
        for (const char *c = rc->cues; *c != '\0';) {
            size_t cl = strcspn(c, " ");
            for (size_t i = 0; i + cl <= req_len; i++) {
                if ((i == 0 || req[i - 1] == ' ' || req[i - 1] == '\t' || req[i - 1] == '\n') &&
                    strncasecmp(req + i, c, cl) == 0) {
                    return nxt;
                }
            }
            c += cl;
            while (*c == ' ') {
                c++;
            }
        }
    }
    return -1;
}

/* Lift the step-1 locator from the step-0 observation. For a url-arg tool: the
 * first http(s):// URL (search results list the top hit first). Otherwise: the
 * first word that names a file — brackets/quotes around it are stripped
 * ("[report.md] …" is doc_search's hit format) — preferring a candidate whose
 * stem also appears in the request ("… the report" picks report.md over
 * readdir order). Returns the length, 0 if nothing liftable. */
static inline size_t agent_obs_locator(int         wants_url,
                                       size_t      on,
                                       const char *obs,
                                       size_t      req_len,
                                       const char *req,
                                       size_t      cap,
                                       char        out[static cap]) {
    if (wants_url) {
        for (size_t i = 0; i + 7 <= on; i++) {
            if (memcmp(obs + i, "http", 4) != 0) {
                continue;
            }
            size_t j = i + 4;
            if (j < on && obs[j] == 's') {
                j++;
            }
            if (j + 3 > on || memcmp(obs + j, "://", 3) != 0) {
                continue;
            }
            size_t e = i;
            while (e < on && obs[e] != ' ' && obs[e] != '\t' && obs[e] != '\n' && obs[e] != '"') {
                e++;
            }
            while (e > i && (obs[e - 1] == ')' || obs[e - 1] == ']' || obs[e - 1] == ',' ||
                             obs[e - 1] == '.' || obs[e - 1] == ';')) {
                e--; /* trailing prose punctuation is not part of the URL */
            }
            size_t n = e - i < cap - 1 ? e - i : cap - 1;
            memcpy(out, obs + i, n);
            out[n] = '\0';
            return n;
        }
        return 0;
    }
    size_t first_n = 0, pref_n = 0;
    char   first[256], pref[256];
    for (size_t i = 0; i < on;) {
        while (i < on && (obs[i] == ' ' || obs[i] == '\t' || obs[i] == '\n')) {
            i++;
        }
        size_t s = i;
        while (i < on && obs[i] != ' ' && obs[i] != '\t' && obs[i] != '\n') {
            i++;
        }
        size_t b = s, e = i; /* strip surrounding brackets/quotes */
        while (b < e && (obs[b] == '[' || obs[b] == '(' || obs[b] == '"' || obs[b] == '\'')) {
            b++;
        }
        while (e > b && (obs[e - 1] == ']' || obs[e - 1] == ')' || obs[e - 1] == '"' ||
                         obs[e - 1] == '\'' || obs[e - 1] == ',' || obs[e - 1] == ':')) {
            e--;
        }
        size_t dot = (size_t) -1; /* the file-extension shape of agent_extract_locator */
        for (size_t j = b; j < e; j++) {
            if (obs[j] == '.') {
                dot = j;
            }
        }
        int ext = dot != (size_t) -1 && dot > b && e - dot >= 2 && e - dot <= 6;
        for (size_t j = dot + 1; ext && j < e; j++) {
            char c = obs[j];
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))) {
                ext = 0;
            }
        }
        if (ext && e - b < sizeof first - 1) {
            if (first_n == 0) {
                memcpy(first, obs + b, e - b);
                first[e - b] = '\0';
                first_n      = e - b;
            }
            if (pref_n == 0 && agent_ci_find(req_len, req, dot - b, obs + b)) {
                memcpy(pref, obs + b, e - b);
                pref[e - b] = '\0';
                pref_n      = e - b;
                break; /* first request-matching candidate wins */
            }
        }
    }
    const char *pick   = pref_n ? pref : first;
    size_t      pick_n = pref_n ? pref_n : first_n;
    if (pick_n == 0 || pick_n + 1 > cap) {
        return 0;
    }
    memcpy(out, pick, pick_n + 1);
    return pick_n;
}

/* ---- in-sampler grammar masking (free mode) --------------------------------
 * The free loop used to hand the open turn to plain greedy decoding, and a
 * small model reliably mangles the call syntax — hallucinated tool names,
 * broken JSON, wrong keys (eval: free-mode routing 4/30). When greedy WANTS to
 * open a call (its next token starts '{', or a ```-fence, which in this
 * context is a wrapped call attempt), the turn is instead decoded along the
 * call grammar over the public peek/prefill API:
 *     {"tool":"<name in whitelist>","args":{"<key in schema>":"<value>"}}
 * The name decodes constrained to the whitelist, each arg KEY to the tool's
 * still-UNUSED schema keys (one remaining: prefilled outright), only the
 * VALUEs are free — each stops at its closing quote. At every pair boundary
 * the model chooses ',' (next key) or '}' (done) by logit, so a multi-key
 * schema fills as many keys as the model wants, each exactly once. The model
 * cannot emit an off-grammar call. ponytail: values are flat strings —
 * typed/nested argument grammars are the remaining upgrade. */

/* Prefill `lit` into the session AND append it to out at w; returns the new w.
 * Keeps the model conditioned on the scaffold it is "inside". */
static inline size_t agent_prefill_lit(
        struct geist_agent *a, const char *lit, size_t cap, char out[static cap], size_t w) {
    geist_token_t ids[64];
    size_t        nid = 0;
    if (geist_session_tokenize(a->session, lit, 64, ids, &nid) == GEIST_OK && nid > 0) {
        (void) geist_session_prefill_tokens(a->session, nid, ids);
    }
    w += (size_t) snprintf(out + w, w < cap ? cap - w : 0, "%s", lit);
    return w;
}

/* True if greedy's next token would open a tool call: '{', or a code fence
 * (models wrap calls in ```json). Peek only — nothing is consumed. */
static inline int agent_next_opens_call(struct geist_agent *a) {
    size_t       n_logits = 0;
    const float *logits   = geist_session_peek_logits(a->session, &n_logits);
    if (!logits || n_logits == 0) {
        return 0;
    }
    size_t best = 0;
    for (size_t i = 1; i < n_logits; i++) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    const char *piece = geist_session_token_to_str(a->session, (geist_token_t) best);
    while (piece && *piece == ' ') {
        piece++;
    }
    return piece && (*piece == '{' || *piece == '`');
}

/* Decode one guaranteed-valid tool call along the grammar into out. The
 * session must be at the open model turn. Returns the call's length, or 0 if
 * the constrained decode fell apart (caller resets and free-decodes). */
static inline size_t
agent_generate_call_masked(struct geist_agent *a, size_t cap, char out[static cap]) {
    size_t w   = agent_prefill_lit(a, "{\"tool\":\"", cap, out, 0);
    int    idx = agent_decode_name_constrained(a);
    if (idx < 0) {
        return 0;
    }
    w = (size_t) snprintf(out + w, cap - w, "%s", a->tools[idx].name) + w;
    w = agent_prefill_lit(a, "\",\"args\":", cap, out, w);

    char   keys[4][GEIST_AGENT_NAME_CAP];
    size_t nk = agent_schema_keys(a->tools[idx].args_schema, 4, keys);
    if (nk == 0) {
        w += (size_t) snprintf(out + w, cap - w, "{}}");
        return w;
    }

    /* key/value pairs: each key constrained to the still-unused schema keys,
     * each value free until its closing quote. The value's quote token may
     * fuse the model's continue/stop choice (`",` / `"}`) — honour it, else
     * ask the logits at the pair boundary. */
    int used[4] = {0};
    w           = agent_prefill_lit(a, "{", cap, out, w);
    for (size_t pair = 0; pair < nk && w + 8 < cap; pair++) {
        w = agent_prefill_lit(a, "\"", cap, out, w);
        const char *opts[4];
        int         map[4];
        size_t      rem = 0;
        for (size_t i = 0; i < nk; i++) {
            if (!used[i]) {
                map[rem]    = (int) i;
                opts[rem++] = keys[i];
            }
        }
        int k;
        if (rem == 1) {
            k = map[0];
            w = agent_prefill_lit(a, keys[k], cap, out, w);
        } else {
            int pick = agent_decode_pick(a, rem, opts);
            if (pick < 0) {
                return 0; /* partial key in the session — caller must reset */
            }
            k = map[pick];
            w = (size_t) snprintf(out + w, cap - w, "%s", keys[k]) + w;
        }
        used[k] = 1;
        w       = agent_prefill_lit(a, "\":\"", cap, out, w);

        /* the VALUE is the model's — free greedy, stopped at the closing
         * quote. end: 0 no quote seen, 1 bare quote, 2 quote+',', 3 quote+'}' */
        int    end    = 0;
        size_t vstart = w;
        for (int t = 0; t < 64 && w + 8 < cap && end == 0; t++) {
            geist_token_t tok = 0;
            if (geist_session_decode_step(a->session, &tok) != GEIST_OK || tok == a->eos ||
                (a->eot != GEIST_TOKEN_NONE && tok == a->eot)) {
                break;
            }
            const char *piece = geist_session_token_to_str(a->session, tok);
            size_t      plen  = piece ? strlen(piece) : 0;
            if (plen == 0 || (piece[0] == '<' && piece[plen - 1] == '>') || piece[0] == '}') {
                break; /* control marker / the model closing the object itself */
            }
            for (const char *p = piece; *p && w + 2 < cap; p++) {
                if (*p == '"') {
                    for (p++; *p == ' '; p++) { /* fused continue/stop choice? */
                    }
                    end = *p == ',' ? 2 : *p == '}' ? 3 : 1;
                    break;
                }
                char ch = (*p == '\n' || *p == '\t' || *p == '\r') ? ' ' : *p;
                if (ch == '\\') {
                    out[w++] = '\\'; /* keep the rebuilt JSON parseable */
                }
                out[w++] = ch;
            }
            if (agent_tail_loop(out + vstart, w - vstart) > 0) {
                break;
            }
        }
        while (w > vstart && out[w - 1] == ' ') {
            w--;
        }
        if (end == 0) {
            /* no clean quote: close the value ourselves and stop decoding —
             * the session is no longer aligned with the text past this point. */
            w += (size_t) snprintf(out + w, cap - w, "\"");
            break;
        }
        out[w++] = '"'; /* the session already consumed the model's quote */
        if (end == 3 || pair + 1 >= nk) {
            break; /* the model closed the object / no keys remain */
        }
        if (end == 2) {
            out[w++] = ','; /* fused: the model already committed to a comma */
            continue;
        }
        /* bare quote: the model chooses ',' (next key) or '}' (done) */
        size_t       nl = 0;
        const float *lg = geist_session_peek_logits(a->session, &nl);
        if (!lg ||
            agent_first_token_logit(a, "}", lg, nl) >= agent_first_token_logit(a, ",", lg, nl)) {
            break;
        }
        w = agent_prefill_lit(a, ",", cap, out, w);
    }
    w += (size_t) snprintf(out + w, cap - w, "}}");
    return w;
}

/* Force the next turn to be a valid call to tool `idx` (chosen by
 * agent_select_tool), whether or not the model would have emitted one — the
 * proof that prompted tool use does NOT require a tool-trained model. The JSON
 * scaffold + the selected tool name are prefilled token-by-token. A single-key
 * string arg's VALUE is lifted from the request (the slash-bearing word for a
 * locator key — see agent_extract_locator — else the request itself), not
 * free-decoded: a weak forced model mangles it. For a 0-key or multi-key schema
 * the args object is forced to {}. Builds the parseable call
 * into out and returns its length. Precondition: the session is set to the
 * transcript with the model turn open. Public peek/prefill/tokenize only — no
 * in-engine sampler change. */
static inline size_t agent_force_call(struct geist_agent *a,
                                      int                 idx,
                                      size_t              req_len,
                                      const char         *req,
                                      size_t              cap,
                                      char                out[static cap]) {
    if (idx < 0 || (size_t) idx >= a->n_tools) {
        if (cap) {
            out[0] = '\0';
        }
        return 0;
    }
    const struct geist_tool *t = &a->tools[idx];
    size_t                   w = 0;
    geist_token_t            ids[64];
    size_t                   nid = 0;
/* Append `lit` to out AND prefill it so the model's later decode is conditioned
 * on the scaffold it is "inside". */
#define AGENT_PREFILL(lit)                                                                 \
    do {                                                                                   \
        if (geist_session_tokenize(a->session, (lit), 64, ids, &nid) == GEIST_OK &&        \
            nid > 0) {                                                                     \
            (void) geist_session_prefill_tokens(a->session, nid, ids);                     \
        }                                                                                  \
        w += (size_t) snprintf(out + w, w < cap ? cap - w : 0, "%s", (lit));               \
    } while (0)

    AGENT_PREFILL("{\"tool\":\"");
    AGENT_PREFILL(t->name); /* the pre-selected tool (see agent_select_tool) */
    AGENT_PREFILL("\",\"args\":");

    char   keys[4][GEIST_AGENT_NAME_CAP];
    size_t nk = agent_schema_keys(t->args_schema, 4, keys);
    if (nk != 1) {
        w += (size_t) snprintf(out + w, w < cap ? cap - w : 0, "{}}"); /* close args + call */
    } else {
        char open[GEIST_AGENT_NAME_CAP + 8];
        snprintf(open, sizeof open, "{\"%s\":\"", keys[0]);
        AGENT_PREFILL(open);
        /* The VALUE is lifted from the request, not free-decoded: a weak forced
         * model mangles it (drops words from a query, splits a path). For a
         * locator key use the slash-bearing word; otherwise the request itself IS
         * the intent (a search query / question), so lift it whole — a search
         * engine handles the extra words. STRUCTURE and VALUE are both forced. */
        char        loc[512];
        const char *val = nullptr;
        size_t      vn  = 0;
        if (agent_key_is_locator(keys[0])) {
            /* a path/url arg: lift the locator word the user named, else default to
             * "." (a bare list_dir = cwd) — NOT the whole request, which is not a
             * path. */
            size_t locn = agent_extract_locator(req_len, req, sizeof loc, loc);
            val = locn > 0 ? loc : ".";
            vn  = locn > 0 ? locn : 1;
        } else {
            /* a free-text arg (a query / question): the request IS the intent. */
            val = req, vn = req_len;
        }
        /* prefill the value tokens to keep the session consistent (no append — we
         * write the JSON-escaped copy ourselves below). */
        geist_token_t vids[256];
        size_t        vnid = 0;
        if (geist_session_tokenize(a->session, val, 256, vids, &vnid) == GEIST_OK && vnid > 0) {
            (void) geist_session_prefill_tokens(a->session, vnid, vids);
        }
        for (size_t c = 0; c < vn && w + 2 < cap; c++) {
            char ch = val[c];
            if (ch == '\n' || ch == '\r' || ch == '\t') {
                ch = ' '; /* keep the JSON string on one line */
            }
            if (ch == '"' || ch == '\\') {
                out[w++] = '\\'; /* keep the rebuilt JSON valid */
            }
            out[w++] = ch;
        }
        w += (size_t) snprintf(out + w, w < cap ? cap - w : 0, "\"}}"); /* close value+args+call */
    }
#undef AGENT_PREFILL
    out[w < cap ? w : cap - 1] = '\0';
    return w < cap ? w : cap - 1;
}

/* Decode one assistant turn into out (greedy, stops on EOS / the template stop /
 * a degenerate repetition loop — see agent_tail_loop). Returns bytes written. */
static inline size_t agent_generate_turn(struct geist_agent *a, size_t cap, char out[static cap]) {
    return geist_generate_greedy(a->session, a->eos, a->eot, a->tmpl.leak, GEIST_AGENT_MAX_DECODE,
                                 cap, out);
}

/* Decode a REPLY turn: the router decided no tool fits, so the turn must be
 * prose — but the system prompt TEACHES the call format, and a plain greedy
 * reply drifts into call tokens ("What is 2 plus 2?" answered "{"; found by
 * the advisory judge). Two measures, the grammar-mask philosophy inverted:
 * a "Answer:" lead is prefilled (prose priming; it stays out of the returned
 * text — the transcript is the source of truth for follow-up turns), and
 * call-opening tokens ('{' / a fence) are banned at EVERY step, so a reply
 * can never contain a call fragment. Stops on EOS/EOT, control markers, a
 * template leak, or a degenerate repetition loop. */
static inline size_t agent_generate_reply(struct geist_agent *a, size_t cap, char out[static cap]) {
    geist_token_t lead[8];
    size_t        nlead = 0;
    if (geist_session_tokenize(a->session, "Answer:", 8, lead, &nlead) == GEIST_OK && nlead > 0) {
        (void) geist_session_prefill_tokens(a->session, nlead, lead);
    }
    size_t w    = 0;
    int    done = 0;
    for (int t = 0; t < GEIST_AGENT_MAX_DECODE && !done && w + 8 < cap; t++) {
        size_t       n_logits = 0;
        const float *logits   = geist_session_peek_logits(a->session, &n_logits);
        if (!logits || n_logits == 0) {
            break;
        }
        geist_token_t excl[8];
        size_t        n_excl = 0;
        while (n_excl < sizeof excl / sizeof *excl) {
            size_t best   = 0;
            float  best_v = -1e30f;
            for (size_t i = 0; i < n_logits; i++) {
                int banned = 0;
                for (size_t e = 0; e < n_excl; e++) {
                    banned |= excl[e] == (geist_token_t) i;
                }
                if (!banned && logits[i] > best_v) {
                    best_v = logits[i], best = i;
                }
            }
            geist_token_t tok = (geist_token_t) best;
            if (tok == a->eos || (a->eot != GEIST_TOKEN_NONE && tok == a->eot)) {
                done = 1;
                break;
            }
            const char *piece = geist_session_token_to_str(a->session, tok);
            size_t      plen  = piece ? strlen(piece) : 0;
            if (plen == 0 || (piece[0] == '<' && piece[plen - 1] == '>')) {
                done = 1; /* control marker — the turn is over */
                break;
            }
            if (strchr(piece, '{') || strchr(piece, '`') || strchr(piece, '\\')) {
                /* call-ish anywhere in the piece — prose needs no braces,
                 * fences, or escapes; banning them blocks the "call quoted as
                 * a string" drift, not just a bare leading '{' */
                excl[n_excl++] = tok;
                continue;
            }
            if (geist_session_prefill_tokens(a->session, 1, &tok) != GEIST_OK ||
                w + plen + 1 >= cap) {
                done = 1;
                break;
            }
            memcpy(out + w, piece, plen);
            w += plen;
            out[w] = '\0';
            if (agent_tail_loop(out, w) > 0) {
                done = 1; /* degenerate repetition */
            }
            break;
        }
        if (n_excl == sizeof excl / sizeof *excl) {
            break; /* nothing speakable in the top candidates */
        }
    }
    for (size_t m = 0; a->tmpl.leak[m] != nullptr; m++) {
        char *hit = strstr(out, a->tmpl.leak[m]);
        if (hit) {
            w    = (size_t) (hit - out); /* cut the leaked next-turn header */
            *hit = '\0';
        }
    }
    size_t lead_ws = 0; /* the lead ends in ':', the continuation starts spaced */
    while (lead_ws < w && (out[lead_ws] == ' ' || out[lead_ws] == '\n')) {
        lead_ws++;
    }
    if (lead_ws > 0) {
        memmove(out, out + lead_ws, w - lead_ws + 1);
        w -= lead_ws;
    }
    while (w > 0 && (out[w - 1] == ' ' || out[w - 1] == '\n')) {
        out[--w] = '\0';
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

/* Sliding-window context bound for multi-turn chat. When the transcript passes
 * GEIST_AGENT_CTX_BUDGET, evict the OLDEST conversation turns: keep the protected
 * system-prompt prefix [0..sys_len) and the most recent whole turns (snapped to a
 * user_open marker), down to GEIST_AGENT_CTX_TARGET. A turn_close is inserted
 * after the prefix so the kept history is well-framed (the system instructions
 * become their own user turn). This bounds per-turn re-prefill — a long chat
 * stays O(n), not O(n^2) — and replaces the hard "context full" stop.
 *
 * No-op outside conversation mode / before a system prompt exists. The model
 * forgets the evicted turns.
 * ponytail: hard drop. To keep the gist instead, fold the evicted span
 * [sys_len..keep_from) into a running summary via summ_generate and splice it in
 * as "[system][summary][recent]" here — the summarizer already exists. */
static inline void agent_compact(struct geist_agent *a) {
    if (!a->conversation || a->sys_len == 0 || a->tlen <= GEIST_AGENT_CTX_BUDGET) {
        return;
    }
    const char *uo  = a->tmpl.user_open;
    size_t      uol = strlen(uo);
    size_t      tcl = strlen(a->tmpl.turn_close);
    /* tail bytes we may keep after the prefix + the inserted turn_close */
    size_t budget_tail =
            (GEIST_AGENT_CTX_TARGET > a->sys_len + tcl) ? GEIST_AGENT_CTX_TARGET - a->sys_len - tcl : 0;
    /* earliest user_open whose tail fits the target (keep the most history);
     * else the latest one (keep only the newest turn — guarantees progress).
     * Scan from sys_len + tcl, not sys_len: a marker within tcl bytes of the
     * prefix would make the memmove destination (sys_len + tcl) overtake the
     * source and shift the body FORWARD past the buffer end. Real turn
     * boundaries are always well past the prefix, so this only skips a marker
     * the user happened to paste into the first request. */
    size_t earliest_fit = 0, latest = 0;
    for (size_t i = a->sys_len + tcl; i + uol <= a->tlen; i++) {
        if (memcmp(a->transcript + i, uo, uol) == 0) {
            latest = i;
            if (earliest_fit == 0 && a->tlen - i <= budget_tail) {
                earliest_fit = i;
            }
        }
    }
    size_t keep_from = earliest_fit ? earliest_fit : latest;
    if (keep_from == 0) {
        return; /* no conversation-turn boundary past the prefix to cut at */
    }
    size_t body = a->tlen - keep_from;
    memmove(a->transcript + a->sys_len + tcl, a->transcript + keep_from, body + 1); /* +NUL */
    memcpy(a->transcript + a->sys_len, a->tmpl.turn_close, tcl);
    a->tlen = a->sys_len + tcl + body;
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

/* A final-answer turn with fewer than 2 alphanumeric chars (e.g. a lone "{" or
 * "}" — a small model fumbling the post-observation turn by starting another JSON
 * call instead of prose). When this happens after a tool ran, surfacing the
 * tool's observation is far more useful than the junk. */
static inline int agent_answer_degenerate(const char *s) {
    int alnum = 0;
    for (; *s != '\0'; s++) {
        if ((*s >= 'a' && *s <= 'z') || (*s >= 'A' && *s <= 'Z') || (*s >= '0' && *s <= '9')) {
            if (++alnum >= 2) {
                return 0;
            }
        }
    }
    return 1;
}

/* In conversation mode, fold the final answer into the transcript (closing the
 * open model turn) so the NEXT geist_agent_run sees it — every return path
 * must do this, including the forced/recipe and repeat-guard exits, or a
 * follow-up turn lands on an unclosed model turn. Only advances tlen if the
 * fold fit (else the turn just goes unrecorded, never corrupted). */
static inline void agent_conv_fold(struct geist_agent *a, const char *answer) {
    if (!a->conversation) {
        return;
    }
    int fw = snprintf(a->transcript + a->tlen,
                      sizeof a->transcript - a->tlen,
                      "%s%s",
                      answer,
                      a->tmpl.turn_close);
    if (fw > 0 && (size_t) fw < sizeof a->transcript - a->tlen) {
        a->tlen += (size_t) fw;
    }
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

    if (a->conversation && a->tlen > 0) {
        /* multi-turn chat: the transcript already holds the prior turns (ending
         * after a closed model turn). Evict the oldest turns if we're over budget,
         * then open a fresh user turn and continue. */
        agent_compact(a);
        a->tlen += (size_t) snprintf(a->transcript + a->tlen,
                                     sizeof a->transcript - a->tlen,
                                     "%s%.*s%s%s",
                                     a->tmpl.user_open,
                                     (int) req_len,
                                     req,
                                     a->tmpl.turn_close,
                                     a->tmpl.model_open);
    } else {
        /* one-shot (or the first chat turn): system prompt opens the user turn. */
        a->tlen    = agent_system_prompt(a, sizeof a->transcript, a->transcript);
        a->sys_len = a->tlen; /* protected prefix: never evicted by agent_compact */
        a->tlen += (size_t) snprintf(a->transcript + a->tlen,
                                     sizeof a->transcript - a->tlen,
                                     "%.*s%s%s",
                                     (int) req_len,
                                     req,
                                     a->tmpl.turn_close,
                                     a->tmpl.model_open);
    }

    char turn[GEIST_AGENT_TURN_CAP];
    char name[GEIST_AGENT_NAME_CAP];
    char args[GEIST_AGENT_ARGS_CAP];
    char obs[GEIST_AGENT_OBS_CAP];
    /* (name,args) hashes of the calls already executed this run — a repeated
     * call cannot progress, its observation is already in the transcript. */
    unsigned seen[16];
    size_t   n_seen = 0;

    for (size_t step = 0; step < a->max_steps; step++) {
        if (geist_session_reset(a->session) != GEIST_OK ||
            geist_session_set_prompt(a->session, a->transcript) != GEIST_OK) {
            return GEIST_E_INVALID_STATE;
        }
        /* ponytail: full-transcript reprefill each step — O(n^2) over the
         * request. Switch to incremental prefill_tokens if requests get long. */
        /* force_call: make turn 0 a guaranteed-valid tool call even on a model
         * that wouldn't emit one (no tool training needed); later turns are free
         * so the model can give a plain-text answer after the observation. The
         * tool is ROUTED (agent_select_tool) so multiple tools don't mis-fire;
         * selection clobbers the session, so re-set the transcript before forcing. */
        size_t tn;
        if (a->force_call && step == 0) {
            agent_emit(a, GEIST_AGENT_ROUTING, step, nullptr, nullptr);
            int idx = agent_select_tool(a, req_len, req);
            agent_emit(a,
                       GEIST_AGENT_ROUTING,
                       step,
                       idx >= 0 ? a->tools[idx].name : AGENT_ANSWER_NAME,
                       "selected");
            if (geist_session_reset(a->session) != GEIST_OK ||
                geist_session_set_prompt(a->session, a->transcript) != GEIST_OK) {
                return GEIST_E_INVALID_STATE;
            }
            /* idx < 0: the answer pseudo-entry won the routing — no tool fits.
             * The decision is TERMINAL: decode a call-proof reply turn and
             * return it as the answer (the model fills text, it does not get
             * to overrule the router with a call after all). */
            if (idx < 0) {
                size_t rn = agent_generate_reply(a, sizeof turn, turn);
                (void) rn;
                agent_emit(a, GEIST_AGENT_ANSWERING, step, nullptr, turn);
                agent_conv_fold(a, turn);
                size_t n = agent_copy(resp_cap, resp, turn);
                if (resp_len) {
                    *resp_len = n;
                }
                return GEIST_OK;
            }
            tn = agent_force_call(a, idx, req_len, req, sizeof turn, turn);
        } else {
            /* free turn under the grammar mask: if greedy would open a call,
             * decode it along the call grammar (guaranteed valid, on-whitelist);
             * plain text stays plain. A fallen-apart constrained decode left
             * scaffold tokens in the session — reset before free-decoding. */
            tn = 0;
            if (a->n_tools > 0 && agent_next_opens_call(a)) {
                tn = agent_generate_call_masked(a, sizeof turn, turn);
                if (tn == 0 && (geist_session_reset(a->session) != GEIST_OK ||
                                geist_session_set_prompt(a->session, a->transcript) != GEIST_OK)) {
                    return GEIST_E_INVALID_STATE;
                }
            }
            if (tn == 0) {
                tn = agent_generate_turn(a, sizeof turn, turn);
            }
        }

        if (!agent_parse_call(tn, turn, sizeof name, name, sizeof args, args)) {
            /* no tool call -> this turn is the final answer. If a tool already ran
             * this request (step > 0) and the model fumbled the answer turn into a
             * degenerate fragment (a lone "{"), surface the last observation
             * instead — the tool's output is what the user actually asked for. */
            const char *answer = (step > 0 && agent_answer_degenerate(turn)) ? obs : turn;
            agent_emit(a, GEIST_AGENT_ANSWERING, step, nullptr, answer);
            agent_conv_fold(a, answer);
            size_t n = agent_copy(resp_cap, resp, answer);
            if (resp_len) {
                *resp_len = n;
            }
            return GEIST_OK;
        }

        /* A call we already EXECUTED this run cannot progress — it reproduces
         * an observation the transcript already holds, and a grammar-masked
         * free turn otherwise wanders call -> obs -> call until max_steps
         * (including A-B-A alternation). The last observation IS the answer.
         * ponytail: djb2 over name+args; a collision merely answers early. */
        unsigned h = 5381;
        for (const char *p = name; *p; p++) {
            h = h * 33u + (unsigned char) *p;
        }
        for (const char *p = args; *p; p++) {
            h = h * 33u + (unsigned char) *p;
        }
        int repeat = 0;
        for (size_t i = 0; i < n_seen; i++) {
            repeat |= seen[i] == h;
        }
        if (repeat && step > 0) {
            agent_emit(a, GEIST_AGENT_ANSWERING, step, nullptr, obs);
            agent_conv_fold(a, obs);
            size_t n = agent_copy(resp_cap, resp, obs);
            if (resp_len) {
                *resp_len = n;
            }
            return GEIST_OK;
        }
        if (n_seen < sizeof seen / sizeof *seen) {
            seen[n_seen++] = h;
        }

        agent_emit(a, GEIST_AGENT_CALLING, step, name, args);

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
            agent_emit(a, GEIST_AGENT_RUNNING, step, t->name, nullptr);
            agent_args_normalize(t->args_schema, sizeof args, args); /* re-key to the schema */
            if (t->invoke(t->ctx, strlen(args), args, sizeof obs, obs, &on) != GEIST_OK) {
                on = (size_t) snprintf(obs, sizeof obs, "error: tool \"%s\" failed", name);
            }
        }
        agent_emit(a, GEIST_AGENT_OBSERVED, step, t ? t->name : name, obs);

        /* force_call is a single-task tool-runner: once the routed tool has run,
         * its observation IS the answer (the listing / the summary). Return it
         * directly rather than letting a weak model fumble a free answer turn or
         * fire more tools. (The normal, un-forced loop keeps composing answers.)
         * Exception: a recipe (see agent_recipe_next) continues a known 2-step
         * chain — the follow-up tool runs directly on a locator lifted from this
         * observation, and ITS observation is the answer. No model in the loop. */
        if (a->force_call && step == 0) {
            int nxt = t ? agent_recipe_next(a, (int) (t - a->tools), req_len, req) : -1;
            if (nxt >= 0) {
                const struct geist_tool *t1 = &a->tools[nxt];
                char                     keys[4][GEIST_AGENT_NAME_CAP];
                char                     loc[512];
                size_t ln = agent_obs_locator(agent_schema_wants_url(t1->args_schema),
                                              on,
                                              obs,
                                              req_len,
                                              req,
                                              sizeof loc,
                                              loc);
                if (ln > 0 && agent_schema_keys(t1->args_schema, 4, keys) == 1) {
                    char   args1[GEIST_AGENT_ARGS_CAP];
                    size_t aw = (size_t) snprintf(args1, sizeof args1, "{\"%s\":\"", keys[0]);
                    for (size_t c = 0; c < ln && aw + 4 < sizeof args1; c++) {
                        if (loc[c] == '"' || loc[c] == '\\') {
                            args1[aw++] = '\\';
                        }
                        args1[aw++] = loc[c];
                    }
                    aw += (size_t) snprintf(args1 + aw, sizeof args1 - aw, "\"}");
                    agent_emit(a, GEIST_AGENT_CALLING, step + 1, t1->name, args1);
                    agent_emit(a, GEIST_AGENT_RUNNING, step + 1, t1->name, nullptr);
                    size_t o1 = 0;
                    if (t1->invoke(t1->ctx, aw, args1, sizeof obs, obs, &o1) == GEIST_OK) {
                        on = o1;
                    } else {
                        on = (size_t) snprintf(
                                obs, sizeof obs, "error: tool \"%s\" failed", t1->name);
                    }
                    agent_emit(a, GEIST_AGENT_OBSERVED, step + 1, t1->name, obs);
                }
                /* nothing liftable / multi-key schema -> fall through: step 0's
                 * observation is still a useful answer (the hit list itself). */
            }
            agent_emit(a, GEIST_AGENT_ANSWERING, step, nullptr, obs);
            agent_conv_fold(a, obs);
            size_t n = agent_copy(resp_cap, resp, obs);
            if (resp_len) {
                *resp_len = n;
            }
            return GEIST_OK;
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

/* Human label for a phase (stable, ASCII). */
static inline const char *geist_agent_phase_name(enum geist_agent_phase p) {
    switch (p) {
    case GEIST_AGENT_ROUTING: return "routing";
    case GEIST_AGENT_CALLING: return "calling";
    case GEIST_AGENT_RUNNING: return "running";
    case GEIST_AGENT_OBSERVED: return "observed";
    case GEIST_AGENT_ANSWERING: return "answering";
    }
    return "?";
}

/* Ready-made on_event handler: print one friendly progress line per step to the
 * FILE* passed as ctx (use stderr so it doesn't mix into the answer on stdout).
 * Wire it with: a.on_event = agent_event_print; a.on_event_ctx = stderr; */
static inline void agent_event_print(void *ctx, const struct geist_agent_event *ev) {
    FILE       *f      = ctx ? (FILE *) ctx : stderr;
    const char *glyph  = ev->phase == GEIST_AGENT_ROUTING     ? "·"
                         : ev->phase == GEIST_AGENT_CALLING   ? "\xe2\x86\x92"  /* → */
                         : ev->phase == GEIST_AGENT_RUNNING   ? "\xe2\x9a\x99"  /* ⚙ */
                         : ev->phase == GEIST_AGENT_OBSERVED  ? "\xe2\x9c\x93"  /* ✓ */
                                                              : "\xe2\x97\x8f"; /* ● */
    fprintf(f, "%s %s", glyph, geist_agent_phase_name(ev->phase));
    if (ev->tool) {
        fprintf(f, " %s", ev->tool);
    }
    if (ev->detail && ev->detail[0]) {
        fprintf(f, ": %.80s%s", ev->detail, strlen(ev->detail) > 80 ? "…" : "");
    }
    fputc('\n', f);
    fflush(f);
}

#endif /* GEIST_AGENT_H */
