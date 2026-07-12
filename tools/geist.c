/*
 * geist.c — the geist command-line interface.
 *
 * Default mode loads a GGUF model and greedy-decodes a text continuation to
 * stdout, using only the STABLE core of the public API (include/geist.h):
 * backend -> model -> session -> set_prompt -> decode_step -> token_to_str.
 *
 * The `agent` subcommand runs the whitelist-gated tool loop instead (the same
 * model can list a directory, summarize a file, search local docs, and search /
 * read the web). It honours GEIST_FORCE_CALL=1 and GEIST_AGENT_TRACE=1.
 *
 *   geist [-m <model>] "prompt"                 # ask — instruct chat (DEFAULT): wraps
 *                                                 the prompt in the model's chat template,
 *                                                 clean answer, offline, no tools
 *   geist [-m <model>]                          # no prompt -> interactive agentic chat
 *   geist [-m <model>] --raw "prompt"           # raw base-model text completion
 *   geist agent [-m <model>] "request"          # one-shot tool-use agent
 *   geist --version
 * Model source: -m <path>  >  an embedded model  >  error.
 *
 * For an embeddable text example see examples/simple_generate.c.
 */
#define _POSIX_C_SOURCE 200809L /* the agent's opendir/fork-based tools */

#include <geist.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* When built with `make EMBED_MODEL=...`, the GGUF is baked into the binary
 * (embedded_model.S); the CLI then takes a prompt/request with no model path. */
#if defined(GEIST_EMBEDDED_MODEL)
extern const unsigned char geist_embedded_model_start[];
extern const unsigned char geist_embedded_model_end[];
enum { geist_embedded = 1 };
#else
enum { geist_embedded = 0 };
#endif

/* The `agent` and `chat` subcommands and their tools. Always compiled — an
 * embedded build gets them too, driving the baked-in model (the 1+ GB model
 * dwarfs the few KB of tool code, so the old "embedded = text-only" split bought
 * nothing). */
#include "agent_docsearch.h"
#include "agent_listdir.h"
#include "agent_main.h"
#include "agent_memory.h"
#include "agent_stocks.h"
#include "agent_summarize.h"
#include "agent_webfetch.h"
#include "agent_websearch.h"
#include "mind.h"

/* Open backend + model (+ greedy session). path == nullptr -> the embedded model.
 * Returns 0 on success; logs and cleans up on failure. */
static int cli_open(const char            *path,
                    struct geist_backend **be,
                    struct geist_model   **model,
                    struct geist_session **sess) {
    *be    = nullptr;
    *model = nullptr;
    *sess  = nullptr;
    if (geist_backend_create("auto", nullptr, nullptr, be) != GEIST_OK) {
        fprintf(stderr, "backend_create failed: %s\n", geist_last_create_error());
        return 1;
    }
    enum geist_status ls;
    if (path != nullptr) {
        ls = geist_model_load(path, *be, model);
    } else {
#if defined(GEIST_EMBEDDED_MODEL)
        ls = geist_model_load_from_memory(
                geist_embedded_model_start,
                (size_t) (geist_embedded_model_end - geist_embedded_model_start),
                *be,
                model);
#else
        fprintf(stderr, "no model path given and this build has no embedded model\n");
        geist_backend_destroy(*be);
        return 1;
#endif
    }
    if (ls != GEIST_OK) {
        fprintf(stderr, "model_load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(*be);
        return 1;
    }
    struct geist_session_opts opts = {0}; /* greedy, deterministic */
    if (geist_session_create(*model, *be, &opts, sess) != GEIST_OK) {
        fprintf(stderr, "session_create failed\n");
        geist_model_destroy(*model);
        geist_backend_destroy(*be);
        return 1;
    }
    return 0;
}

static const char *AGENT_SYSTEM =
        "You are a file, web and memory assistant. To see a directory's contents reply with "
        "{\"tool\":\"list_dir\",\"args\":{\"path\":\".\"}}. To summarize a file reply with "
        "{\"tool\":\"summarize_file\",\"args\":{\"path\":\"<file>\"}}. To search local documents "
        "reply "
        "with {\"tool\":\"doc_search\",\"args\":{\"query\":\"<query>\"}}. To search the web reply "
        "with "
        "{\"tool\":\"web_search\",\"args\":{\"query\":\"<query>\"}}. To read a web page reply with "
        "{\"tool\":\"web_fetch\",\"args\":{\"url\":\"<url>\"}}. To save a note reply with "
        "{\"tool\":\"remember\",\"args\":{\"text\":\"<note>\"}}. To load a saved note reply with "
        "{\"tool\":\"recall\",\"args\":{\"slug\":\"<slug>\"}}. After the tool result, "
        "answer the user in one or two sentences.";

/* Built after the model loads (summarize_file's sub-session needs model+backend).
 * doc_search scans GEIST_DOCS (default ./docs); web_fetch's nullptr allowlist =
 * any http/https, fine for a local demo — tighten via webfetch_tool("host,..."). */
static size_t agent_tools(struct geist_model   *model,
                          struct geist_backend *be,
                          struct geist_tool    *out,
                          size_t                cap,
                          void                 *ctx) {
    (void) cap;
    (void) ctx;
    static struct summarize_ctx sctx;
    sctx             = (struct summarize_ctx) {.model = model, .be = be, .root = "."};
    const char *docs = getenv("GEIST_DOCS");
    out[0]           = listdir_tool();
    out[1]           = summarize_file_tool(&sctx);
    out[2]           = docsearch_tool(docs && docs[0] ? docs : "./docs");
    /* DuckDuckGo rate-limits quickly; GEIST_SEARX_ENDPOINT points web_search at
     * a SearXNG instance instead (same knob as the agent eval / live smoke). */
    out[3]   = websearch_tool(getenv("GEIST_SEARX_ENDPOINT"));
    out[4]   = webfetch_tool(nullptr);
    out[5]   = stock_movers_tool(nullptr);
    size_t n = 6;
    /* Memory tools are opt-in: include them only when a palace is configured
     * (GEIST_MIND_DIR). On weak models the router scores tool NAMES, and adding
     * remember/recall to the default set makes common requests (e.g. "summarize
     * report.md") mis-route to recall on some CPU backends (seen on BitNet/NEON).
     * `geist chat`'s /remember,/recall slash commands work regardless of this. */
    if (getenv("GEIST_MIND_DIR")) {
        out[n++] = remember_tool();
        out[n++] = recall_tool();
    }
    return n;
}

/* Returns a system prompt = base + the notes index (so recall is usable: the
 * model sees the available slugs). Static buffer — valid for the process. */
static const char *system_with_index(const char *base) {
    static char sys[1 << 13];
    char        ipath[1100];
    static char index[1 << 12];
    snprintf(ipath, sizeof ipath, "%s/INDEX.md", mind_dir());
    if (mind_slurp(ipath, index, sizeof index) > 0) {
        snprintf(sys, sizeof sys, "%s\nYour memory palace (stored notes):\n%s", base, index);
    } else {
        snprintf(sys, sizeof sys, "%s", base);
    }
    return sys;
}

/* `geist chat` — multi-turn conversation on the agent engine (conversation mode),
 * with the same toolset plus reliable /slash control over the memory palace. */
enum { CHAT_LINE_CAP = 8192, CHAT_RESP_CAP = 1 << 13, CHAT_NOTE_CAP = 1 << 15 };

static void chat_rstrip(char *s) {
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == ' ' || s[n - 1] == '\t')) {
        s[--n] = '\0';
    }
}

static int chat_selftest(void) {
    setenv("GEIST_MIND_DIR", "./.mind_selftest", 1);
    static char buf[CHAT_NOTE_CAP];
    int         ok = mind_remember("Test Note", "hello world contents") == 0 &&
                     mind_recall("test-note", buf, sizeof buf) > 0 &&
                     strstr(buf, "hello world contents") != nullptr &&
                     mind_slurp("./.mind_selftest/INDEX.md", buf, sizeof buf) > 0 &&
                     strstr(buf, "[Test Note](test-note.md)") != nullptr;
    remove("./.mind_selftest/test-note.md");
    remove("./.mind_selftest/INDEX.md");
    remove("./.mind_selftest");
    puts(ok ? "geist chat selftest ok" : "geist chat selftest FAILED");
    return ok ? 0 : 1;
}

/* True if an agent turn's answer is a tool failure — the signal for the offline
 * fallback. Keyed on our tools' "error:" prefix convention (a failed web fetch
 * offline surfaces exactly this). Deliberately narrow: matching looser words
 * ("curl", "unreachable") would false-trigger on a legitimate answer that merely
 * mentions them. */
static int chat_tool_failed(const char *resp) {
    return strncmp(resp, "error:", 6) == 0;
}

static void chat_prompt_append(char *dst, size_t cap, size_t *len, const char *src) {
    if (*len >= cap) {
        return;
    }
    size_t room = cap - *len - 1;
    size_t n    = strnlen(src, room);
    memcpy(dst + *len, src, n);
    *len += n;
    dst[*len] = '\0';
}

/* Offline fallback: answer `req` directly (instruct, no tools) on a THROWAWAY
 * session, leaving the agent's own session/transcript untouched. Used when an
 * agent tool fails (e.g. the web is unreachable) so the REPL still answers from
 * the model's own knowledge instead of surfacing "fetch failed". Returns bytes.
 * ponytail: a fresh session per fallback (rare failure path); the fallback turn
 * is dropped from the conversation transcript rather than folded back in. */
static size_t chat_instruct_fallback(struct geist_model   *model,
                                     struct geist_backend *be,
                                     const char           *req,
                                     size_t                cap,
                                     char                  out[static cap]) {
    struct geist_session_opts opts = {0};
    struct geist_session     *s    = nullptr;
    if (geist_session_create(model, be, &opts, &s) != GEIST_OK) {
        return 0;
    }
    struct geist_chat_template tmpl = geist_chat_template_for_model(model);
    static char                buf[1 << 14];
    size_t                     len = 0;
    buf[0]                         = '\0';
    chat_prompt_append(buf, sizeof buf, &len, tmpl.user_open);
    chat_prompt_append(buf, sizeof buf, &len, req);
    chat_prompt_append(buf, sizeof buf, &len, tmpl.turn_close);
    chat_prompt_append(buf, sizeof buf, &len, tmpl.model_open);
    size_t w = 0;
    if (geist_session_set_prompt(s, buf) == GEIST_OK) {
        geist_token_t eos = geist_model_eos_token(model);
        geist_token_t eot =
                tmpl.stop[0] ? geist_model_token_by_text(model, tmpl.stop) : GEIST_TOKEN_NONE;
        w = geist_generate_greedy(s, eos, eot, tmpl.leak, GEIST_AGENT_MAX_DECODE, cap, out);
    }
    geist_session_destroy(s);
    return w;
}

/* The interactive agentic REPL on a resolved model (model_path == nullptr ->
 * embedded). Shared by bare `geist` (no prompt) and the `geist chat` alias. */
static int run_chat(const char *model_path) {

    struct geist_backend *be    = nullptr;
    struct geist_model   *model = nullptr;
    struct geist_session *sess  = nullptr;
    if (cli_open(model_path, &be, &model, &sess) != 0) {
        return 1;
    }

    struct geist_tool tools[AGENT_MAIN_TOOLS_CAP];
    size_t            n_tools = agent_tools(model, be, tools, AGENT_MAIN_TOOLS_CAP, nullptr);
    static struct geist_agent agent;
    geist_agent_init(&agent, model, sess, n_tools, tools, 0, system_with_index(AGENT_SYSTEM));
    agent.conversation = true; /* keep the transcript across turns */
    /* chat: trace is opt-in (a conversation stays quiet by default — set
     * GEIST_AGENT_TRACE=1 to watch tool steps). The one-shot `geist agent` traces
     * by default; see agent_trace_enabled. */
    const char *chat_trace = getenv("GEIST_AGENT_TRACE");
    if (chat_trace != nullptr && strcmp(chat_trace, "0") != 0) {
        agent.on_event     = agent_event_print;
        agent.on_event_ctx = stderr;
    }

    fprintf(stderr, "loaded %s — /help for commands, /quit to exit\n", geist_model_arch(model));
    static char line[CHAT_LINE_CAP];
    static char pending[CHAT_NOTE_CAP]; /* notes /recall-ed, prepended to next turn */
    static char resp[CHAT_RESP_CAP];
    char        ipath[1100];
    snprintf(ipath, sizeof ipath, "%s/INDEX.md", mind_dir());
    fputs("> ", stdout);
    fflush(stdout);
    while (fgets(line, sizeof line, stdin)) {
        chat_rstrip(line);
        if (line[0] == '\0') {
            fputs("> ", stdout);
            fflush(stdout);
            continue;
        }
        if (strcmp(line, "/quit") == 0 || strcmp(line, "/exit") == 0) {
            break;
        }
        if (strcmp(line, "/help") == 0) {
            puts("/remember <title> | <text>   /recall <slug>   /notes   /quit");
            puts("(the model can also call remember/recall itself)");
        } else if (strcmp(line, "/notes") == 0) {
            static char idx[CHAT_NOTE_CAP];
            puts(mind_slurp(ipath, idx, sizeof idx) > 0 ? idx : "(no notes yet)");
        } else if (strncmp(line, "/remember ", 10) == 0) {
            char *bar = strchr(line + 10, '|');
            if (!bar) {
                puts("usage: /remember <title> | <text>");
            } else {
                *bar        = '\0';
                char *title = line + 10, *text = bar + 1;
                chat_rstrip(title);
                while (*text == ' ') {
                    text++;
                }
                puts(mind_remember(title, text) == 0 ? "remembered." : "remember failed.");
            }
        } else if (strncmp(line, "/recall ", 8) == 0) {
            static char note[CHAT_NOTE_CAP];
            static char entry[CHAT_NOTE_CAP + 256]; /* > note + prefix: format can't truncate */
            const char *slug = line + 8;
            if (mind_recall(slug, note, sizeof note) > 0) {
                /* Format the entry into a buffer sized to hold a full note + the
                 * prefix, then append to `pending` only if it fits — so the format
                 * write is bounded and the append is an explicit, guarded copy. */
                int el = snprintf(entry, sizeof entry, "Recalled note %.200s:\n%s\n", slug, note);
                size_t pl = strlen(pending);
                if (el > 0 && pl + (size_t) el + 1 <= sizeof pending) {
                    memcpy(pending + pl, entry, (size_t) el + 1);
                    printf("recalled %s — it will be in context for your next message\n", slug);
                } else {
                    puts("(recall context full — /quit and start over)");
                }
            } else {
                printf("no note '%s'\n", slug);
            }
        } else {
            /* a chat turn: prepend any /recall-ed notes, run one agent turn. */
            static char req[CHAT_NOTE_CAP + CHAT_LINE_CAP];
            if (pending[0]) {
                snprintf(req, sizeof req, "%s\n%s", pending, line);
                pending[0] = '\0';
            } else {
                snprintf(req, sizeof req, "%s", line);
            }
            size_t rn         = 0;
            size_t saved_tlen = agent.tlen; /* to drop a failed tool turn cleanly */
            if (geist_agent_run(&agent, strlen(req), req, sizeof resp, resp, &rn) != GEIST_OK) {
                fprintf(stderr, "chat: turn failed\n");
            } else {
                if (chat_tool_failed(resp)) {
                    /* a tool failed (likely offline) — rewind the transcript to
                     * before this turn and answer directly from the model. */
                    agent.tlen                   = saved_tlen;
                    agent.transcript[agent.tlen] = '\0';
                    size_t fn = chat_instruct_fallback(model, be, req, sizeof resp, resp);
                    if (fn > 0) {
                        rn = fn;
                    }
                }
                fwrite(resp, 1, rn, stdout);
                putchar('\n');
            }
        }
        fputs("> ", stdout);
        fflush(stdout);
    }

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return 0;
}

static int usage(const char *prog, int code) {
    FILE *o = code ? stderr : stdout;
#if defined(GEIST_EMBEDDED_MODEL)
    fprintf(o,
            "geist %s — minimal CPU LLM inference (model embedded in this binary)\n\n"
            "Usage:\n"
            "  %s \"prompt\"                     ask a question (instruct chat)\n"
            "  %s                              no prompt -> interactive agentic chat\n"
            "  %s --raw \"prompt\"               raw base-model text completion\n"
            "  %s agent \"request\"              one-shot tool-use agent\n"
            "  %s --version\n\n"
            "Options:\n"
            "  -m, --model <path>   run a different GGUF instead of the baked-in model\n"
            "  --raw                raw base-model completion (default is instruct chat:\n"
            "                       the prompt is wrapped in the model's chat template)\n"
            "  -n, --max-tokens N   max new tokens to generate (default 64)\n"
            "  -v, --version        print version and exit\n"
            "  -h, --help           print this help and exit\n\n"
            "Example:\n"
            "  OMP_WAIT_POLICY=active %s \"What is the capital of France?\"\n",
            geist_version_string(),
            prog,
            prog,
            prog,
            prog,
            prog,
            prog);
#else
    fprintf(o,
            "geist %s — minimal CPU LLM inference\n\n"
            "Usage:\n"
            "  %s -m <model> [prompt]                 ask a question (instruct chat)\n"
            "  %s -m <model>                          no prompt -> interactive agentic chat\n"
            "  %s -m <model> --raw [prompt]           raw base-model text completion\n"
            "  %s agent -m <model> \"request\"          one-shot tool-use agent\n"
            "  %s --version\n\n"
            "Options:\n"
            "  -m, --model <path>   the model GGUF to load (required unless embedded)\n"
            "  --raw                raw base-model completion (default is instruct chat:\n"
            "                       the prompt is wrapped in the model's chat template)\n"
            "  -n, --max-tokens N   max new tokens to generate (default 64)\n"
            "  -v, --version        print version and exit\n"
            "  -h, --help           print this help and exit\n\n"
            "Example:\n"
            "  OMP_WAIT_POLICY=active %s -m model.gguf \"What is the capital of France?\"\n",
            geist_version_string(),
            prog,
            prog,
            prog,
            prog,
            prog,
            prog);
#endif
    return code;
}

int main(int argc, char **argv) {
    /* `geist agent ...` -> one-shot tool loop. Drop argv[0]; geist_agent_main
     * parses "agent" as its prog name and the rest after it. In an embedded build
     * the model is baked in (pass its bounds; no model positional). */
    if (argc > 1 && strcmp(argv[1], "agent") == 0) {
#if defined(GEIST_EMBEDDED_MODEL)
        return geist_agent_main(argc - 1,
                                argv + 1,
                                system_with_index(AGENT_SYSTEM),
                                agent_tools,
                                nullptr,
                                geist_embedded_model_start,
                                geist_embedded_model_end);
#else
        return geist_agent_main(argc - 1,
                                argv + 1,
                                system_with_index(AGENT_SYSTEM),
                                agent_tools,
                                nullptr,
                                nullptr,
                                nullptr);
#endif
    }
    /* `geist chat [-m <model>]` -> the agentic REPL; explicit alias for bare
     * `geist` with no prompt. Carries --selftest and a legacy positional model. */
    if (argc > 1 && strcmp(argv[1], "chat") == 0) {
        if (argc == 3 && strcmp(argv[2], "--selftest") == 0) {
            return chat_selftest();
        }
        const char *m = nullptr;
        for (int i = 2; i < argc; i++) {
            if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
                m = argv[++i];
            } else if (argv[i][0] != '-' && m == nullptr) {
                m = argv[i]; /* legacy positional model */
            }
        }
        if (!geist_embedded && m == nullptr) {
            fprintf(stderr, "geist chat: no model — pass -m <model.gguf>\n");
            return 2;
        }
        return run_chat(m);
    }

    /* Default dispatch: `geist [-m <model>] [prompt] [-n N] [--raw]`.
     *   prompt given -> one-shot instruct answer (--raw = base-model completion);
     *   no prompt    -> the interactive agentic REPL (run_chat).
     * Model source: -m <path>  >  embedded  >  error. */
    const char *prog       = "geist";
    const char *model_path = nullptr; /* from -m; embedded build ignores it */
    const char *prompt     = nullptr; /* one positional; nullptr -> REPL */
    int         max_new    = 64;
    bool        n_explicit = false; /* explicit -n is a hard cap; the default is a soft target */
    bool        chat_mode  = true;  /* default: instruct chat (template-wrapped); --raw opts out */

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            return usage(prog, 0);
        } else if (strcmp(a, "-v") == 0 || strcmp(a, "--version") == 0) {
            printf("geist %s\n", geist_version_string());
            return 0;
        } else if (strcmp(a, "-m") == 0 || strcmp(a, "--model") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s: %s needs an argument\n", prog, a);
                return 2;
            }
            model_path = argv[++i];
        } else if (strcmp(a, "-n") == 0 || strcmp(a, "--max-tokens") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s: %s needs an argument\n", prog, a);
                return 2;
            }
            max_new = atoi(argv[++i]);
            if (max_new <= 0) {
                fprintf(stderr, "%s: invalid token count\n", prog);
                return 2;
            }
            n_explicit = true;
        } else if (strcmp(a, "--raw") == 0) {
            chat_mode = false;
        } else if (strcmp(a, "-c") == 0 || strcmp(a, "--chat") == 0) {
            chat_mode = true; /* explicit; instruct chat is already the default */
        } else if (a[0] == '-' && a[1] != '\0') {
            fprintf(stderr, "%s: unknown option '%s'\n", prog, a);
            return usage(prog, 2);
        } else if (prompt == nullptr) {
            prompt = a; /* the single positional is the PROMPT (model is -m now) */
        } else {
            fprintf(stderr, "%s: unexpected argument '%s'\n", prog, a);
            return usage(prog, 2);
        }
    }
    /* Model: -m, else embedded, else it's an error (there is nothing to run). */
    if (!geist_embedded && model_path == nullptr) {
        fprintf(stderr,
                "%s: no model — pass -m <model.gguf>, or use a build with an embedded model\n",
                prog);
        return usage(prog, 2);
    }
    /* No prompt -> the interactive agentic REPL. */
    if (prompt == nullptr) {
        return run_chat(model_path);
    }

    /* "auto" picks the best backend compiled into this build for the host. */
    struct geist_backend *be = nullptr;
    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK) {
        fprintf(stderr, "backend_create failed: %s\n", geist_last_create_error());
        return 1;
    }

    /* Model precedence: an explicit -m path wins even in an embedded build
     * (-m > embedded > error, matching the arg check above). */
    struct geist_model *model = nullptr;
    enum geist_status   ls;
    const char         *src;
    if (model_path != nullptr) {
        ls  = geist_model_load(model_path, be, &model);
        src = model_path;
    } else {
#if defined(GEIST_EMBEDDED_MODEL)
        ls = geist_model_load_from_memory(
                geist_embedded_model_start,
                (size_t) (geist_embedded_model_end - geist_embedded_model_start),
                be,
                &model);
        src = "<embedded>";
#else
        ls  = GEIST_E_INVALID_STATE; /* unreachable: guarded above */
        src = "<none>";
#endif
    }
    if (ls != GEIST_OK) {
        fprintf(stderr, "model_load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        return 1;
    }
    fprintf(stderr, "loaded %s (arch: %s)\n", src, geist_model_arch(model));

    /* Instruct chat is the DEFAULT (most GGUFs people run are instruct-tuned):
     * wrap the prompt in the model's chat template so the CLI gives a clean
     * answer that stops at the turn marker — the same wrapper the agent/serve
     * layer uses, offline and tool-free. --raw opts out to base-model completion
     * (a bare continuation, for base models or a template geist can't detect). */
    const char                *gen_prompt = prompt;
    struct geist_chat_template tmpl       = {0};
    static char                chat_buf[1 << 14];
    if (chat_mode) {
        tmpl = geist_chat_template_for_model(model);
        snprintf(chat_buf,
                 sizeof chat_buf,
                 "%s%s%s%s",
                 tmpl.user_open,
                 prompt,
                 tmpl.turn_close,
                 tmpl.model_open);
        gen_prompt = chat_buf;
    }

    /* max_new is a hard cap when the user passed -n; otherwise it's a soft target:
     * keep going past it until a sentence ends (capped at 2x), so a bare completion
     * prompt — the base model never emits an end token for one — stops on a clean
     * boundary instead of mid-word. */
    int budget = n_explicit ? max_new : max_new * 2;

    /* Zero-initialized opts == greedy decode (temperature 0). Size the KV
     * window to prompt + decode budget so long prompts clear the default
     * 4096 ceiling (otherwise prefill returns GEIST_E_TOO_MANY_TOKENS).
     * A throwaway default session tokenizes first — tokenize touches no KV
     * cache, so this only pays for the tokenizer pass. */
    struct geist_session_opts opts = {0};
    struct geist_session     *sess = nullptr;
    if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
        fprintf(stderr, "session_create failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return 1;
    }
    {
        const size_t   cap = strlen(gen_prompt) + 8; /* one token/byte upper bound + BOS */
        size_t         np  = 0;
        geist_token_t *tmp = malloc(cap * sizeof(geist_token_t));
        if (tmp != nullptr && geist_session_tokenize(sess, gen_prompt, cap, tmp, &np) == GEIST_OK) {
            const size_t need = np + (size_t) budget + 8;
            /* 4096 = the state-default window (arch_state.c). Only rebuild
             * when the workload exceeds it, so short/medium prompts keep
             * the single-session fast path. */
            if (need > 4096u) {
                geist_session_destroy(sess);
                opts.max_seq_len = need;
                sess             = nullptr;
                if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
                    fprintf(stderr,
                            "session_create (ctx=%zu) failed: %s\n",
                            need,
                            geist_last_create_error());
                    free(tmp);
                    geist_model_destroy(model);
                    geist_backend_destroy(be);
                    return 1;
                }
            }
        }
        free(tmp);
    }

    if (geist_session_set_prompt(sess, gen_prompt) != GEIST_OK) {
        fprintf(stderr, "set_prompt failed: %s\n", geist_session_errmsg(sess));
        geist_session_destroy(sess);
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return 1;
    }

    if (chat_mode) {
        /* Instruct path: generate the assistant turn only, stopping at the model's
         * EOS or the template's turn terminator; print just the answer (no echo of
         * the templated prompt). Reuses the agent/summarizer greedy generator. */
        geist_token_t eos = geist_model_eos_token(model);
        geist_token_t eot =
                tmpl.stop[0] ? geist_model_token_by_text(model, tmpl.stop) : GEIST_TOKEN_NONE;
        static char answer[1 << 14];
        geist_generate_greedy(sess, eos, eot, tmpl.leak, budget, sizeof answer, answer);
        puts(answer);
    } else {
        printf("%s", prompt);
        fflush(stdout);
        for (int i = 0; i < budget; i++) {
            geist_token_t tok = 0;
            if (geist_session_decode_step(sess, &tok) != GEIST_OK) {
                fprintf(stderr, "\ndecode_step failed: %s\n", geist_session_errmsg(sess));
                break;
            }
            const char *piece = geist_session_token_to_str(sess, tok);
            if (piece == nullptr)
                break;
            size_t len = strlen(piece);
            if (len >= 2 && piece[0] == '<' && piece[len - 1] == '>')
                break; /* control/special */
            fputs(piece, stdout);
            fflush(stdout);
            if (!n_explicit && i + 1 >= max_new &&
                len > 0) { /* soft target: stop at sentence end */
                char last = piece[len - 1];
                if (last == '.' || last == '!' || last == '?' || last == '\n')
                    break;
            }
        }
        putchar('\n');
    }

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return 0;
}
