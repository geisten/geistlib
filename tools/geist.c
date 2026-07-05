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
 *   geist <model.gguf> [prompt] [-n N]          # generate text
 *   geist agent <model.gguf> [request] [-n N]   # tool-use agent (REPL if no request)
 *   geist --version
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
#include "agent_summarize.h"
#include "agent_webfetch.h"
#include "agent_websearch.h"
#include "mind.h"

/* Open backend + model (+ greedy session). path == nullptr -> the embedded model.
 * Returns 0 on success; logs and cleans up on failure. */
static int cli_open(const char *path, struct geist_backend **be, struct geist_model **model,
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
                (size_t) (geist_embedded_model_end - geist_embedded_model_start), *be, model);
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
        "{\"tool\":\"summarize_file\",\"args\":{\"path\":\"<file>\"}}. To search local documents reply "
        "with {\"tool\":\"doc_search\",\"args\":{\"query\":\"<query>\"}}. To search the web reply with "
        "{\"tool\":\"web_search\",\"args\":{\"query\":\"<query>\"}}. To read a web page reply with "
        "{\"tool\":\"web_fetch\",\"args\":{\"url\":\"<url>\"}}. To save a note reply with "
        "{\"tool\":\"remember\",\"args\":{\"text\":\"<note>\"}}. To load a saved note reply with "
        "{\"tool\":\"recall\",\"args\":{\"slug\":\"<slug>\"}}. After the tool result, "
        "answer the user in one or two sentences.";

/* Built after the model loads (summarize_file's sub-session needs model+backend).
 * doc_search scans GEIST_DOCS (default ./docs); web_fetch's nullptr allowlist =
 * any http/https, fine for a local demo — tighten via webfetch_tool("host,..."). */
static size_t agent_tools(struct geist_model *model, struct geist_backend *be,
                          struct geist_tool *out, size_t cap, void *ctx) {
    (void) cap;
    (void) ctx;
    static struct summarize_ctx sctx;
    sctx             = (struct summarize_ctx) {.model = model, .be = be, .root = "."};
    const char *docs = getenv("GEIST_DOCS");
    out[0]           = listdir_tool();
    out[1]           = summarize_file_tool(&sctx);
    out[2]           = docsearch_tool(docs && docs[0] ? docs : "./docs");
    out[3]           = websearch_tool(nullptr);
    out[4]           = webfetch_tool(nullptr);
    size_t n         = 5;
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

static int run_chat(int argc, char **argv) {
    /* argv here starts at "chat"; the model path is argv[1]. */
    if (argc == 2 && strcmp(argv[1], "--selftest") == 0) {
        return chat_selftest();
    }
    /* Non-embedded: `geist chat <model>`. Embedded: `geist chat` (baked-in model). */
    const char *model_path = nullptr;
    if (geist_embedded) {
        if (argc != 1) {
            fprintf(stderr, "usage: geist chat   (or: geist chat --selftest)\n");
            return 2;
        }
    } else {
        if (argc != 2) {
            fprintf(stderr, "usage: geist chat <model.gguf>   (or: geist chat --selftest)\n");
            return 2;
        }
        model_path = argv[1];
    }

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
            static char  note[CHAT_NOTE_CAP];
            static char  entry[CHAT_NOTE_CAP + 256]; /* > note + prefix: format can't truncate */
            const char  *slug = line + 8;
            if (mind_recall(slug, note, sizeof note) > 0) {
                /* Format the entry into a buffer sized to hold a full note + the
                 * prefix, then append to `pending` only if it fits — so the format
                 * write is bounded and the append is an explicit, guarded copy. */
                int    el = snprintf(entry, sizeof entry, "Recalled note %.200s:\n%s\n", slug, note);
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
            size_t rn = 0;
            if (geist_agent_run(&agent, strlen(req), req, sizeof resp, resp, &rn) != GEIST_OK) {
                fprintf(stderr, "chat: turn failed\n");
            } else {
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
        "  %s [prompt] [-n N]              generate text\n"
        "  %s agent [request] [-n N]       one-shot tool-use agent\n"
        "  %s chat                         multi-turn chat + memory palace\n"
        "  %s --version\n\n"
        "Options:\n"
        "  -n, --max-tokens N   max new tokens to generate (default 64)\n"
        "  -v, --version        print version and exit\n"
        "  -h, --help           print this help and exit\n\n"
        "Example:\n"
        "  OMP_WAIT_POLICY=active %s \"The capital of France is\" -n 40\n",
        geist_version_string(), prog, prog, prog, prog, prog);
#else
    fprintf(o,
        "geist %s — minimal CPU LLM inference\n\n"
        "Usage:\n"
        "  %s <model.gguf> [prompt] [-n N]        generate text\n"
        "  %s agent <model.gguf> [request] [-n N] one-shot tool-use agent\n"
        "  %s chat <model.gguf>                   multi-turn chat + memory palace\n"
        "  %s --version\n\n"
        "Options:\n"
        "  -n, --max-tokens N   max new tokens to generate (default 64)\n"
        "  -v, --version        print version and exit\n"
        "  -h, --help           print this help and exit\n\n"
        "Example:\n"
        "  OMP_WAIT_POLICY=active %s model.gguf \"The capital of France is\" -n 40\n",
        geist_version_string(), prog, prog, prog, prog, prog);
#endif
    return code;
}

int main(int argc, char **argv) {
    /* `geist agent ...` -> one-shot tool loop. Drop argv[0]; geist_agent_main
     * parses "agent" as its prog name and the rest after it. In an embedded build
     * the model is baked in (pass its bounds; no model positional). */
    if (argc > 1 && strcmp(argv[1], "agent") == 0) {
#if defined(GEIST_EMBEDDED_MODEL)
        return geist_agent_main(argc - 1, argv + 1, system_with_index(AGENT_SYSTEM), agent_tools,
                                nullptr, geist_embedded_model_start, geist_embedded_model_end);
#else
        return geist_agent_main(argc - 1, argv + 1, system_with_index(AGENT_SYSTEM), agent_tools,
                                nullptr, nullptr, nullptr);
#endif
    }
    /* `geist chat ...` -> multi-turn conversation with the same tools + memory. */
    if (argc > 1 && strcmp(argv[1], "chat") == 0) {
        return run_chat(argc - 1, argv + 1);
    }
    const char *prog = "geist";
    const char *model_path = nullptr;
    const char *prompt = "Hello, my name is";
    int  max_new    = 64;
    bool n_explicit = false; /* explicit -n is a hard cap; the default is a soft target */
    int  got_prompt = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            return usage(prog, 0);
        } else if (strcmp(a, "-v") == 0 || strcmp(a, "--version") == 0) {
            printf("geist %s\n", geist_version_string());
            return 0;
        } else if (strcmp(a, "-n") == 0 || strcmp(a, "--max-tokens") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "%s: %s needs an argument\n", prog, a); return 2; }
            max_new = atoi(argv[++i]);
            if (max_new <= 0) { fprintf(stderr, "%s: invalid token count\n", prog); return 2; }
            n_explicit = true;
        } else if (a[0] == '-' && a[1] != '\0') {
            fprintf(stderr, "%s: unknown option '%s'\n", prog, a);
            return usage(prog, 2);
        } else if (!geist_embedded && model_path == nullptr) {
            model_path = a;
        } else if (!got_prompt) {
            prompt = a; got_prompt = 1;
        } else {
            fprintf(stderr, "%s: unexpected argument '%s'\n", prog, a);
            return usage(prog, 2);
        }
    }
    if (!geist_embedded && model_path == nullptr) return usage(prog, 2);

    /* "auto" picks the best backend compiled into this build for the host. */
    struct geist_backend *be = nullptr;
    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK) {
        fprintf(stderr, "backend_create failed: %s\n", geist_last_create_error());
        return 1;
    }

    struct geist_model *model = nullptr;
    enum geist_status ls;
    const char *src;
#if defined(GEIST_EMBEDDED_MODEL)
    ls  = geist_model_load_from_memory(
        geist_embedded_model_start,
        (size_t) (geist_embedded_model_end - geist_embedded_model_start), be, &model);
    src = "<embedded>";
#else
    ls  = geist_model_load(model_path, be, &model);
    src = model_path;
#endif
    if (ls != GEIST_OK) {
        fprintf(stderr, "model_load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        return 1;
    }
    fprintf(stderr, "loaded %s (arch: %s)\n", src, geist_model_arch(model));

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
        const size_t   cap  = strlen(prompt) + 8; /* one token/byte upper bound + BOS */
        size_t         np   = 0;
        geist_token_t *tmp  = malloc(cap * sizeof(geist_token_t));
        if (tmp != nullptr &&
            geist_session_tokenize(sess, prompt, cap, tmp, &np) == GEIST_OK) {
            const size_t need = np + (size_t) budget + 8;
            /* 4096 = the state-default window (arch_state.c). Only rebuild
             * when the workload exceeds it, so short/medium prompts keep
             * the single-session fast path. */
            if (need > 4096u) {
                geist_session_destroy(sess);
                opts.max_seq_len = need;
                sess             = nullptr;
                if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
                    fprintf(stderr, "session_create (ctx=%zu) failed: %s\n", need,
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

    if (geist_session_set_prompt(sess, prompt) != GEIST_OK) {
        fprintf(stderr, "set_prompt failed: %s\n", geist_session_errmsg(sess));
        geist_session_destroy(sess); geist_model_destroy(model); geist_backend_destroy(be);
        return 1;
    }

    printf("%s", prompt);
    fflush(stdout);
    for (int i = 0; i < budget; i++) {
        geist_token_t tok = 0;
        if (geist_session_decode_step(sess, &tok) != GEIST_OK) {
            fprintf(stderr, "\ndecode_step failed: %s\n", geist_session_errmsg(sess));
            break;
        }
        const char *piece = geist_session_token_to_str(sess, tok);
        if (piece == nullptr) break;
        size_t len = strlen(piece);
        if (len >= 2 && piece[0] == '<' && piece[len - 1] == '>') break; /* control/special */
        fputs(piece, stdout);
        fflush(stdout);
        if (!n_explicit && i + 1 >= max_new && len > 0) { /* past soft target: stop at sentence end */
            char last = piece[len - 1];
            if (last == '.' || last == '!' || last == '?' || last == '\n') break;
        }
    }
    putchar('\n');

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return 0;
}
