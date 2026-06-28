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

/* The `agent` subcommand is a normal-build feature; an embedded single-file
 * binary stays text-only (and lean — no curl/opendir code linked in). */
#if !defined(GEIST_EMBEDDED_MODEL)
#include "agent_docsearch.h"
#include "agent_listdir.h"
#include "agent_main.h"
#include "agent_summarize.h"
#include "agent_webfetch.h"
#include "agent_websearch.h"

static const char *AGENT_SYSTEM =
        "You are a file and web assistant. To see a directory's contents reply with "
        "{\"tool\":\"list_dir\",\"args\":{\"path\":\".\"}}. To summarize a file reply with "
        "{\"tool\":\"summarize_file\",\"args\":{\"path\":\"<file>\"}}. To search local documents reply "
        "with {\"tool\":\"doc_search\",\"args\":{\"query\":\"<query>\"}}. To search the web reply with "
        "{\"tool\":\"web_search\",\"args\":{\"query\":\"<query>\"}}. To read a web page reply with "
        "{\"tool\":\"web_fetch\",\"args\":{\"url\":\"<url>\"}}. After the tool result, "
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
    return 5;
}
#endif /* !GEIST_EMBEDDED_MODEL */

/* When built with `make EMBED_MODEL=...`, the GGUF is baked into the binary
 * (embedded_model.S) and the CLI takes only a prompt — no model-path argument. */
#if defined(GEIST_EMBEDDED_MODEL)
extern const unsigned char geist_embedded_model_start[];
extern const unsigned char geist_embedded_model_end[];
enum { geist_embedded = 1 };
#else
enum { geist_embedded = 0 };
#endif

static int usage(const char *prog, int code) {
    FILE *o = code ? stderr : stdout;
#if defined(GEIST_EMBEDDED_MODEL)
    fprintf(o,
        "geist %s — minimal CPU LLM inference (model embedded in this binary)\n\n"
        "Usage:\n"
        "  %s [prompt] [-n N]\n"
        "  %s --version\n\n"
        "Options:\n"
        "  -n, --max-tokens N   max new tokens to generate (default 64)\n"
        "  -v, --version        print version and exit\n"
        "  -h, --help           print this help and exit\n\n"
        "Example:\n"
        "  OMP_WAIT_POLICY=active %s \"The capital of France is\" -n 40\n",
        geist_version_string(), prog, prog, prog);
#else
    fprintf(o,
        "geist %s — minimal CPU LLM inference\n\n"
        "Usage:\n"
        "  %s <model.gguf> [prompt] [-n N]        generate text\n"
        "  %s agent <model.gguf> [request] [-n N] tool-use agent (REPL if no request)\n"
        "  %s --version\n\n"
        "Options:\n"
        "  -n, --max-tokens N   max new tokens to generate (default 64)\n"
        "  -v, --version        print version and exit\n"
        "  -h, --help           print this help and exit\n\n"
        "Example:\n"
        "  OMP_WAIT_POLICY=active %s model.gguf \"The capital of France is\" -n 40\n",
        geist_version_string(), prog, prog, prog, prog);
#endif
    return code;
}

int main(int argc, char **argv) {
#if !defined(GEIST_EMBEDDED_MODEL)
    /* `geist agent ...` -> the tool-use loop. Drop argv[0]; geist_agent_main then
     * parses "agent" as its prog name and <model>/<request>/-n after it. */
    if (argc > 1 && strcmp(argv[1], "agent") == 0) {
        return geist_agent_main(argc - 1, argv + 1, AGENT_SYSTEM, agent_tools, nullptr);
    }
#endif
    const char *prog = "geist";
    const char *model_path = nullptr;
    const char *prompt = "Hello, my name is";
    int max_new = 64;
    int got_prompt = 0;

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

    /* Zero-initialized opts == greedy decode (temperature 0). */
    struct geist_session_opts opts = {0};
    struct geist_session *sess = nullptr;
    if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
        fprintf(stderr, "session_create failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return 1;
    }

    if (geist_session_set_prompt(sess, prompt) != GEIST_OK) {
        fprintf(stderr, "set_prompt failed: %s\n", geist_session_errmsg(sess));
        geist_session_destroy(sess); geist_model_destroy(model); geist_backend_destroy(be);
        return 1;
    }

    printf("%s", prompt);
    fflush(stdout);

    for (int i = 0; i < max_new; i++) {
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
    }
    putchar('\n');

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return 0;
}
