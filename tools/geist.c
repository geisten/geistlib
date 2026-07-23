/*
 * geist.c — the geist command-line interface.
 *
 * Loads a GGUF model and greedy-decodes a text continuation to stdout, using
 * only the STABLE core of the public API (include/geist.h):
 * backend -> model -> session -> set_prompt -> decode_step -> token_to_str.
 *
 * The `--serve` flag hands the process to the resident dynamic-tools daemon
 * (agent_main.h): the model stays warm and a host supplies the complete toolset
 * per request over a Unix socket (dynamic-tools-v1). The engine CLI compiles NO
 * tools of its own — concrete tools live in the consumer (e.g. geistwissen, the
 * Home Assistant integration). See docs/agent.md.
 *
 *   geist [-m <model>] "prompt"        # ask — instruct chat (DEFAULT): wraps the
 *                                        prompt in the model's chat template
 *   geist [-m <model>] --raw "prompt"  # raw base-model text completion
 *   geist [-m <model>] --serve <sock>  # resident dynamic-tools daemon
 *   geist --version
 * Model source: -m <path>  >  an embedded model  >  error.
 *
 * For an embeddable text example see examples/simple_generate.c.
 */
#define _POSIX_C_SOURCE 200809L /* agent_main.h's --serve socket daemon */

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

/* The reusable agent CLI engine — used here ONLY for `--serve` (the resident
 * dynamic-tools daemon). Tools arrive per request from the host, so the served
 * agent needs no compiled toolset (no_tools below). */
#include "agent_main.h"

/* --serve compiles no tools of its own: the dynamic-tools host supplies the
 * complete allowed toolset with every request. */
static size_t no_tools(struct geist_model   *model,
                       struct geist_backend *be,
                       struct geist_tool    *out,
                       size_t                cap,
                       void                 *ctx) {
    (void) model;
    (void) be;
    (void) out;
    (void) cap;
    (void) ctx;
    return 0;
}

/* Base system prompt for a served conversation. The daemon appends the host's
 * per-request language + context; the host also supplies the tools. */
static const char *SERVE_SYSTEM =
        "You are a helpful assistant. Use the provided tools when they help, then "
        "answer the user in one or two sentences.";

static int usage(const char *prog, int code) {
    FILE *o = code ? stderr : stdout;
#if defined(GEIST_EMBEDDED_MODEL)
    fprintf(o,
            "geist %s — minimal CPU LLM inference (model embedded in this binary)\n\n"
            "Usage:\n"
            "  %s \"prompt\"                     ask a question (instruct chat)\n"
            "  %s --raw \"prompt\"               raw base-model text completion\n"
            "  %s --serve <socket>             resident dynamic-tools daemon\n"
            "  %s --version\n\n"
            "Options:\n"
            "  -m, --model <path>   run a different GGUF instead of the baked-in model\n"
            "  --raw                raw base-model completion (default is instruct chat:\n"
            "                       the prompt is wrapped in the model's chat template)\n"
            "  --serve <socket>     serve dynamic-tools-v1 over a chmod-600 Unix socket\n"
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
            prog);
#else
    fprintf(o,
            "geist %s — minimal CPU LLM inference\n\n"
            "Usage:\n"
            "  %s -m <model> \"prompt\"                 ask a question (instruct chat)\n"
            "  %s -m <model> --raw \"prompt\"           raw base-model text completion\n"
            "  %s -m <model> --serve <socket>         resident dynamic-tools daemon\n"
            "  %s --version\n\n"
            "Options:\n"
            "  -m, --model <path>   the model GGUF to load (required unless embedded)\n"
            "  --raw                raw base-model completion (default is instruct chat:\n"
            "                       the prompt is wrapped in the model's chat template)\n"
            "  --serve <socket>     serve dynamic-tools-v1 over a chmod-600 Unix socket\n"
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
            prog);
#endif
    return code;
}

int main(int argc, char **argv) {
    /* `--serve <socket>` anywhere -> the resident dynamic-tools daemon. The whole
     * argv is handed to geist_agent_main (it parses -m/--serve/-n itself). The
     * host supplies the toolset per request, so the served agent compiles none
     * (no_tools). In an embedded build the baked-in model is passed through. */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--serve") == 0) {
#if defined(GEIST_EMBEDDED_MODEL)
            return geist_agent_main(argc,
                                    argv,
                                    SERVE_SYSTEM,
                                    no_tools,
                                    nullptr,
                                    geist_embedded_model_start,
                                    geist_embedded_model_end);
#else
            return geist_agent_main(argc, argv, SERVE_SYSTEM, no_tools, nullptr, nullptr, nullptr);
#endif
        }
    }

    /* Default dispatch: `geist [-m <model>] "prompt" [-n N] [--raw]`.
     * Model source: -m <path>  >  embedded  >  error. */
    const char *prog       = "geist";
    const char *model_path = nullptr; /* from -m; embedded build ignores it */
    const char *prompt     = nullptr; /* the single positional prompt */
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
    /* A prompt is required (no interactive REPL in the engine CLI — use --serve
     * for a resident agent, or link libgeist for your own loop). */
    if (prompt == nullptr) {
        fprintf(stderr,
                "%s: no prompt — pass a \"prompt\", or --serve <socket> for a daemon\n",
                prog);
        return usage(prog, 2);
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
     * answer that stops at the turn marker — offline and tool-free. --raw opts
     * out to base-model completion (a bare continuation). */
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
         * the templated prompt). Reuses the shared greedy generator. */
        geist_token_t eos = geist_model_eos_token(model);
        geist_token_t eot =
                tmpl.stop[0] ? geist_model_token_by_text(model, tmpl.stop) : GEIST_TOKEN_NONE;
        static char answer[1 << 14];
        geist_generate_greedy(
                sess, eos, eot, tmpl.leak, budget, nullptr, nullptr, sizeof answer, answer);
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
