/*
 * geist_shell — a "file + web assistant" agent CLI: lists a directory
 * (list_dir), reads+summarizes a text file (summarize_file), and searches/reads
 * the web (web_search, web_fetch). Shows how function tooling is wired and that
 * the security model holds — the local tools are opendir/read only, no shell.
 *
 *   geist_shell <model.gguf> "Fasse die Datei README.md zusammen"
 *   geist_shell <model.gguf>                 # interactive REPL
 *
 * GEIST_FORCE_CALL=1 grammar-forces turn 0 into a tool call; GEIST_AGENT_TRACE=1
 * prints per-step progress on stderr. Both are handled by geist_agent_main.
 *
 * summarize_file's ctx needs the loaded model + backend (its sub-session runs on
 * them), so the table is built in shell_tools() — invoked after the model loads.
 */
#define _POSIX_C_SOURCE 200809L

#include "agent_listdir.h"
#include "agent_main.h" /* the reusable agent CLI engine */
#include "agent_summarize.h"
#include "agent_webfetch.h"
#include "agent_websearch.h"

#include <stdlib.h>

static const char *SHELL_SYSTEM =
        "You are a file and web assistant. To see a directory's contents reply with "
        "{\"tool\":\"list_dir\",\"args\":{\"path\":\".\"}}. To summarize a file reply with "
        "{\"tool\":\"summarize_file\",\"args\":{\"path\":\"<file>\"}}. To search the web reply with "
        "{\"tool\":\"web_search\",\"args\":{\"query\":\"<query>\"}}. To read a web page reply with "
        "{\"tool\":\"web_fetch\",\"args\":{\"url\":\"<url>\"}}. After the tool result, "
        "answer the user in one or two sentences.";

static size_t shell_tools(struct geist_model *model, struct geist_backend *be,
                          struct geist_tool *out, size_t cap, void *ctx) {
    (void) cap;
    (void) ctx;
    /* summarize_file's sub-session runs on this model+backend; ctx must outlive
     * the agent (static). root="." -> reads confined to the current tree. */
    static struct summarize_ctx sctx;
    sctx = (struct summarize_ctx) {.model = model, .be = be, .root = "."};
    /* web_search hits a fixed engine (low risk); web_fetch lets the model pick the
     * host — nullptr allowlist = any http/https, fine for a local demo, tighten via
     * webfetch_tool("example.com,...") for an exposed deployment. */
    out[0] = listdir_tool();
    out[1] = summarize_file_tool(&sctx);
    out[2] = websearch_tool(nullptr);
    out[3] = webfetch_tool(nullptr);
    return 4;
}

int main(int argc, char **argv) {
    return geist_agent_main(argc, argv, SHELL_SYSTEM, shell_tools, nullptr);
}
