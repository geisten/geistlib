/*
 * geist_agent.c — a ready-to-use agent CLI: answers from a local document
 * folder via the whitelist-gated tool loop. This is also the reference shape a
 * separate app repo copies — define your tools + system prompt, forward argv:
 *
 *   GEIST_DOCS=./kb geist_agent <model.gguf> "How long is the warranty?"
 *   GEIST_DOCS=./kb geist_agent <model.gguf>           # interactive REPL
 *
 * The doc directory is an env knob (GEIST_DOCS, default ./docs) so the generic
 * argv (model / question / -n) stays owned by geist_agent_main.
 */
#define _POSIX_C_SOURCE 200809L /* docsearch_tool -> opendir/readdir */

#include <geist.h>

#include "agent.h"
#include "agent_docsearch.h"
#include "agent_main.h"

#include <stdlib.h>

int main(int argc, char **argv) {
    const char *docs = getenv("GEIST_DOCS");
    if (!docs || !docs[0]) {
        docs = "./docs";
    }
    struct geist_tool tools[] = {docsearch_tool(docs)};
    return geist_agent_main(argc,
                            argv,
                            "Answer the user's question using the local documents. Search them "
                            "with the doc_search tool; if they do not contain the answer, say so. "
                            "Cite the file name you used.",
                            sizeof tools / sizeof *tools,
                            tools);
}
