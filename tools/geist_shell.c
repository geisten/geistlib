/*
 * geist_shell — a ~15-line agent CLI that exposes ONE host capability to the
 * model: list a directory (the list_dir tool). Shows how function tooling is
 * added (define a geist_tool, forward argc/argv to geist_agent_main) and how the
 * security model holds: the model can ask to list a folder, it cannot run an
 * arbitrary command — list_dir is opendir/readdir, no shell.
 *
 *   geist_shell <model.gguf> "Zeige mir den Inhalt des aktuellen Ordners"
 *   geist_shell <model.gguf>                 # interactive REPL
 */
#define _POSIX_C_SOURCE 200809L

#include "agent_listdir.h"
#include "agent_main.h"

int main(int argc, char **argv) {
    struct geist_tool tools[] = {listdir_tool()};
    return geist_agent_main(
            argc,
            argv,
            "You are a file assistant. To see a directory's contents, reply with the "
            "list_dir tool call: {\"tool\":\"list_dir\",\"args\":{\"path\":\".\"}} (omit "
            "the path or use \".\" for the current directory). After the tool result "
            "comes back, tell the user in one sentence what is in the folder.",
            sizeof tools / sizeof *tools,
            tools);
}
