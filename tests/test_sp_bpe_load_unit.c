/*
 * test_sp_bpe_load_unit — sp_bpe_tokenizer_load trusts a tokenizer.bin's
 * lengths and counts only as far as the file backs them.
 *
 * Every length and count in the file is untrusted. A length is checked by
 * subtraction, not `p + len > end`, which forms a pointer past the mapping
 * (undefined; AGENT.md §4); and the tables must not be allocated for
 * whatever counts the header claims before the entries are read: a 6 KB
 * file claiming 2^20 tokens, merges or special tokens must not grow the
 * process by tens of MB before the truncation is noticed.
 *
 * The tokenizer_fixtures vocab as a tokenizer.bin must load; every proper
 * prefix of it must be refused; and a header whose counts the rest of the
 * file cannot hold must be refused without the process growing.
 */
#include "test_helpers.h"

#include "sp_bpe_tokenizer.h"
#include "tokenizer_fixtures.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

static const char SPM_MARKER[] = "\xe2\x96\x81"; /* ▁ */

/* Header layout (tf_sp_bpe_bin): magic, version, then these u32 counts. */
enum { OFF_VOCAB = 8, OFF_MERGES = 12, OFF_SPECIALS = 16 };

static char g_path[] = "/tmp/test_sp_bpe_load_XXXXXX";

/* Writes the first n bytes of b to g_path and loads it; *tok is set on
 * success and must be freed. */
static bool load_prefix(const struct tf_buf *b, size_t n, struct sp_bpe_tokenizer **tok) {
    FILE *f = fopen(g_path, "wb");
    if (f == nullptr) {
        fprintf(stderr, "cannot write %s\n", g_path);
        exit(GEIST_TEST_ERROR);
    }
    if (n > 0) {
        xfwrite(b->b, 1, n, f);
    }
    fclose(f);
    *tok = nullptr;
    return sp_bpe_tokenizer_load(tok, g_path);
}

static long peak_rss_kb(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
    return ru.ru_maxrss / 1024; /* bytes on macOS, KB on Linux */
#else
    return ru.ru_maxrss;
#endif
}

/* Loads all of b in a child process, whose peak RSS starts where ours is
 * and so shows this load's growth alone. Returns false if the child could
 * not report. */
static bool load_in_child(const struct tf_buf *b, bool *loaded, long *grew_kb) {
    int fds[2];
    if (pipe(fds) != 0) {
        return false;
    }
    fflush(stdout);
    fflush(stderr);
    const pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        return false;
    }
    if (pid == 0) {
        close(fds[0]);
        const long               before = peak_rss_kb();
        struct sp_bpe_tokenizer *tok    = nullptr;
        const long report[2] = {load_prefix(b, b->n, &tok) ? 1 : 0, peak_rss_kb() - before};
        sp_bpe_tokenizer_free(tok);
        const bool sent = write(fds[1], report, sizeof report) == (ssize_t) sizeof report;
        _exit(sent ? 0 : 1);
    }
    close(fds[1]);
    long       report[2] = {0, 0};
    const bool got       = read(fds[0], report, sizeof report) == (ssize_t) sizeof report;
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    *loaded  = report[0] != 0;
    *grew_kb = report[1];
    return got && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

int main(void) {
    const int fd = mkstemp(g_path);
    if (fd < 0) {
        fprintf(stderr, "mkstemp failed\n");
        return GEIST_TEST_ERROR;
    }
    close(fd);

    struct tf_vocab v     = tf_make_vocab(SPM_MARKER, true);
    struct tf_buf   bin   = tf_sp_bpe_bin(&v);
    int             fails = 0;

    /* A claimed count the file cannot hold, one section at a time. */
    static const struct {
        size_t      off;
        const char *what;
    } COUNT[] = {
            {OFF_VOCAB, "vocab"},
            {OFF_MERGES, "merge"},
            {OFF_SPECIALS, "special-token"},
    };
    for (size_t i = 0; i < sizeof COUNT / sizeof COUNT[0]; i++) {
        struct tf_buf bad = {0};
        tf_put(&bad, bin.b, bin.n);
        const uint32_t huge = 1u << 20;
        memcpy(bad.b + COUNT[i].off, &huge, 4);
        bool       loaded   = false;
        long       grew     = 0;
        const bool measured = load_in_child(&bad, &loaded, &grew);
        char       what[160];
        snprintf(what,
                 sizeof what,
                 "a %zu-byte file claiming 2^20 %s entries is refused without growing "
                 "(grew %ld KB)",
                 bad.n,
                 COUNT[i].what,
                 grew);
        fails += geist_expect(measured && !loaded && grew < 8 * 1024, what);
        free(bad.b);
    }

    /* The whole file loads, and says what it holds. */
    struct sp_bpe_tokenizer *tok = nullptr;
    const bool               ok  = load_prefix(&bin, bin.n, &tok);
    fails += geist_expect(ok && sp_bpe_tokenizer_vocab_size(tok) == v.n_tok, "the fixture loads");
    size_t len = 0;
    if (ok) {
        const char *last = sp_bpe_tokenizer_id_to_text(tok, (uint32_t) (v.n_tok - 1), &len);
        fails += geist_expect(last != nullptr && len == strlen(v.tok[v.n_tok - 1]) &&
                                      memcmp(last, v.tok[v.n_tok - 1], len) == 0,
                              "its last token reads back");
    }
    sp_bpe_tokenizer_free(tok);

    /* Every proper prefix ends inside the header or inside an entry the
     * header's counts promise: all must be refused. The loader says why on
     * stderr, a line per prefix, so that goes to /dev/null meanwhile. */
    size_t    loaded_prefixes = 0, first_loaded = 0;
    const int saved_stderr = dup(2);
    const int devnull      = open("/dev/null", O_WRONLY);
    fflush(stderr);
    dup2(devnull, 2);
    for (size_t n = 0; n < bin.n; n++) {
        struct sp_bpe_tokenizer *t = nullptr;
        if (load_prefix(&bin, n, &t)) {
            if (loaded_prefixes++ == 0) {
                first_loaded = n;
            }
            sp_bpe_tokenizer_free(t);
        }
    }
    fflush(stderr);
    dup2(saved_stderr, 2);
    close(saved_stderr);
    close(devnull);
    if (loaded_prefixes > 0) {
        fprintf(stderr, "  first loaded prefix: %zu of %zu bytes\n", first_loaded, bin.n);
    }
    char what[128];
    snprintf(what,
             sizeof what,
             "every proper prefix of the %zu-byte file is refused (%zu loaded)",
             bin.n,
             loaded_prefixes);
    fails += geist_expect(loaded_prefixes == 0, what);

    unlink(g_path);
    free(bin.b);
    tf_free_vocab(&v);
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("PASS: tokenizer.bin lengths and counts are checked against the file\n");
    return GEIST_TEST_PASS;
}
