/*
 * mind.h — the file-based "memory palace": markdown notes + a markdown index.
 *
 * Header-only (static inline) so the CLI and the tests share one copy without
 * build wiring. One note per file under $GEIST_MIND_DIR (default ./mind), one
 * index line per note in INDEX.md. No DB, no embeddings — grep + the index.
 *
 * All paths are caller-bounded; functions return -1 / 0 on failure so the
 * palace never silently loses a note (no assert()).
 */
#ifndef GEIST_MIND_H
#define GEIST_MIND_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

enum { MIND_PATH_CAP = 1024 };

static inline const char *mind_dir(void) {
    const char *d = getenv("GEIST_MIND_DIR");
    return (d && d[0]) ? d : "mind";
}

/* "My Web Note!" -> "my-web-note". out cap should be >= strlen(title)+1.
 * Empty / all-punctuation titles collapse to "note" so a slug is always valid. */
static inline void mind_slugify(const char *title, char *out, size_t cap) {
    size_t w         = 0;
    int    prev_dash = 1; /* trim leading dashes */
    for (const char *p = title; *p && w + 1 < cap; p++) {
        char c = *p;
        if (c >= 'A' && c <= 'Z') {
            c = (char) (c - 'A' + 'a');
        }
        int alnum = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
        if (alnum) {
            out[w++]  = c;
            prev_dash = 0;
        } else if (!prev_dash) {
            out[w++]  = '-';
            prev_dash = 1;
        }
    }
    while (w > 0 && out[w - 1] == '-') {
        w--; /* trim trailing dash */
    }
    out[w] = '\0';
    if (w == 0) {
        snprintf(out, cap, "note");
    }
}

static inline void mind_today(char *out, size_t cap) {
    time_t    t = time(nullptr);
    struct tm tm;
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    strftime(out, cap, "%Y-%m-%d", &tm);
}

/* Read a whole small file into buf (truncates at cap-1, NUL-terminated).
 * Returns bytes read, or -1 if the file can't be opened. */
static inline long mind_slurp(const char *path, char *buf, size_t cap) {
    if (cap) {
        buf[0] = '\0'; /* always leave buf NUL-terminated, even on failure */
    }
    FILE *f = fopen(path, "r");
    if (!f) {
        return -1;
    }
    size_t n = fread(buf, 1, cap - 1, f);
    buf[n]   = '\0';
    fclose(f);
    return (long) n;
}

/* Write $MIND_DIR/<slug>.md (frontmatter + body) and append an INDEX.md line.
 * Returns 0 on success, -1 on any I/O failure (the palace must not lose data). */
static inline int mind_remember(const char *title, const char *text) {
    const char *dir = mind_dir();
    mkdir(dir, 0755); /* ignore EEXIST; the open below is the real check */

    char slug[256];
    mind_slugify(title, slug, sizeof slug);
    char date[16];
    mind_today(date, sizeof date);

    char path[MIND_PATH_CAP];
    snprintf(path, sizeof path, "%s/%s.md", dir, slug);
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "remember: cannot write %s\n", path);
        return -1;
    }
    fprintf(f, "---\ntitle: %s\ndate: %s\n---\n\n%s\n", title, date, text);
    if (fclose(f) != 0) {
        fprintf(stderr, "remember: write failed %s\n", path);
        return -1;
    }

    /* index hook = first line / ~70 chars of the body */
    char   hook[80];
    size_t h = 0;
    for (const char *p = text; *p && *p != '\n' && h + 1 < sizeof hook; p++) {
        hook[h++] = *p;
    }
    hook[h] = '\0';

    char ipath[MIND_PATH_CAP];
    snprintf(ipath, sizeof ipath, "%s/INDEX.md", dir);
    FILE *ix = fopen(ipath, "a");
    if (!ix) {
        fprintf(stderr, "remember: cannot append %s\n", ipath);
        return -1;
    }
    fprintf(ix, "- [%s](%s.md) — %s · %s\n", title, slug, hook, date);
    if (fclose(ix) != 0) {
        fprintf(stderr, "remember: index write failed\n");
        return -1;
    }
    return 0;
}

/* Load note <slug> into buf. Returns bytes read, or -1 if absent. */
static inline long mind_recall(const char *slug, char *buf, size_t cap) {
    char path[MIND_PATH_CAP];
    snprintf(path, sizeof path, "%s/%s.md", mind_dir(), slug);
    return mind_slurp(path, buf, cap);
}

#endif /* GEIST_MIND_H */
