/*
 * src/engine/pair_merge.h — the greedy pairwise merge at the heart of the
 * BPE and SentencePiece encoders, in O(n log n).
 *
 * Layer: ENGINE (internal). Header-only: gguf_tokenizer.c (gpt2, SPM and
 * unigram modes) and sp_bpe_tokenizer.c share it.
 *
 * An encoder splits a chunk into symbols, one per UTF-8 codepoint, and
 * then, while any adjacent pair can merge, merges the pair with the lowest
 * key (the merge rank for BPE; the score, highest first, for unigram), the
 * leftmost one among equal keys. Rescanning every pair after each merge is
 * O(n^2) in the chunk: 64 KB of text between special tokens took 30 s.
 *
 * Here every mergeable adjacent pair sits in a binary min-heap ordered by
 * (key, position of its left symbol). A merge pushes the two pairs it
 * creates with its neighbours; a popped pair whose symbols changed since
 * it was pushed (either one merged with something else) is dropped. The
 * heap then pops exactly the pair the rescan would find — lowest key,
 * leftmost — so the merges, and the tokens, are the same.
 */
#ifndef GEIST_INTERNAL_ENGINE_PAIR_MERGE_H
#define GEIST_INTERNAL_ENGINE_PAIR_MERGE_H

#include "heap.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* One symbol: a slice of the chunk's bytes in a doubly-linked list. */
struct pair_merge_sym {
    size_t off;  /* byte offset in the chunk */
    size_t len;  /* bytes; 0 once merged into its left neighbour */
    int    prev; /* -1 = head */
    int    next; /* -1 = tail */
};

/* The key of merging the adjacent symbols [off, off + llen) and
 * [off + llen, off + llen + rlen) of buf; false when they do not merge.
 * Lower keys merge first. */
typedef bool (*pair_merge_key_fn)(
        const void *ctx, const char *buf, size_t off, size_t llen, size_t rlen, uint64_t *key);

/* 24 bytes. Symbol indices grow with the offset in the buffer (a merge
 * keeps its left symbol), so the left index orders ties leftmost first. */
struct pair_merge_entry {
    uint64_t key;
    uint32_t llen, rlen; /* the lengths when pushed; changed = stale */
    int      left, right;
};

static inline bool pair_merge_before(const struct pair_merge_entry *a,
                                     const struct pair_merge_entry *b) {
    return a->key < b->key || (a->key == b->key && a->left < b->left);
}

static inline void
pair_merge_push(struct pair_merge_entry *heap, size_t *n, struct pair_merge_entry e) {
    size_t i = (*n)++;
    while (i > 0) {
        const size_t parent = (i - 1) / 2;
        if (!pair_merge_before(&e, &heap[parent])) {
            break;
        }
        heap[i] = heap[parent];
        i       = parent;
    }
    heap[i] = e;
}

static inline struct pair_merge_entry pair_merge_pop(struct pair_merge_entry *heap, size_t *n) {
    const struct pair_merge_entry top  = heap[0];
    const struct pair_merge_entry last = heap[--(*n)];
    size_t                        i    = 0;
    while (true) {
        size_t c = 2 * i + 1;
        if (c >= *n) {
            break;
        }
        if (c + 1 < *n && pair_merge_before(&heap[c + 1], &heap[c])) {
            c++;
        }
        if (!pair_merge_before(&heap[c], &last)) {
            break;
        }
        heap[i] = heap[c];
        i       = c;
    }
    if (*n > 0) {
        heap[i] = last;
    }
    return top;
}

/* Pushes the pair (left, right) if it merges. */
static inline void pair_merge_offer(struct pair_merge_entry     *heap,
                                    size_t                      *n,
                                    const struct pair_merge_sym *syms,
                                    int                          left,
                                    int                          right,
                                    const char                  *buf,
                                    pair_merge_key_fn            key_fn,
                                    const void                  *ctx) {
    uint64_t key = 0;
    if (key_fn(ctx, buf, syms[left].off, syms[left].len, syms[right].len, &key)) {
        pair_merge_push(heap,
                        n,
                        (struct pair_merge_entry) {.key   = key,
                                                   .llen  = (uint32_t) syms[left].len,
                                                   .rlen  = (uint32_t) syms[right].len,
                                                   .left  = left,
                                                   .right = right});
    }
}

/* Heap entries a chunk of up to this many symbols keeps on the stack
 * (2.3 KB) instead of allocating them. */
enum { PAIR_MERGE_STACK_SYMS = 32 };

/* Runs the merges over syms[0, n), a list in buffer order (head 0, tail
 * n - 1, n and every length <= INT_MAX), in place: afterwards the list from syms[0] holds
 * the merged symbols. The heap holds at most 3n entries: n - 1 initial
 * pairs and two per merge. Returns false when that heap cannot be
 * allocated. */
[[nodiscard]] static inline bool pair_merge_run(size_t                 n,
                                                struct pair_merge_sym *syms,
                                                const char            *buf,
                                                pair_merge_key_fn      key_fn,
                                                const void            *ctx) {
    if (n < 2) {
        return true;
    }
    struct pair_merge_entry  stack_heap[3 * PAIR_MERGE_STACK_SYMS];
    struct pair_merge_entry *heap = stack_heap;
    if (n > PAIR_MERGE_STACK_SYMS) {
        heap = heap_alloc_array_aligned(struct pair_merge_entry, 3 * n);
        if (heap == nullptr) {
            return false;
        }
    }
    size_t n_heap = 0;
    for (size_t i = 0; i + 1 < n; i++) {
        pair_merge_offer(heap, &n_heap, syms, (int) i, (int) i + 1, buf, key_fn, ctx);
    }
    while (n_heap > 0) {
        const struct pair_merge_entry e = pair_merge_pop(heap, &n_heap);
        struct pair_merge_sym        *l = &syms[e.left];
        struct pair_merge_sym        *r = &syms[e.right];
        if (l->len != e.llen || l->next != e.right || r->len != e.rlen) {
            continue; /* stale: one of the two merged with something else */
        }
        l->len += r->len;
        l->next = r->next;
        if (r->next >= 0) {
            syms[r->next].prev = e.left;
        }
        r->len = 0;
        if (l->prev >= 0) {
            pair_merge_offer(heap, &n_heap, syms, l->prev, e.left, buf, key_fn, ctx);
        }
        if (l->next >= 0) {
            pair_merge_offer(heap, &n_heap, syms, e.left, l->next, buf, key_fn, ctx);
        }
    }
    if (heap != stack_heap) {
        safe_free((void **) &heap);
    }
    return true;
}

#endif /* GEIST_INTERNAL_ENGINE_PAIR_MERGE_H */
