#include "qwen-m1-rounding-io.h"
#include "qwen-m1-deltanet.h"
#include <float.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { block_size = Q1_BLOCK_SIZE };

struct table_header {
    uint32_t magic, version, first, last, blocks, words;
};
struct block {
    uint32_t first, last, offset, width;
};

static_assert(sizeof(float) == sizeof(uint32_t) && FLT_RADIX == 2 && FLT_MANT_DIG == 24);
static_assert(sizeof(struct table_header) == 6 * sizeof(uint32_t));
static_assert(sizeof(struct block) == 4 * sizeof(uint32_t));

/* Interpolation and residuals use integer arithmetic on IEEE-754 encodings. */
static unsigned bit_width(uint32_t value) {
    unsigned n = 0;
    for (; value; value >>= 1)
        n++;
    return n;
}

static bool hex(const char *s, uint32_t *out) {
    char               *end;
    const unsigned long n = strtoul(s, &end, 16);
    if (end == s || *end || n > UINT32_MAX)
        return false;
    *out = (uint32_t) n;
    return true;
}

static bool encode_block(size_t               n,
                         const uint32_t       values[static n],
                         struct block        *block,
                         struct table_header *header,
                         FILE                *file) {
    *block = (struct block) {.first = values[0], .last = values[n - 1], .offset = header->words};
    uint32_t codes[block_size];
    for (size_t i = 0; i < n; i++) {
        const int64_t  delta = (int64_t) values[i] -
                               q1_predict((uint32_t) n, (uint32_t) i, block->first, block->last);
        const uint64_t code  = delta < 0 ? (uint64_t) (-2 * delta - 1) : (uint64_t) (2 * delta);
        if (code > UINT32_MAX)
            return false;
        codes[i]             = (uint32_t) code;
        const unsigned width = bit_width(codes[i]);
        if (width > block->width)
            block->width = width;
    }
    if (block->width == 0)
        return true;
    const size_t count = (n * block->width + 31) / 32;
    if (count > UINT32_MAX - header->words)
        return false;
    uint32_t packed[block_size] = {0};
    for (size_t i = 0; i < n; i++) {
        const size_t   bit = i * block->width, word = bit / 32;
        const unsigned shift = (unsigned) (bit % 32);
        packed[word] |= codes[i] << shift;
        if (shift + block->width > 32)
            packed[word + 1] |= codes[i] >> (32 - shift);
    }
    if (fwrite(packed, sizeof(*packed), count, file) != count)
        return false;
    header->words += (uint32_t) count;
    return true;
}

static bool encode(uint32_t first, uint32_t last, FILE *input, FILE *output) {
    const uint64_t      n      = (uint64_t) last - first + 1;
    const size_t        count  = (size_t) ((n + block_size - 1) / block_size);
    struct table_header h      = {Q1_MAGIC, Q1_VERSION, first, last, (uint32_t) count, 0};
    struct block       *blocks = calloc(count, sizeof(*blocks));
    if (!blocks)
        return false;
    bool ok = fseek(output, (long) (sizeof(h) + count * sizeof(*blocks)), SEEK_SET) == 0;
    for (size_t i = 0; ok && i < count; i++) {
        const uint64_t remaining = n - (uint64_t) i * block_size;
        const size_t   used      = remaining < block_size ? (size_t) remaining : block_size;
        uint32_t       values[block_size];
        ok = fread(values, sizeof(*values), used, input) == used &&
             encode_block(used, values, &blocks[i], &h, output);
    }
    ok = ok && fgetc(input) == EOF && !ferror(input) && fseek(output, 0, SEEK_SET) == 0 &&
         fwrite(&h, sizeof(h), 1, output) == 1 &&
         fwrite(blocks, sizeof(*blocks), count, output) == count;
    fprintf(stderr, "%" PRIu32 " blocks, %" PRIu32 " payload words\n", h.blocks, h.words);
    free(blocks);
    return ok;
}

static uint32_t *read_table(size_t *count, FILE *input) {
    if (fseek(input, 0, SEEK_END) != 0)
        return nullptr;
    const long length = ftell(input);
    if (length <= 0 || (size_t) length % sizeof(uint32_t) || fseek(input, 0, SEEK_SET))
        return nullptr;
    *count          = (size_t) length / sizeof(uint32_t);
    uint32_t *table = malloc((size_t) length);
    if (table && fread(table, sizeof(*table), *count, input) == *count &&
        q1_table_valid(*count, table))
        return table;
    free(table);
    return nullptr;
}

static uint32_t bits(float value) {
    uint32_t result;
    memcpy(&result, &value, sizeof(result));
    return result;
}

static float value(uint32_t raw) {
    float result;
    memcpy(&result, &raw, sizeof(result));
    return result;
}

static bool check(FILE *table, FILE *keys, FILE *reference) {
    size_t    count;
    uint32_t *data = read_table(&count, table);
    if (!data)
        return false;
    uint32_t key, expected;
    size_t   n = 0, matches = 0;
    size_t   got;
    bool     ok = true;
    while ((got = fread(&key, 1, sizeof(key), keys)) == sizeof(key)) {
        if (fread(&expected, sizeof(expected), 1, reference) != 1) {
            ok = false;
            break;
        }
        if (q1_lookup(key, data) == expected)
            matches++;
        n++;
    }
    printf("%zu/%zu exact\n", matches, n);
    free(data);
    return ok && got == 0 && n > 0 && matches == n && fgetc(reference) == EOF && !ferror(keys) &&
           !ferror(reference);
}

static bool check_silu(FILE *exp_table, FILE *rcp_table, FILE *inputs, FILE *reference) {
    size_t    exp_count, rcp_count;
    uint32_t *exp = read_table(&exp_count, exp_table);
    uint32_t *rcp = read_table(&rcp_count, rcp_table);
    bool      ok  = exp && rcp && exp[2] == Q1_EXP_FIRST && exp[3] == Q1_EXP_LAST &&
                    rcp[2] == Q1_RCP_FIRST && rcp[3] == Q1_RCP_LAST;
    uint32_t  key, expected;
    size_t    n = 0, matches = 0, got = 0;
    while (ok && (got = fread(&key, 1, sizeof(key), inputs)) == sizeof(key)) {
        ok = fread(&expected, sizeof(expected), 1, reference) == 1;
        if (!ok)
            break;
        const float x           = value(key);
        const float e           = value(q1_exp_bits(key & 0x7fffffffu, exp));
        const float denominator = 1.0f + e;
        const float reciprocal  = value(q1_lookup(bits(denominator), rcp));
        const float numerator   = x >= 0.0f ? x : x * e;
        if (bits(numerator * reciprocal) == expected)
            matches++;
        n++;
    }
    printf("%zu/%zu SiLU results exact\n", matches, n);
    free(rcp);
    free(exp);
    return ok && got == 0 && n > 0 && matches == n && fgetc(reference) == EOF && !ferror(inputs) &&
           !ferror(reference);
}

static bool verify(FILE *source, FILE *table) {
    size_t    count;
    uint32_t *data = read_table(&count, table);
    if (!data)
        return false;
    const uint32_t first = data[2], last = data[3];
    bool           ok = true;
    for (uint32_t key = first; key <= last; key++) {
        uint32_t expected;
        if (fread(&expected, sizeof(expected), 1, source) != 1 ||
            expected != q1_lookup(key, data)) {
            fprintf(stderr, "Round trip failed at %08" PRIx32 "\n", key);
            ok = false;
            break;
        }
    }
    free(data);
    if (!ok || fgetc(source) != EOF || ferror(source))
        return false;
    printf("%" PRIu32 " consecutive FP32 values exact\n", last - first + 1);
    return true;
}

static bool check_deltanet(const char *const paths[static 4], const char *fixture) {
    const uint32_t starts[] = {Q1_EXP_FIRST, Q1_RCP_FIRST, Q1_EXP_FIRST, Q1_RSQRT_FIRST};
    const uint32_t ends[]   = {Q1_EXP_LAST, Q1_RCP_LAST, Q1_EXP_LAST, Q1_RSQRT_LAST};
    enum { exp_idx, rcp_idx, beta_idx, rsqrt_idx, table_count };
    uint32_t *table[table_count] = {0};
    bool      ok                 = true;
    for (size_t i = 0; ok && i < table_count; i++) {
        FILE  *file = fopen(paths[i], "rb");
        size_t count;
        if (file == nullptr) {
            ok = false;
        } else {
            table[i] = read_table(&count, file);
            ok       = table[i] && table[i][2] == starts[i] && table[i][3] == ends[i];
            fclose(file);
        }
    }
    FILE  *ref          = fopen(fixture, "r");
    size_t beta_matches = 0, rsqrt_matches = 0, rows = 0;
    if (ok && ref != nullptr) {
        uint32_t beta_input, beta_gold, rsqrt_input, rsqrt_gold;
        int      fields;
        while ((fields = fscanf(ref,
                                "%" SCNx32 " %" SCNx32 " %" SCNx32 " %" SCNx32,
                                &beta_input,
                                &beta_gold,
                                &rsqrt_input,
                                &rsqrt_gold)) == 4) {
            const uint32_t magnitude = beta_input & 0x7fffffffu;
            uint32_t       beta;
            if (beta_input >> 31) {
                beta = q1_beta_neg_bits(magnitude, table[beta_idx]);
            } else {
                const float e = value(q1_exp_bits(magnitude, table[exp_idx]));
                beta          = q1_lookup(bits(1.0f + e), table[rcp_idx]);
            }
            beta_matches += beta == beta_gold;
            rsqrt_matches += q1_rsqrt_bits(rsqrt_input, table[rsqrt_idx]) == rsqrt_gold;
            rows++;
        }
        ok = fields == EOF && !ferror(ref) && rows == 16 && beta_matches == rows &&
             rsqrt_matches == rows;
        printf("Archived beta %zu/%zu exact, K factors %zu/%zu exact\n",
               beta_matches,
               rows,
               rsqrt_matches,
               rows);
    } else {
        ok = false;
    }
    if (ref != nullptr)
        fclose(ref);
    for (size_t i = 0; i < table_count; i++)
        free(table[i]);
    return ok;
}

int main(int argc, char **argv) {
    bool ok = false;
    if (argc == 6 && !strcmp(argv[1], "encode")) {
        uint32_t low, high;
        if (!hex(argv[2], &low) || !hex(argv[3], &high) || low > high || high > Q1_INFINITY)
            return 2;
        FILE *source      = fopen(argv[4], "rb");
        FILE *destination = fopen(argv[5], "wb");
        ok                = source && destination && encode(low, high, source, destination);
        if (source && fclose(source) != 0)
            ok = false;
        if (destination && fclose(destination) != 0)
            ok = false;
    } else if (argc == 4 && !strcmp(argv[1], "verify")) {
        FILE *source = fopen(argv[2], "rb");
        FILE *table  = fopen(argv[3], "rb");
        ok           = source && table && verify(source, table);
        if (source)
            fclose(source);
        if (table)
            fclose(table);
    } else if (argc == 5 && !strcmp(argv[1], "check")) {
        FILE *table     = fopen(argv[2], "rb");
        FILE *keys      = fopen(argv[3], "rb");
        FILE *reference = fopen(argv[4], "rb");
        ok              = table && keys && reference && check(table, keys, reference);
        if (table)
            fclose(table);
        if (keys)
            fclose(keys);
        if (reference)
            fclose(reference);
    } else if (argc == 6 && !strcmp(argv[1], "check-silu")) {
        FILE *exp       = fopen(argv[2], "rb");
        FILE *rcp       = fopen(argv[3], "rb");
        FILE *inputs    = fopen(argv[4], "rb");
        FILE *reference = fopen(argv[5], "rb");
        ok = exp && rcp && inputs && reference && check_silu(exp, rcp, inputs, reference);
        if (exp)
            fclose(exp);
        if (rcp)
            fclose(rcp);
        if (inputs)
            fclose(inputs);
        if (reference)
            fclose(reference);
    } else if (argc == 7 && !strcmp(argv[1], "check-deltanet")) {
        const char *paths[4] = {argv[2], argv[3], argv[4], argv[5]};
        ok                   = check_deltanet(paths, argv[6]);
    } else {
        fputs("Usage: qwen-m1-rounding encode first-hex last-hex raw-u32 table\n"
              "       qwen-m1-rounding verify raw-u32 table\n"
              "       qwen-m1-rounding check table input-u32 expected-u32\n"
              "       qwen-m1-rounding check-silu exp-table rcp-table input-f32 "
              "expected-f32\n"
              "       qwen-m1-rounding check-deltanet exp rcp beta-neg rsqrt reference.txt\n",
              stderr);
        return 2;
    }
    return ok ? 0 : 1;
}
