/*
 * src/archs/transformer/family_list.h — the one list of model families.
 *
 * Layer: ARCHITECTURE (internal).
 *
 * Every family the transformer arch can load is one line here, and nothing
 * else names the set (#345). Generated from it:
 *
 *   arch_family.h   the extern of each family table and of its name
 *   arch_family.c   REGISTRY[], geist_arch_transformer_gguf_names[], the
 *                   name strings, and the "no GEIST_FAMILY_* given, so all
 *                   of them" default
 *   mk/common.mk    ALL_FAMILIES and the -DGEIST_FAMILY_<UNIT>=1 flags of
 *                   make FAMILIES=...; make refuses a families/<unit>.c
 *                   without a line here, and a line without its file
 *
 * A family unit defines its table with TRANSFORMER_FAMILY(sym, ...)
 * (arch_family.h), which takes the name from here: a unit whose symbol has
 * no line does not compile.
 *
 * X(unit, UNIT, sym, gguf_name)
 *   unit       families/<unit>.c, and the name make FAMILIES=... takes
 *   UNIT       the unit's selection macro, GEIST_FAMILY_<UNIT>
 *   sym        the table, transformer_family_<sym>; a unit can have several
 *   gguf_name  the `general.architecture` value that selects the table
 *
 * Order is the registry's: transformer_family_select takes the first match.
 * Keep one entry per line, as written: mk/common.mk reads the first two
 * fields of each "X(" line.
 */
#ifndef GEIST_INTERNAL_ARCH_TRANSFORMER_FAMILY_LIST_H
#define GEIST_INTERNAL_ARCH_TRANSFORMER_FAMILY_LIST_H

/* clang-format off */
#define GEIST_FAMILY_LIST(X)                                  \
    X(gemma4,     GEMMA4,     gemma4,      "gemma4")          \
    X(gemma3,     GEMMA3,     gemma3,      "gemma3")          \
    X(llama,      LLAMA,      llama,       "llama")           \
    X(qwen3,      QWEN3,      qwen3,       "qwen3")           \
    X(qwen35,     QWEN35,     qwen35,      "qwen35")          \
    X(qwen35,     QWEN35,     qwen35moe,   "qwen35moe")       \
    X(bitnet,     BITNET,     bitnet_b158, "bitnet-b1.58")    \
    X(bitnet,     BITNET,     bitnet,      "bitnet")          \
    X(qwen3moe,   QWEN3MOE,   qwen3moe,    "qwen3moe")        \
    X(olmoe,      OLMOE,      olmoe,       "olmoe")           \
    X(granitemoe, GRANITEMOE, granitemoe,  "granitemoe")      \
    X(gpt_oss,    GPT_OSS,    gpt_oss,     "gpt-oss")         \
    X(deepseek2,  DEEPSEEK2,  deepseek2,   "deepseek2")       \
    X(glm4moe,    GLM4MOE,    glm4moe,     "glm4moe")         \
    X(llama4,     LLAMA4,     llama4,      "llama4")
/* clang-format on */

/* ---- Selection ----------------------------------------------------------
 *
 * GEIST_FAMILY_ON(UNIT) is 1 when the build selected the unit
 * (-DGEIST_FAMILY_<UNIT>=1, what make FAMILIES=... passes) and 0 otherwise;
 * a build that selects none (another build system) gets all of them. It
 * works inside a macro expansion, where #if cannot: the selection macro
 * either expands to 1, which pastes into GEIST_PP_ON_1 and shifts a 1 into
 * the second argument, or it stays a name and the 0 stays second (the
 * IS_ENABLED idiom). A selection macro set to anything but 1 counts as off;
 * mk/common.mk sets exactly 1. */
#define GEIST_PP_ON_1 0,
#define GEIST_PP_SECOND(a, b, ...) b
/* NOLINTNEXTLINE(bugprone-macro-parentheses): parentheses would hide the "0," */
#define GEIST_PP_IS_1_(junk_or_comma) GEIST_PP_SECOND(junk_or_comma 1, 0, 0)
#define GEIST_PP_IS_1(x) GEIST_PP_IS_1_(GEIST_PP_ON_##x)
#define GEIST_PP_IS_1_X(x) GEIST_PP_IS_1(x)
#define GEIST_PP_IF_0(...)
#define GEIST_PP_IF_1(...) __VA_ARGS__
#define GEIST_PP_IF_(c) GEIST_PP_IF_##c
#define GEIST_PP_IF(c) GEIST_PP_IF_(c)

/* NOLINTNEXTLINE(bugprone-macro-parentheses): a term of the sum in the #if below */
#define GEIST_FAMILY_SELECTED_(unit, UNIT, sym, gguf_name) +GEIST_PP_IS_1_X(GEIST_FAMILY_##UNIT)
#if (0 GEIST_FAMILY_LIST(GEIST_FAMILY_SELECTED_)) == 0
#define GEIST_FAMILY_ON(UNIT) 1
#else
#define GEIST_FAMILY_ON(UNIT) GEIST_PP_IS_1_X(GEIST_FAMILY_##UNIT)
#endif

#endif /* GEIST_INTERNAL_ARCH_TRANSFORMER_FAMILY_LIST_H */
