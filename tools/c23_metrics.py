#!/usr/bin/env python3
"""c23_metrics.py — count the C23 refactoring's targets in the sources (#341).

The baseline of #333 came from grep -E, and BSD grep has no \\b: it reported
0 `nullptr` for a tree with 4171. This tool lexes the C sources instead, so
comments, string and character literals never count and a token is a token
on every host. It needs no compiler and no build, and gives the same output
on macOS and Linux.

Scope: the tracked .c, .h and .m files under src/ and include/, except the
generated SPIR-V and Metal shader headers (*_spv.h, metal_shaders.h).

    write   write benchmark/c23_metrics.json
    check   compare the tree with benchmark/c23_metrics.json; exit 1 when a
            gap grew past its committed ceiling (runs in CI)
    show    print every metric, the gaps and what is already done, as a
            table; --json prints it as JSON instead

benchmark/c23_metrics.json is a ratchet: one ceiling per gap, the total
#333's tickets reduce. `check` fails only when a total rises above its
ceiling, so a pull request that adds such a construct raises the ceiling
(`make c23-metrics`) and review sees it. A pull request that removes some
passes as is; it may lower the ceiling the same way, and `check` says so.
Per-file counts are in `show`, not in the committed file: concurrent pull
requests touching different files used to conflict over every line of it.
The constructs that are already done (nullptr, [[nodiscard]], ...) and the
sizes grow with any new code; they are in `show` only.

Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "benchmark" / "c23_metrics.json"
GENERATED = ("_spv.h", "metal_shaders.h")

# The gaps of #333, in its order; the committed file holds these.
GAPS = {
    "switch_with_default": "switch statements with a default: label of their own",
    "enum_named_unfixed": "named enum definitions without a fixed underlying type",
    "enum_anonymous": "enum { ... } used as constants (no tag, no typedef)",
    "define_object": "object-like #define with a replacement",
    "define_function": "function-like #define",
    "void_ptr": "void * (one per `void` followed by `*`)",
    "void_ptr_cast": "(void *) and (const void *) casts",
    "void_pp_cast": "(void **) casts, as in safe_free((void **) &p)",
    "_Static_assert": "_Static_assert",
    "__attribute__": "__attribute__",
    "__builtin": "__builtin_* calls and names",
    "pp_conditional": "#if, #ifdef, #ifndef",
    "pp_elif": "#elif, #elifdef, #elifndef",
    "goto": "goto",
}
# Already done; reported by `show`, not committed.
DONE = {
    "files": "files in scope",
    "lines": "lines in scope",
    "nullptr": "nullptr",
    "NULL": "NULL",
    "nodiscard": "[[nodiscard]]",
    "static_array_param": "[static n] array parameters",
    "ckd": "ckd_add, ckd_sub, ckd_mul",
    "constexpr": "constexpr",
    "static_assert": "static_assert",
    "std_attribute": "[[gnu::...]] and [[clang::...]]",
    "switch": "switch statements",
    "enum_named": "named enum definitions",
    "stdbit_include": "#include <stdbit.h>",
}

# keywords after which `(void *)` is a cast; any other name before the
# parenthesis makes it a call or a declarator
CAST_AFTER = ("return", "case", "else", "do")
PUNCT3 = ("...", "<<=", ">>=")
PUNCT2 = ("::", "->", "++", "--", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||",
          "*=", "/=", "%=", "+=", "-=", "&=", "^=", "|=", "##")


class Tok:
    __slots__ = ("kind", "text", "line", "space_before", "bol")

    def __init__(self, kind, text, line, space_before, bol):
        self.kind = kind  # "id", "num", "str", "punct"
        self.text = text
        self.line = line
        self.space_before = space_before
        self.bol = bol  # first token of a logical line


def lex(src: str) -> list[Tok]:
    """C tokens without comments; literals collapse to one "str" token."""
    toks: list[Tok] = []
    i, n, line = 0, len(src), 1
    space, bol = False, True
    while i < n:
        c = src[i]
        if c == "\\" and src.startswith("\n", i + 1):  # line splice
            i += 2
            line += 1
            continue
        if c == "\n":
            i += 1
            line += 1
            space, bol = True, True
            continue
        if c in " \t\r\f\v":
            i += 1
            space = True
            continue
        if src.startswith("//", i):
            while i < n and not (src[i] == "\n" and src[i - 1] != "\\"):
                if src[i] == "\n":
                    line += 1
                i += 1
            space = True
            continue
        if src.startswith("/*", i):
            end = src.find("*/", i + 2)
            end = n if end < 0 else end + 2
            line += src.count("\n", i, end)
            i = end
            space = True
            continue
        start = i
        if c.isalpha() or c == "_":
            while i < n and (src[i].isalnum() or src[i] == "_"):
                i += 1
            word = src[start:i]
            if i < n and src[i] in "'\"" and word in ("u8", "u", "U", "L"):
                i = _literal(src, i)
                toks.append(Tok("str", src[start:i], line, space, bol))
            else:
                toks.append(Tok("id", word, line, space, bol))
        elif c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            i += 1
            while i < n:
                if src[i] in "eEpP" and i + 1 < n and src[i + 1] in "+-":
                    i += 2
                elif src[i].isalnum() or src[i] in "._'":
                    i += 1
                else:
                    break
            toks.append(Tok("num", src[start:i], line, space, bol))
        elif c in "'\"":
            i = _literal(src, i)
            toks.append(Tok("str", src[start:i], line, space, bol))
        else:
            for p in PUNCT3 + PUNCT2:
                if src.startswith(p, i):
                    i += len(p)
                    break
            else:
                i += 1
            toks.append(Tok("punct", src[start:i], line, space, bol))
        line += src.count("\n", start, i)
        space, bol = False, False
    return toks


def _literal(src: str, i: int) -> int:
    """End of the string or character literal opening at src[i]."""
    quote = src[i]
    i += 1
    while i < len(src) and src[i] != quote:
        if src[i] == "\\":
            i += 1
        elif src[i] == "\n":  # unterminated: stop at the line end
            return i
        i += 1
    return i + 1


def count(toks: list[Tok]) -> dict[str, int]:
    m = dict.fromkeys(list(GAPS) + list(DONE), 0)
    texts = [t.text for t in toks]
    n = len(toks)

    def at(k):
        return texts[k] if 0 <= k < n else ""

    # switch frames: [brace depth of the body, paren depth, has default]
    frames: list[list] = []
    braces = parens = 0
    pending_switch = None  # paren depth at which a switch condition closes
    i = 0
    while i < n:
        t = toks[i]
        s = t.text
        if t.bol and s == "#":
            j = i + 1
            d = at(j)
            if d in ("if", "ifdef", "ifndef"):
                m["pp_conditional"] += 1
            elif d in ("elif", "elifdef", "elifndef"):
                m["pp_elif"] += 1
            elif d == "include" and at(j + 1) == "<" and at(j + 2) == "stdbit":
                m["stdbit_include"] += 1
            elif d == "define" and j + 1 < n and toks[j + 1].kind == "id":
                name = toks[j + 1]
                nxt = toks[j + 2] if j + 2 < n else None
                if nxt is not None and not nxt.bol and nxt.text == "(" \
                        and not nxt.space_before:
                    m["define_function"] += 1
                elif nxt is not None and not nxt.bol:
                    m["define_object"] += 1
            # the directive's own tokens still count below (void * in a macro)
        if t.kind == "id":
            if s == "switch":
                m["switch"] += 1
                pending_switch = parens
            elif s == "default" and at(i + 1) == ":" and frames \
                    and frames[-1][0] == braces and frames[-1][1] == parens:
                frames[-1][2] = True
            elif s == "enum":
                k = i + 1
                if at(k) == "{" or at(k) == ":":
                    if at(i - 1) != "typedef":
                        m["enum_anonymous"] += 1
                elif k < n and toks[k].kind == "id" and at(k + 1) in ("{", ":"):
                    m["enum_named"] += 1
                    if at(k + 1) == "{":
                        m["enum_named_unfixed"] += 1
            elif s == "void" and at(i + 1) == "*":
                m["void_ptr"] += 1
                k = i - 1
                if at(k) == "const":
                    k -= 1
                before = toks[k - 1] if k > 0 else None
                if at(k) == "(" and (before is None or before.text in CAST_AFTER or (
                        before.kind != "id" and before.text not in (")", "]"))):
                    e = i + 1
                    stars = 0
                    while at(e) in ("*", "const"):
                        stars += at(e) == "*"
                        e += 1
                    if at(e) == ")":
                        m["void_ptr_cast" if stars == 1 else "void_pp_cast"] += 1
            elif s in ("_Static_assert", "__attribute__", "goto", "nullptr", "NULL",
                       "constexpr", "static_assert"):
                m[s] += 1
            elif s.startswith("__builtin_"):
                m["__builtin"] += 1
            elif s in ("ckd_add", "ckd_sub", "ckd_mul"):
                m["ckd"] += 1
        elif s == "(":
            parens += 1
        elif s == ")":
            parens -= 1
        elif s == ";" and pending_switch is not None and parens == pending_switch:
            pending_switch = None  # a switch without braces
        elif s == "{":
            braces += 1
            if pending_switch is not None and parens == pending_switch:
                frames.append([braces, parens, False])
                pending_switch = None
        elif s == "}":
            if frames and frames[-1][0] == braces:
                if frames.pop()[2]:
                    m["switch_with_default"] += 1
            braces -= 1
        elif s == "[" and at(i + 1) == "[":
            if at(i + 2) == "nodiscard":
                m["nodiscard"] += 1
            elif at(i + 2) in ("gnu", "clang") and at(i + 3) == "::":
                m["std_attribute"] += 1
        elif s == "[" and at(i + 1) == "static":
            m["static_array_param"] += 1
        i += 1
    return m


def sources() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "src", "include"],
        cwd=ROOT, capture_output=True, check=True,
    ).stdout.decode()
    return sorted(
        f for f in out.split("\0")
        if f.endswith((".c", ".h", ".m")) and not f.endswith(GENERATED)
    )


def measure() -> tuple[dict, dict]:
    """(totals, {metric: {file: count}}) over every source in scope."""
    totals = dict.fromkeys(list(GAPS) + list(DONE), 0)
    per_file: dict[str, dict[str, int]] = {k: {} for k in totals}
    for f in sources():
        text = (ROOT / f).read_text(encoding="utf-8", errors="replace")
        m = count(lex(text))
        m["files"] = 1
        m["lines"] = text.count("\n")
        for k, v in m.items():
            totals[k] += v
            if v:
                per_file[k][f] = v
    return totals, per_file


def committed(totals: dict, per_file: dict) -> dict:
    return {
        "generated_by": "tools/c23_metrics.py write",
        "scope": "src/ and include/: tracked .c .h .m, without *_spv.h and metal_shaders.h",
        "ceiling": {k: {"what": GAPS[k], "max": totals[k]} for k in GAPS},
    }


def ratchet(ceiling: dict, totals: dict) -> tuple[list, list]:
    """(gaps above their ceiling or without one, gaps below it)."""
    grew = [k for k in GAPS if totals[k] > ceiling.get(k, {}).get("max", -1)]
    fell = [k for k in GAPS if k not in grew and totals[k] < ceiling[k]["max"]]
    return grew, fell


def dump(doc: dict) -> str:
    return json.dumps(doc, indent=1, sort_keys=True) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("write", "check", "show"))
    ap.add_argument("--json", action="store_true", help="show: print JSON")
    args = ap.parse_args()
    totals, per_file = measure()
    if args.command == "show":
        if args.json:
            print(json.dumps({"totals": totals, "files": per_file}, indent=1, sort_keys=True))
            return 0
        for title, keys in (("done", DONE), ("gaps", GAPS)):
            print(f"## {title}")
            for k, what in keys.items():
                top = sorted(per_file[k].items(), key=lambda kv: (-kv[1], kv[0]))[:3]
                where = ", ".join(f"{'/'.join(Path(f).parts[-2:])} {v}" for f, v in top) if k not in (
                    "files", "lines") else ""
                print(f"{totals[k]:>7}  {what}" + (f"  ({where})" if where else ""))
        return 0
    text = dump(committed(totals, per_file))
    if args.command == "write":
        OUT.write_text(text)
        return 0
    ceiling = json.loads(OUT.read_text()).get("ceiling", {}) if OUT.exists() else {}
    grew, fell = ratchet(ceiling, totals)
    for k in fell:
        print(f"  {k}: {ceiling[k]['max']} -> {totals[k]} (may lower the ceiling: make c23-metrics)")
    if not grew:
        return 0
    print(f"{OUT.relative_to(ROOT)}: a gap grew past its ceiling. Remove the new construct, "
          "or raise the ceiling with `make c23-metrics` and commit it:")
    for k in grew:
        cap = ceiling.get(k, {}).get("max")
        top = sorted(per_file[k].items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        print(f"  {k}: {cap} -> {totals[k]}  (largest: " + ", ".join(f"{f} {v}" for f, v in top) + ")")
    return 1

if __name__ == "__main__":
    sys.exit(main())
