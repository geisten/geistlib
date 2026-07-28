#!/bin/sh
# check-api-contract.sh — fail if the documented API contract and the headers
# disagree.
#
# docs/API_CONTRACT.md lists the symbols an out-of-tree agent runtime links
# across a release boundary. This guard proves three things about that list:
#
#   1. every listed symbol is DECLARED in include/geist.h or include/geist_util.h;
#   2. every listed symbol carries an @stability STABLE tag (a contract symbol
#      must never be EXPERIMENTAL);
#   3. every symbol the contract smoke pins is actually IN the document, so the
#      list cannot silently grow past what was promised.
#
# The complementary half — signatures and linkability — is
# examples/agent_contract_smoke.c via `make agent-contract-smoke`.
#
# ponytail: a guard, not a generator. It does not edit anything; it makes drift
# a build failure.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
doc=$root/docs/API_CONTRACT.md
smoke=$root/examples/agent_contract_smoke.c
headers="$root/include/geist.h $root/include/geist_util.h"

[ -r "$doc" ] || { echo "check-api-contract: missing $doc" >&2; exit 2; }
[ -r "$smoke" ] || { echo "check-api-contract: missing $smoke" >&2; exit 2; }

# The contract IS the tables: only backticked `geist_*` names in table rows
# count, so a passing mention in prose never stands in for a promise. The
# "Explicitly NOT in the contract" section is skipped on purpose — the symbols
# listed there are EXPERIMENTAL by design.
symbols=$(awk '/^### / { f = ($0 !~ /Explicitly NOT/) } f && /^\|/' "$doc" |
    grep -oE '`geist_[a-z0-9_]+`' | tr -d '`' | sort -u)
[ -n "$symbols" ] || { echo "check-api-contract: no symbols found in $doc" >&2; exit 2; }

fail=0

for sym in $symbols; do
    # 1. declared somewhere in the public headers?
    # A declaration may start the line ([[nodiscard]] puts the return type on
    # the previous one) or follow a space / pointer star.
    decl=$(grep -hnE "(^|[ *])$sym\\(" $headers 2>/dev/null | head -1 || true)
    if [ -z "$decl" ]; then
        echo "  MISSING: $sym is in the contract but not declared in include/" >&2
        fail=1
        continue
    fi
    # 2. the nearest preceding @stability tag must say STABLE.
    tag=$(awk -v fn="$sym" '
        /@stability/ { s = $0 }
        $0 ~ "(^|[ *])" fn "\\(" { print s; exit }
    ' $headers | grep -oE 'STABLE since [0-9][0-9.]*|EXPERIMENTAL' || true)
    case "$tag" in
        STABLE*) ;;
        *)
            echo "  UNSTABLE: $sym is in the contract but tagged '${tag:-<none>}'" >&2
            fail=1
            ;;
    esac
done

# 3. the smoke must not pin anything the document does not promise.
documented=" $(printf '%s ' $symbols)"
for sym in $(grep -oE '"geist_[a-z0-9_]+"' "$smoke" | tr -d '"' | sort -u); do
    case "$documented" in
        *" $sym "*) ;;
        *)
            echo "  UNDOCUMENTED: $sym is pinned by the smoke but absent from the contract" >&2
            fail=1
            ;;
    esac
done

if [ "$fail" -ne 0 ]; then
    echo "check-api-contract: FAIL — docs/API_CONTRACT.md and include/ disagree" >&2
    exit 1
fi

count=$(printf '%s\n' $symbols | wc -l | tr -d ' ')
echo "api contract OK: $count symbols documented, declared, and tagged STABLE"
