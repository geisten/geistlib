"""ptqtp_version — single source of truth for the PTQTP algorithm version.

Imported by every PTQTP entry point (`ptqtp_quantize_full.py`,
`ptqtp_single_tensor.py`, `ptqtp_quality_probe.py`) and by the P0 paper-
parity test. Recipe artifacts emitted by the quantizer must record this
string so a later reader can tell which algorithm produced them.

## Versioning rule

Bump `ALGORITHM_VERSION` when ANY of the following changes:

  * The alternation loop logic (`ptqtp_quantize_n`, `ptqtp_quantize`):
    initialization, T-update strategy, alpha-update strategy.
  * The default `group_size`, `max_iter`, or `tol` baked into the entry
    points (changing the default re-shifts every artifact that did not
    pass the value explicitly).
  * The ridge solver: λ schedule, condition-number probe, fallback path.
  * Numerical dtype contract: which arrays are FP32 vs FP64 at which
    stage.

Do NOT bump for:

  * Refactors that preserve bit-identical output (CI test
    `tests/test_ptqtp_p0_paper_parity.py` covers this).
  * Docs, comments, type hints, logging, progress-bar changes.
  * New CLI flags that default to current behavior.

## Format

`<major>.<minor>.<patch>` plus an ISO date. The date is informational —
the version string itself is the contract.

## History

  v1.0.0 — 2026-06-02 — Initial pinned release. Locks the
    `ptqtp_quantize_n` alternating-minimization loop from
    arxiv 2509.16989 Algorithm 1 with:
      - sign() initialization for T1, zeros for T_{2..K}
      - per-(row, group) ridge regression with adaptive λ ∈ [1e-6, 1.0]
      - exhaustive 3^K candidate search for T update
      - default group_size=128, max_iter=50, tol=1e-4
      - FP32 working precision (FP64 only for cos_sim diagnostic)
    Validated by tests/test_ptqtp_p0_paper_parity.py at cos sim ≥ 0.95
    on a synthetic Gemma-class attn_q tensor (shape 1024×1024, std=0.02,
    seed=0). Measured cos sim 0.985, bit-identical re-run.
"""

# The contract string. Embedded in recipe JSON, .ptqtp.bin headers (future),
# and any benchmark output that compares quant artifacts.
ALGORITHM_VERSION: str = "ptqtp-1.0.0"

# Date is informational only; not part of the contract.
ALGORITHM_VERSION_DATE: str = "2026-06-02"


def header_blob() -> dict:
    """Compact dict for embedding into recipe / artifact metadata."""
    return {
        "ptqtp_algorithm_version": ALGORITHM_VERSION,
        "ptqtp_algorithm_date":    ALGORITHM_VERSION_DATE,
    }
