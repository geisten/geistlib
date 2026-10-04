"""Question-level decision metrics; calibration and test evidence stay separate."""
from __future__ import annotations

import math
import re
from statistics import mean


def log_probabilities(logits, temperature=1.0):
    if (not logits or not math.isfinite(temperature) or temperature <= 0
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in logits)):
        raise ValueError("finite logits and a positive finite temperature are required")
    scaled = [x / temperature for x in logits]
    maximum = max(scaled)
    if not math.isfinite(maximum) or any(not math.isfinite(x) for x in scaled):
        raise ValueError("temperature scaling overflow")
    shifted = [x - maximum for x in scaled]
    if any(not math.isfinite(x) for x in shifted):
        raise ValueError("log-probability range overflow")
    normalizer = math.log(sum(math.exp(x) for x in shifted))
    return [x - normalizer for x in shifted]


def quality(rows, temperature=1.0, bins=10):
    """One canonical row per question; repeats/permutations cannot inflate N."""
    if type(bins) is not int or bins <= 0:
        raise ValueError("positive bin count required")
    seen = set()
    losses, briers, correct = [], [], []
    groups = [[] for _ in range(bins)]
    subjects = {}
    for row in rows:
        name = row["question_id"]
        if name in seen or row.get("variant", 0) != 0:
            raise ValueError("quality requires unique canonical questions")
        seen.add(name)
        gold = row["target_index"]
        logits = row["logits"]
        if type(gold) is not int or not 0 <= gold < len(logits):
            raise ValueError("invalid gold index")
        lp = log_probabilities(logits, temperature)
        probabilities = [math.exp(x) for x in lp]
        prediction = max(range(len(logits)), key=lambda i: logits[i])
        hit = prediction == gold
        confidence = probabilities[prediction]
        losses.append(-lp[gold])
        briers.append(sum((p - (i == gold)) ** 2 for i, p in enumerate(probabilities)))
        correct.append(hit)
        groups[min(int(confidence * bins), bins - 1)].append((confidence, hit))
        subjects.setdefault(row.get("subject", "unknown"), []).append(hit)
    if not correct:
        raise ValueError("no labelled questions")
    reliability = [{"lower": i / bins, "upper": (i + 1) / bins, "n": len(g),
                    "mean_confidence": mean(p for p, _ in g) if g else None,
                    "accuracy": mean(hit for _, hit in g) if g else None}
                   for i, g in enumerate(groups)]
    return {"n": len(correct), "accuracy": mean(correct), "log_loss": mean(losses),
            "brier_sum_over_classes": mean(briers), "temperature": temperature,
            "ece_equal_width": sum(len(g) * abs(mean(p for p, _ in g) - mean(h for _, h in g))
                                    for g in groups if g) / len(correct),
            "reliability": reliability,
            "by_subject": {s: {"n": len(h), "accuracy": mean(h)} for s, h in sorted(subjects.items())}}


def fit_temperature(rows, binding):
    """Fit only canonical calibration questions, with explicit policy binding."""
    if not rows or not binding:
        raise ValueError("calibration data and policy binding required")
    if any(r.get("split") != "calibration" or r.get("variant", 0) != 0 for r in rows):
        raise ValueError("temperature may be fitted only on canonical calibration data")
    quality(rows)  # Validate IDs, labels, finite scores and uniqueness before fitting.
    def loss(log_t):
        t = math.exp(log_t)
        return mean(-log_probabilities(r["logits"], t)[r["target_index"]] for r in rows)
    lower, upper = math.log(.01), math.log(100.)
    ratio = (math.sqrt(5.) - 1.) / 2.
    a, b = upper - ratio * (upper - lower), lower + ratio * (upper - lower)
    fa, fb = loss(a), loss(b)
    for _ in range(70):
        if fa < fb:
            upper, b, fb = b, a, fa
            a = upper - ratio * (upper - lower)
            fa = loss(a)
        else:
            lower, a, fa = a, b, fb
            b = lower + ratio * (upper - lower)
            fb = loss(b)
    candidates = [(loss(x), x) for x in (math.log(.01), math.log(100.), 0., (lower + upper) / 2.)]
    best_loss, log_t = min(candidates)
    return {"temperature": math.exp(log_t), "binding": binding,
            "question_ids": sorted(r["question_id"] for r in rows), "n": len(rows),
            "calibration_log_loss_before": loss(0.), "calibration_log_loss_after": best_loss,
            "search_bounds": [.01, 100.], "at_boundary": abs(log_t) > math.log(100.) - 1e-5}


def apply_calibration(rows, calibration, binding):
    if binding != calibration["binding"]:
        raise ValueError("calibration policy/checkpoint binding mismatch")
    if set(r["question_id"] for r in rows) & set(calibration["question_ids"]):
        raise ValueError("calibration and evaluation questions overlap")
    return quality(rows, calibration["temperature"])


def parse_answer(text, thinking=False):
    """Strict final answer, excluding reasoning; no last-letter guessing."""
    if not isinstance(text, str):
        raise ValueError("answer text must be a string")
    if thinking:
        if text.count("</think>") != 1:
            return None
        text = text.split("</think>", 1)[1]
    if "<think>" in text or "</think>" in text:
        return None
    # End-of-turn controls are stripped only at the very end.
    text = re.sub(r"(?:<\|im_end\|>|<\|endoftext\|>)\s*$", "", text.strip()).strip()
    match = re.fullmatch(r"(?:Answer:\s*)?([ABCD])", text)
    return ord(match[1]) - ord("A") if match else None


def _binomial_cdf(n, k, p):
    if k < 0:
        return 0.
    if k >= n or p == 0:
        return 1.
    if p == 1:
        return 0.
    # Sum from the end nearest the mode towards the smaller tail. This
    # avoids underflow at P(X=0), including large held-out test sets.
    left = k < n * p
    i = k if left else k + 1
    term = math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                    + i * math.log(p) + (n - i) * math.log1p(-p))
    total = term
    if left:
        while i > 0:
            term *= i / (n - i + 1) * (1 - p) / p
            total += term
            i -= 1
            if term < total * 1e-15:
                break
        return min(total, 1.)
    while i < n:
        term *= (n - i) / (i + 1) * p / (1 - p)
        total += term
        i += 1
        if term < total * 1e-15:
            break
    return max(0., 1. - total)


def _exact_binomial_interval(n, k, alpha):
    if k == 0:
        lower = 0.
    else:
        lo, hi = 0., 1.
        for _ in range(65):
            mid = (lo + hi) / 2
            if _binomial_cdf(n, k - 1, mid) > 1 - alpha / 2:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2
    if k == n:
        upper = 1.
    else:
        lo, hi = 0., 1.
        for _ in range(65):
            mid = (lo + hi) / 2
            if _binomial_cdf(n, k, mid) > alpha / 2:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2
    return lower, upper


def paired_quality(direct, baseline, margin_pp, confidence=.95):
    if (not direct or set(direct) != set(baseline)
            or any(type(v) is not bool for v in (*direct.values(), *baseline.values()))
            or type(margin_pp) not in (int, float) or not math.isfinite(margin_pp)
            or not 0 <= margin_pp <= 100 or not 0 < confidence < 1):
        raise ValueError("paired unique boolean outcomes and finite margin required")
    n = len(direct)
    gains = sum(direct[k] and not baseline[k] for k in direct)
    losses = sum(baseline[k] and not direct[k] for k in direct)
    # Exact Clopper-Pearson intervals for the paired gain/loss events.
    # Bonferroni gives at least the requested joint coverage; unlike a
    # naive bootstrap, zero observed disagreements still has uncertainty.
    gain = _exact_binomial_interval(n, gains, (1 - confidence) / 2)
    loss = _exact_binomial_interval(n, losses, (1 - confidence) / 2)
    lower, upper = 100 * (gain[0] - loss[1]), 100 * (gain[1] - loss[0])
    return {"n": n, "gains": gains, "losses": losses,
            "direct_accuracy": mean(direct.values()), "baseline_accuracy": mean(baseline.values()),
            "difference_pp": 100 * (gains - losses) / n,
            "confidence": confidence, "interval_pp": [lower, upper], "margin_pp": margin_pp,
            "noninferior": lower >= -margin_pp,
            "interval_method": "paired gain/loss exact binomial intervals with Bonferroni joint coverage; conservative",
            "assumptions": "Independent questions from the target population. A small subject-stratified pilot is exploratory; repeated timings and label rotations are not additional quality samples."}
