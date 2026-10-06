"""Explicit split and duplicate policies for offline, labelled decision cases."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def read_json(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def write_json(path, value, ensure_ascii=False):
    path.write_text(json.dumps(value, ensure_ascii=ensure_ascii, indent=2, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_question(row):
    if (not isinstance(row, dict) or not isinstance(row.get("subject"), str)
            or not row["subject"] or not isinstance(row.get("question"), str) or not row["question"].strip()
            or not isinstance(row.get("choices"), list) or len(row["choices"]) != 4
            or any(not isinstance(c, str) or not c.strip() for c in row["choices"])
            or type(row.get("answer")) is not int or not 0 <= row["answer"] < 4):
        raise ValueError("expected subject, question, four choices and integer answer 0..3")


def question_key(row):
    validate_question(row)
    # Match reordered answer options too; preserve meaningful internal
    # whitespace such as code indentation. No near-duplicate claim.
    normalize = lambda text: text.replace("\r\n", "\n").strip()
    return digest([normalize(row["question"]), sorted(normalize(c) for c in row["choices"])])


def partition(sources, overlap_policy="reject", seed="geist-decision-v1"):
    """Keep dev exemplars; split validation; reserve all earlier question keys.

    Exclusion changes the benchmark population and must be explicitly chosen
    and reported. It never silently sanitizes an alleged stock MMLU score.
    """
    if set(sources) != {"dev", "validation", "test"} or overlap_policy not in ("reject", "exclude"):
        raise ValueError("dev/validation/test sources and explicit overlap policy required")
    records = {}
    golds = defaultdict(set)
    for split in ("dev", "validation", "test"):
        records[split] = []
        for index, row in enumerate(sources[split]):
            key = question_key(row)
            item = {**row, "question_id": key, "source_split": split,
                    "source_index": index, "id": f"{split}/{row['subject']}/{index}"}
            records[split].append(item)
            golds[key].add(row["choices"][row["answer"]].strip())
    conflicts = {key for key, answers in golds.items() if len(answers) > 1}
    few_shots = defaultdict(list)
    for row in records["dev"]:
        if row["question_id"] in conflicts:
            raise ValueError("conflicting annotations touch the fixed few-shot exemplars")
        few_shots[row["subject"]].append(row)
    reserved = {r["question_id"] for r in records["dev"]}
    excluded, cleaned = [], {}
    for split in ("validation", "test"):
        used, cleaned[split] = set(), []
        for row in records[split]:
            key = row["question_id"]
            reason = ("conflicting_annotation" if key in conflicts else
                      "earlier_split_overlap" if key in reserved else
                      "within_split_duplicate" if key in used else None)
            used.add(key)
            if reason:
                if overlap_policy == "reject":
                    raise ValueError(f"{row['id']}: {reason}; explicit exclusion policy required")
                excluded.append({"id": row["id"], "question_id": key, "reason": reason})
            else:
                cleaned[split].append(row)
        # Also reserve excluded records so validation/test cannot leak.
        reserved.update(r["question_id"] for r in records[split])
    result = {"development": [], "calibration": [], "test": []}
    by_subject = defaultdict(list)
    for row in cleaned["validation"]:
        by_subject[row["subject"]].append(row)
    for subject, rows in sorted(by_subject.items()):
        rows.sort(key=lambda r: digest([seed, r["question_id"]]))
        for i, row in enumerate(rows):
            split = "development" if i % 2 == 0 else "calibration"
            result[split].append({**row, "split": split})
    result["test"] = [{**r, "split": "test"} for r in cleaned["test"]]
    sets = [set(r["question_id"] for r in result[s]) for s in result]
    if any(a & b for i, a in enumerate(sets) for b in sets[i + 1:]):
        raise ValueError("split separation invariant failed")
    return result, dict(few_shots), {
        "policy": overlap_policy, "seed": seed,
        "source_counts": {s: len(rows) for s, rows in records.items()},
        "prepared_counts": {s: len(rows) for s, rows in result.items()},
        "excluded_counts": dict(Counter(r["reason"] for r in excluded)), "excluded": excluded,
        "scope": "Exact question/option matches, including option permutations; not semantic near-duplicate detection"}


def select_per_subject(rows, count, seed="geist-decision-pilot-v1", subject_limit=0):
    if (type(count) is not int or count < 0
            or type(subject_limit) is not int or subject_limit < 0):
        raise ValueError("nonnegative per-subject count and subject limit required")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["subject"]].append(row)
    selected = []
    subjects = sorted(grouped, key=lambda s: digest([seed, "subject", s]))
    chosen = set(subjects[:subject_limit] if subject_limit else subjects)
    for subject, group in sorted(grouped.items()):
        if subject not in chosen:
            continue
        group.sort(key=lambda r: digest([seed, r["question_id"]]))
        selected.extend(group[:count] if count else group)
    return selected


def permute(row, shift):
    if type(shift) is not int or not 0 <= shift < 4:
        raise ValueError("rotation 0..3 required")
    order = [(i + shift) % 4 for i in range(4)]
    return {**row, "choices": [row["choices"][i] for i in order],
            "answer": order.index(row["answer"]), "variant": shift,
            "option_order": order, "id": f"{row['id']}/rotation{shift}"}
