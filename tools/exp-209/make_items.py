"""items.bin for dump_last_logits: chat_direct prompts, grouped by subject; the
prefix is the subject group's longest common prefix (each item keeps >= 1 own token)."""
import json, struct, sys
from itertools import groupby
cases, out = sys.argv[1], sys.argv[2]
recs = [json.loads(l) for l in open(cases)]
recs.sort(key=lambda r: (r["subject"], r["id"]))  # stable, subject-grouped
order = []
with open(out, "wb") as f:
    f.write(struct.pack("<I", len(recs)))
    tot = own = 0
    for subj, grp in groupby(recs, key=lambda r: r["subject"]):
        grp = list(grp)
        ids = [r["profiles"]["chat_direct"]["prompt_ids"] for r in grp]
        lcp = min(len(x) for x in ids) - 1
        for x in ids[1:]:
            k = 0
            while k < lcp and x[k] == ids[0][k]: k += 1
            lcp = k
        for r, x in zip(grp, ids):
            f.write(struct.pack("<II", lcp, len(x)) + struct.pack(f"<{len(x)}i", *x))
            order.append({"id": r["id"], "question_id": r["question_id"], "subject": subj,
                          "target_index": r["target_index"], "candidate_ids": r["profiles"]["chat_direct"]["candidate_ids"]})
            tot += len(x); own += len(x) - lcp
json.dump(order, open(out + ".order.json", "w"))
print(f"{len(recs)} items, mean prompt {tot/len(recs):.0f} tokens, mean own suffix {own/len(recs):.0f}")
