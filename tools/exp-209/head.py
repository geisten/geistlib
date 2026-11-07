#!/opt/homebrew/bin/python3.14
"""#209 pilot: a linear head on Bonsai 2 27B's frozen last-token logits (chat_direct prompt).

Features: the full logit vector at the last prompt token. It is W h for the final
hidden state h (rank <= d_model), so a linear head on it is a linear head on h.
Reduced by PCA fitted on the training split only (centred, top k components,
whitened), then multinomial logistic regression over the four letters with L2,
full-batch gradient descent. Train 2,000 / calibration 500 (C and k chosen there,
then temperature), test = the #171/#236 500 (never used for any choice).

usage: head.py <dir>   (reads logits_{train,test}.bin, items_*.bin.order.json)
"""
import json, sys
import numpy as np

D = sys.argv[1]
rng = np.random.default_rng(209)


def load(split):
    with open(f"{D}/logits_{split}.bin", "rb") as f:
        n, v = np.frombuffer(f.read(8), dtype=np.uint32)
        x = np.fromfile(f, dtype=np.float32, count=int(n) * int(v)).reshape(int(n), int(v))
    order = json.load(open(f"{D}/items_{split}.bin.order.json"))
    y = np.array([o["target_index"] for o in order])
    cand = np.array([o["candidate_ids"] for o in order])
    return x, y, cand, order


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1)
    e = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def fit(z, y, lam, steps=3000, lr=0.5):
    n, k = z.shape
    w, b = np.zeros((k, 4)), np.zeros(4)
    onehot = np.eye(4)[y]
    for _ in range(steps):
        p = softmax(z @ w + b)
        g = p - onehot
        w -= lr * (z.T @ g / n + lam * w)
        b -= lr * g.mean(0)
    return w, b


def temperature(logit, y):
    best = (1e9, 1.0)
    for t in np.exp(np.linspace(np.log(0.2), np.log(5), 60)):
        p = softmax(logit / t)
        nll = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        best = min(best, (nll, t))
    return best[1]


xtr_all, ytr_all, ctr_all, otr = load("train")
xte, yte, cte, ote = load("test")
assert len(otr) == 2500 and len(ote) == 500
perm = rng.permutation(2500)
tr, ca = perm[:2000], perm[2000:]

# direct scoring: argmax of the four letter logits (the #171 reference)
def direct(x, c):
    return np.take_along_axis(x, c, 1)
d_te = direct(xte, cte)
direct_acc = (d_te.argmax(1) == yte).mean()

mu = xtr_all[tr].mean(0)
xc = xtr_all[tr] - mu
# PCA through the 2000 x 2000 Gram matrix (n << vocab)
g = xc @ xc.T
evals, evecs = np.linalg.eigh(g)
idx = np.argsort(evals)[::-1]
evals, evecs = evals[idx], evecs[:, idx]
results = {}
for k in (64, 128, 256, 512, 1024):
    comps = (xc.T @ evecs[:, :k]) / np.sqrt(np.maximum(evals[:k], 1e-9))  # [vocab, k], unit-norm columns
    scale = np.sqrt(np.maximum(evals[:k], 1e-9) / len(tr))
    proj = lambda x: ((x - mu) @ comps) / scale  # whitened
    ztr, zca = proj(xtr_all[tr]), proj(xtr_all[ca])
    for lam in (1e-3, 1e-2, 1e-1, 1.0):
        w, b = fit(ztr, ytr_all[tr], lam)
        acc = ((zca @ w + b).argmax(1) == ytr_all[ca]).mean()
        results[(k, lam)] = (acc, w, b, comps, scale)
        print(f"k {k:5d} lam {lam:g}: calibration acc {acc:.3f}", flush=True)
(k, lam), (cal_acc, w, b, comps, scale) = max(results.items(), key=lambda kv: kv[1][0])
proj = lambda x: ((x - mu) @ comps) / scale
t = temperature(proj(xtr_all[ca]) @ w + b, ytr_all[ca])
p_te = softmax((proj(xte) @ w + b) / t)
head_acc = (p_te.argmax(1) == yte).mean()
t_d = temperature(direct(xtr_all[ca], ctr_all[ca]), ytr_all[ca])
p_d = softmax(d_te / t_d)

reason = json.load(open("/Users/germar/workspace/geistlib-campaigns/236c/test_baseline_correct.json"))
r_ok = np.array([reason[o["question_id"]] for o in ote])
head_ok, dir_ok = p_te.argmax(1) == yte, d_te.argmax(1) == yte
sys.path.insert(0, "/tmp/geistlib-209/tools")
from decision_metrics import paired_quality
ids = [o["question_id"] for o in ote]
pq_head_vs_reason = paired_quality(dict(zip(ids, map(bool, head_ok))), dict(zip(ids, map(bool, r_ok))), 2.0)
pq_head_vs_direct = paired_quality(dict(zip(ids, map(bool, head_ok))), dict(zip(ids, map(bool, dir_ok))), 2.0)
out = {
    "chosen": {"pca_k": k, "l2": lam, "calibration_acc": cal_acc, "temperature": t},
    "test_n": 500,
    "direct_scoring_acc": float(direct_acc), "direct_ece_after_temperature": float(ece(p_d, yte)),
    "head_acc": float(head_acc), "head_ece_after_temperature": float(ece(p_te, yte)),
    "full_reasoning_cap8192_acc": float(r_ok.mean()),
    "head_vs_full_reasoning": {k_: pq_head_vs_reason[k_] for k_ in ("gains", "losses", "difference_pp", "interval_pp", "noninferior")},
    "head_vs_direct": {k_: pq_head_vs_direct[k_] for k_ in ("gains", "losses", "difference_pp", "interval_pp")},
}
print(json.dumps(out, indent=1))
json.dump(out, open(f"{D}/head_result.json", "w"), indent=1)
