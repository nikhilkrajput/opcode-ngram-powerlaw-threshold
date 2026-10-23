"""
Step 23 -- accuracy against the feature budget and repeated cross-validation,
at n = 3 (Sections 6.3 and 6.6).

(a) ACCURACY AGAINST k.  The power-law set is the top-k by frequency with k
    chosen by the fit, so comparisons against knee (~400) and top-5,000 are
    comparisons of k. This traces LightGBM accuracy and macro F1 for top-K by
    in-fold corpus frequency over a grid of K, on the same five folds as
    Table 5 (StratifiedKFold, seed 42), so the reader can see where the
    fitted k lands on the curve (Fig. 2).

(b) REPEATED CROSS-VALIDATION WITH A CORRECTED TEST (Table 7).  Table 5
    rests on one
    5-fold split. Here 5 x 5 repeated stratified CV compares power-law,
    chi-squared (at the power-law k of each fold), knee detection, top-5,000
    and the unreduced matrix. Differences against power-law are tested with
    the Nadeau--Bengio corrected resampled t-test, which inflates the variance
    by (1/J + n_test/n_train) to account for overlapping training sets, and
    Holm-corrected across the four comparisons for each metric. Mutual
    information is omitted: at 5,336 s per five folds it would add about
    7.4 h for a selector that Table 5 already shows is not significantly
    different.

Both parts checkpoint after every fit, so a killed run resumes.

Usage
-----
    python 23_kcurve_repcv.py --manifest ../results/malware/manifest.json \
        --labels /path/to/trainLabels.csv --out-dir ../results/malware \
        [--parts kcurve repcv] [--repeats 5]

Output
------
    <out-dir>/kcurve_3gram.json, <out-dir>/repcv_3gram.json
"""

import argparse
import importlib.util
import os
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def _load(name, file):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bc = _load("bc", "05_baseline_comparison.py")
kn = _load("kn", "06_knee_baseline.py")

K_GRID = [100, 250, 500, 1000, 2000, 4000, 7000, 10000, 20000, 50000,
          100000, 200000]


def make_clf():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                          verbosity=-1, random_state=bc.SEED,
                          n_jobs=int(os.environ.get("NJOBS", "-1")))


def fit_score(X_tr, y_tr, X_te, y_te, idx=None):
    t0 = time.time()
    clf = make_clf()
    if idx is None:
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
    else:
        clf.fit(X_tr[:, idx], y_tr)
        pred = clf.predict(X_te[:, idx])
    return {"acc": 100 * accuracy_score(y_te, pred),
            "macro_f1": 100 * f1_score(y_te, pred, average="macro",
                                       zero_division=0),
            "seconds": round(time.time() - t0, 1),
            "pred": pred.tolist()}


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.d = json.loads(self.path.read_text()) if self.path.exists() else {}

    def get(self, key):
        return self.d.get(key)

    def put(self, key, val):
        self.d[key] = val
        self.path.write_text(json.dumps(self.d))


def pooled(rows, tes, y):
    """Accuracy and macro F1 over the pooled out-of-fold predictions, the same
    convention as Table 5 (per-fold macro F1 would weight ~8 Simda samples
    per fold very differently)."""
    pred = np.empty_like(y)
    for r, te in zip(rows, tes):
        pred[te] = r["pred"]
    return {"acc": round(100 * accuracy_score(y, pred), 2),
            "macro_f1": round(100 * f1_score(y, pred, average="macro",
                                             zero_division=0), 2)}


def run_kcurve(X, y, out_dir):
    ck = Checkpoint(out_dir / "kcurve_3gram.ckpt.json")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=bc.SEED)
    rows = {K: [] for K in K_GRID}
    pl_k, tes = [], []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        tes.append(te)
        freqs = np.asarray(X_tr.sum(axis=0)).ravel()
        order = np.argsort(freqs)[::-1]
        key = f"pl{fold}"
        if ck.get(key) is None:
            idx_pl, p = bc.select_powerlaw(X_tr, y_tr)
            ck.put(key, {"k": int(len(idx_pl)), "xmin": p["xmin"]})
        pl_k.append(ck.get(key)["k"])
        for K in K_GRID:
            key = f"f{fold}_K{K}"
            if ck.get(key) is None:
                ck.put(key, fit_score(X_tr, y_tr, X_te, y_te, order[:K]))
                r = ck.get(key)
                print(f"  kcurve fold {fold} K={K:,}: acc={r['acc']:.2f} "
                      f"macroF1={r['macro_f1']:.2f} ({r['seconds']}s)", flush=True)
            rows[K].append(ck.get(key))
    out = {"n": 3, "folds": "StratifiedKFold(5, shuffle, seed 42), as Table 5",
           "metric_note": "pooled out-of-fold predictions, as Table 5",
           "powerlaw_k_per_fold": pl_k,
           "curve": [{"K": K, **pooled(rows[K], tes, y)} for K in K_GRID]}
    (out_dir / "kcurve_3gram.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["curve"], indent=1))


def nb_test(d, n_tr, n_te):
    """Nadeau--Bengio corrected resampled t-test on per-fold differences."""
    d = np.asarray(d, float)
    J = len(d)
    var = np.var(d, ddof=1) * (1.0 / J + n_te / n_tr)
    t = d.mean() / np.sqrt(var) if var > 0 else 0.0
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    half = stats.t.ppf(0.975, df=J - 1) * np.sqrt(var)
    return {"mean_diff": round(float(d.mean()), 3), "t": round(float(t), 3),
            "p": float(p), "ci95": [round(float(d.mean() - half), 3),
                                    round(float(d.mean() + half), 3)]}


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def run_repcv(X, y, out_dir, repeats):
    ck = Checkpoint(out_dir / "repcv_3gram.ckpt.json")
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats,
                                   random_state=bc.SEED)
    methods = ["power-law", "chi2", "knee", "top-5000", "full"]
    res = {m: [] for m in methods}
    ks, n_tr, n_te = [], None, None
    for j, (tr, te) in enumerate(rskf.split(X, y)):
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        n_tr, n_te = len(tr), len(te)
        key = f"s{j}_pl"
        if ck.get(key) is None:
            idx_pl, p = bc.select_powerlaw(X_tr, y_tr)
            ck.put(key, {"idx": idx_pl.tolist(), "xmin": p["xmin"]})
        idx_pl = np.asarray(ck.get(key)["idx"])
        k = len(idx_pl)
        ks.append({"split": j, "k": int(k), "xmin": ck.get(key)["xmin"]})
        for m in methods:
            key = f"s{j}_{m}"
            if ck.get(key) is None:
                if m == "power-law":
                    idx = idx_pl
                elif m == "chi2":
                    idx, _ = bc.select_chi2(X_tr, y_tr, k)
                elif m == "knee":
                    idx, _ = kn.select_knee(X_tr, y_tr)
                elif m == "top-5000":
                    idx, _ = bc.select_top_freq(X_tr, y_tr, 5000)
                else:
                    idx = None
                ck.put(key, fit_score(X_tr, y_tr, X_te, y_te, idx))
                r = ck.get(key)
                print(f"  repcv split {j + 1} {m}: acc={r['acc']:.2f} "
                      f"macroF1={r['macro_f1']:.2f} ({r['seconds']}s)", flush=True)
            res[m].append(ck.get(key))
    out = {"n": 3, "design": f"RepeatedStratifiedKFold(5 folds x {repeats} repeats, seed 42)",
           "test": "Nadeau-Bengio corrected resampled t-test, Holm over the four comparisons per metric",
           "splits": len(ks), "powerlaw_k": ks,
           "powerlaw_low_mode_splits": int(sum(x["xmin"] < 100 for x in ks)),
           "summary": {}, "vs_powerlaw": {}}
    for m in methods:
        out["summary"][m] = {
            met: {"mean": round(float(np.mean([r[met] for r in res[m]])), 2),
                  "sd": round(float(np.std([r[met] for r in res[m]], ddof=1)), 2)}
            for met in ("acc", "macro_f1")}
    for met in ("acc", "macro_f1"):
        tests = {m: nb_test([b[met] - a[met] for a, b in zip(res["power-law"], res[m])],
                            n_tr, n_te) for m in methods if m != "power-law"}
        adj = holm([t["p"] for t in tests.values()])
        for (m, t), a in zip(tests.items(), adj):
            t["p_holm"] = float(a)
        out["vs_powerlaw"][met] = tests
    (out_dir / "repcv_3gram.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("summary", "vs_powerlaw")}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--parts", nargs="+", default=["kcurve", "repcv"])
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    X, names = bc.build_full_matrix(args.manifest, 3)
    y = LabelEncoder().fit_transform(bc.load_labels(args.labels, names))
    if "kcurve" in args.parts:
        run_kcurve(X, y, out_dir)
    if "repcv" in args.parts:
        run_repcv(X, y, out_dir, args.repeats)


if __name__ == "__main__":
    main()
