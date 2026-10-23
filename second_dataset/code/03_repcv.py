"""
Step 4 -- selector comparison on 20 Newsgroups, 5 x 5 repeated stratified CV
(Table 9 of the article).

Every selector is refitted inside each training fold:
  power-law  freq >= x_min of powerlaw.Fit (continuous)   no labels, no k
  chi2       chi-squared top-k, k = that fold's power-law k   labels
  min_df 2/5/10  document-frequency thresholds           no labels, hand-chosen
  knee       max distance from the log-log chord         no labels, no k
  top-5000   top 5,000 by frequency                      no labels, fixed k
  full       no selection
Classifier as in common.py (TF-IDF + LinearSVC, fixed). Each selector is
tested against power-law with the Nadeau--Bengio corrected resampled t-test
(variance factor 1/J + n_test/n_train, J = 25) and Holm-corrected across the
seven comparisons per metric. Per-split metrics are within the test fold.

Checkpoints after every fit (results/repcv.ckpt.json); a killed run resumes.

Output: results/repcv.json
"""

import argparse
import json

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

import common as C

METHODS = ["power-law", "chi2", "min_df-2", "min_df-5", "min_df-10", "knee",
           "top-5000", "full"]
LABEL_USE = {"power-law": "no labels, no k", "chi2": "labels, k from power-law",
             "min_df-2": "no labels, hand-chosen cut-off",
             "min_df-5": "no labels, hand-chosen cut-off",
             "min_df-10": "no labels, hand-chosen cut-off",
             "knee": "no labels, no k", "top-5000": "no labels, fixed k",
             "full": "no selection"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    X, y, meta = C.build_matrix()
    ck = C.Checkpoint(C.RESULTS / "repcv.ckpt.json")
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats,
                                   random_state=C.SEED)
    res = {m: [] for m in METHODS}
    pl_info, n_tr, n_te = [], None, None
    for j, (tr, te) in enumerate(rskf.split(X, y)):
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        n_tr, n_te = len(tr), len(te)
        key = f"s{j}_plsel"
        idx_pl = None
        if ck.get(key) is None:
            idx_pl, p = C.select_powerlaw(X_tr)
            ck.put(key, {"k": int(len(idx_pl)), **p})
        info = ck.get(key)
        pl_info.append({"split": j, **info})
        for m in METHODS:
            key = f"s{j}_{m}"
            if ck.get(key) is None:
                if m in ("power-law", "chi2") and idx_pl is None:
                    idx_pl, _ = C.select_powerlaw(X_tr)
                    assert len(idx_pl) == info["k"]
                if m == "power-law":
                    idx = idx_pl
                elif m == "chi2":
                    idx, _ = C.select_chi2(X_tr, y_tr, info["k"])
                elif m.startswith("min_df"):
                    idx, _ = C.select_min_df(X_tr, y_tr, int(m.split("-")[1]))
                elif m == "knee":
                    idx, _ = C.select_knee(X_tr)
                elif m == "top-5000":
                    idx, _ = C.select_top_freq(X_tr, y_tr, 5000)
                else:
                    idx = None
                r = C.fit_score(X_tr, y_tr, X_te, y_te, idx)
                del r["pred"]
                ck.put(key, r)
                print(f"split {j + 1:2d} {m:10s} k={r['k']:>9,} acc={r['acc']:.2f} "
                      f"macroF1={r['macro_f1']:.2f} ({r['seconds']}s)", flush=True)
            res[m].append(ck.get(key))

    out = {"design": f"RepeatedStratifiedKFold(5 folds x {args.repeats} repeats, seed 42)",
           "classifier": C.CLASSIFIER_NOTE,
           "test": ("Nadeau-Bengio corrected resampled t-test (variance x (1/J + "
                    "n_test/n_train)), difference = method - power-law in percentage "
                    "points, Holm over the seven comparisons per metric; 95% CI is "
                    "the corrected t interval of the mean difference"),
           "metric_note": "per-split accuracy / macro F1 on the test fold, averaged over splits",
           "J": len(pl_info), "n_train": n_tr, "n_test": n_te,
           "powerlaw_per_split": pl_info,
           "powerlaw_xmin_values": sorted({p["xmin"] for p in pl_info}),
           "label_use": LABEL_USE, "summary": {}, "vs_powerlaw": {}}
    for m in METHODS:
        ks = [r["k"] for r in res[m]]
        out["summary"][m] = {
            "k_mean": round(float(np.mean(ks)), 1), "k_min": int(min(ks)),
            "k_max": int(max(ks)),
            "seconds_mean": round(float(np.mean([r["seconds"] for r in res[m]])), 1),
            **{met: {"mean": round(float(np.mean([r[met] for r in res[m]])), 2),
                     "sd": round(float(np.std([r[met] for r in res[m]], ddof=1)), 2)}
               for met in ("acc", "macro_f1")}}
    for met in ("acc", "macro_f1"):
        tests = {m: C.nb_test([b[met] - a[met] for a, b in zip(res["power-law"], res[m])],
                              n_tr, n_te) for m in METHODS if m != "power-law"}
        adj = C.holm([t["p"] for t in tests.values()])
        for (m, t), a in zip(tests.items(), adj):
            t["p_holm"] = float(a)
            t["significant_holm_0.05"] = bool(a < 0.05)
        out["vs_powerlaw"][met] = tests
    (C.RESULTS / "repcv.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("summary", "vs_powerlaw")}, indent=1))


if __name__ == "__main__":
    main()
