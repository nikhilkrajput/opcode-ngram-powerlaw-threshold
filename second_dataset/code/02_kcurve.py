"""
Step 3 -- accuracy and macro F1 against K on 20 Newsgroups.

Top-K features by in-fold corpus frequency over a log grid of K from 100 to
the full vocabulary, StratifiedKFold(5, shuffle, seed 42), classifier as in
common.py (TF-IDF + LinearSVC, fixed). The power-law selection of each fold
(freq >= x_min, refitted on the training fold) is scored on the same folds,
so its k can be placed on the curve. Metrics are pooled out-of-fold, as in
Table 5 of the article.

Checkpoints after every fit (results/kcurve.ckpt.json); a killed run resumes.

Output: results/kcurve.json
"""

import json

import numpy as np
from sklearn.model_selection import StratifiedKFold

import common as C

K_GRID = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000,
          200000, 500000]


def main():
    X, y, meta = C.build_matrix()
    V = X.shape[1]
    grid = K_GRID + [V]
    ck = C.Checkpoint(C.RESULTS / "kcurve.ckpt.json")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED)
    rows = {K: [] for K in grid}
    rows_pl, tes, pl = [], [], []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        tes.append(te)
        order = np.argsort(C.corpus_freqs(X_tr), kind="stable")[::-1]
        key = f"pl{fold}"
        if ck.get(key) is None:
            idx, p = C.select_powerlaw(X_tr)
            r = C.fit_score(X_tr, y_tr, X_te, y_te, idx)
            r.update(p)
            ck.put(key, r)
        r = ck.get(key)
        rows_pl.append(r)
        pl.append({"fold": fold, "k": r["k"], "xmin": r["xmin"], "alpha": r["alpha"]})
        print(f"fold {fold} power-law: xmin={r['xmin']} k={r['k']:,} "
              f"acc={r['acc']:.2f}", flush=True)
        for K in grid:
            key = f"f{fold}_K{K}"
            if ck.get(key) is None:
                idx = None if K == V else order[:K]
                ck.put(key, C.fit_score(X_tr, y_tr, X_te, y_te, idx))
                r = ck.get(key)
                print(f"  fold {fold} K={K:,}: acc={r['acc']:.2f} "
                      f"macroF1={r['macro_f1']:.2f} ({r['seconds']}s)", flush=True)
            rows[K].append(ck.get(key))

    curve = [{"K": K, **C.pooled(rows[K], tes, y),
              "fold_acc_mean": round(float(np.mean([r["acc"] for r in rows[K]])), 2)}
             for K in grid]
    full = curve[-1]
    plp = C.pooled(rows_pl, tes, y)
    out = {"folds": "StratifiedKFold(5, shuffle, seed 42)",
           "classifier": C.CLASSIFIER_NOTE,
           "selection": "top-K by in-fold corpus frequency (ties broken by column order)",
           "metric_note": "pooled out-of-fold predictions",
           "full_vocabulary_K": int(V),
           "powerlaw_per_fold": pl,
           "powerlaw_k_mean": round(float(np.mean([p["k"] for p in pl])), 1),
           "powerlaw_pooled": plp,
           "curve": curve}
    for met in ("acc", "macro_f1"):
        reach = [c["K"] for c in curve if c[met] >= full[met]]
        within = [c["K"] for c in curve if c[met] >= full[met] - 0.5]
        best = max(curve, key=lambda c: c[met])
        out[f"{met}_full"] = full[met]
        out[f"{met}_best"] = {"K": best["K"], met: best[met]}
        out[f"{met}_smallest_K_reaching_full"] = reach[0] if reach else None
        out[f"{met}_smallest_K_within_0.5pp_of_full"] = within[0] if within else None
        out[f"{met}_powerlaw_minus_full"] = round(plp[met] - full[met], 2)
    (C.RESULTS / "kcurve.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "curve"}, indent=1))
    for c in curve:
        print(c)


if __name__ == "__main__":
    main()
