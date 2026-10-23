"""
Step 9 -- a class-conditional threshold (Section 8.1, Eq. (5)).

WHY
---
The corpus-level threshold trails chi-squared on macro F1 (98.87 vs 99.23,
Table 5), and Table 6 shows the deficit is almost entirely one family: on Simda,
42 samples, x_min scores F1 = 94.12 against chi-squared's 97.56. The reason is
structural. x_min is fitted on pooled corpus frequencies, so an n-gram that is
rare overall but concentrated in one small family falls below the threshold
and is discarded. A supervised selector sees the labels and keeps it.

THE VARIANT
-----------
Fit x_min separately on each family's own frequency vector, then retain the
union of the per-family sets:

    keep = U_c { g : f_g^(c) >= xmin-hat^(c) }

An n-gram that is common *within* Simda now clears Simda's own threshold even
though it is invisible in the pooled distribution.

WHAT IT COSTS
-------------
This is no longer unsupervised. Partitioning by family needs labels, so the
variant sits between the corpus-level threshold and chi-squared rather than
replacing either. It keeps the "no k" property -- each family's threshold still
comes from its own fit -- but loses the "no labels" property that is the
corpus-level rule's main argument. Reporting it as a strict improvement would
be wrong.

The union can also be large: nine thresholds each admit their own tail. k is
reported per fold precisely so that a reduction bought back by retaining far
more features is visible rather than hidden.

Compared on identical folds against the corpus-level threshold and against
chi-squared given the union's k, so no selector gets a dimensionality
advantage.

Usage
-----
    python 09_class_conditional.py --manifest ../results/malware/manifest.json \\
        --labels /path/to/trainLabels.csv --n 3 --out-dir ../results/malware
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import powerlaw
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

_spec = importlib.util.spec_from_file_location(
    "bl", Path(__file__).resolve().parent / "05_baseline_comparison.py")
bl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bl)

SEED = 42
MIN_NONZERO = 50   # below this a per-class fit is meaningless; keep all nonzero


def select_class_conditional(X_tr, y_tr, classes):
    """Union of per-family power-law thresholds. Returns indices and detail."""
    keep = np.zeros(X_tr.shape[1], dtype=bool)
    detail = {}
    for c in classes:
        rows = np.flatnonzero(y_tr == c)
        if rows.size == 0:
            continue
        freqs = np.asarray(X_tr[rows].sum(axis=0)).ravel()
        nz = freqs[freqs > 0]
        if nz.size < MIN_NONZERO:
            # too few distinct n-grams to fit; keep everything this class has
            sel = freqs > 0
            detail[int(c)] = {"xmin": None, "alpha": None,
                              "kept": int(sel.sum()), "note": "fit skipped"}
        else:
            fit = powerlaw.Fit(nz, verbose=False)
            sel = freqs >= fit.xmin
            detail[int(c)] = {"xmin": float(fit.xmin),
                              "alpha": float(fit.alpha),
                              "kept": int(sel.sum()),
                              "class_vocab": int(nz.size)}
        keep |= sel
    return np.flatnonzero(keep), detail


def run(X, y, n, out_dir):
    from lightgbm import LGBMClassifier

    def clf():
        return LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                              verbosity=-1, random_state=SEED, n_jobs=-1)

    enc = LabelEncoder().fit(y)
    y_enc = enc.transform(y)
    names = [bl.FAMILY[c] for c in enc.classes_]
    uniq = np.unique(y_enc)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    methods = ["corpus-level", "class-conditional", "chi2-matched"]
    preds = {m: np.zeros_like(y_enc) for m in methods}
    ks = {m: [] for m in methods}
    secs = {m: 0.0 for m in methods}
    per_fold_detail = []

    for fold, (tr, te) in enumerate(skf.split(X, y_enc), 1):
        X_tr, X_te, y_tr = X[tr], X[te], y_enc[tr]

        t0 = time.time()
        idx_corpus, p_corpus = bl.select_powerlaw(X_tr, y_tr)
        secs["corpus-level"] += time.time() - t0

        t0 = time.time()
        idx_cc, detail = select_class_conditional(X_tr, y_tr, uniq)
        secs["class-conditional"] += time.time() - t0
        per_fold_detail.append({names[i]: detail.get(int(c), {})
                                for i, c in enumerate(uniq)})

        t0 = time.time()
        idx_chi, _ = bl.select_chi2(X_tr, y_tr, len(idx_cc))
        secs["chi2-matched"] += time.time() - t0

        print(f"  fold {fold}: corpus k={len(idx_corpus):,} "
              f"(xmin={p_corpus['xmin']:.0f})   "
              f"class-cond k={len(idx_cc):,}   chi2 matched at {len(idx_chi):,}",
              flush=True)

        for m, idx in (("corpus-level", idx_corpus),
                       ("class-conditional", idx_cc),
                       ("chi2-matched", idx_chi)):
            ks[m].append(int(len(idx)))
            c = clf()
            c.fit(X_tr[:, idx], y_tr)
            preds[m][te] = c.predict(X_te[:, idx])

    res = {"n": n, "n_samples": int(X.shape[0]),
           "full_vocabulary": int(X.shape[1]),
           "per_fold_class_detail": per_fold_detail, "methods": {}}

    print(f"\n  {'variant':<20}{'acc':>8}{'macro F1':>10}{'mean k':>10}")
    for m in methods:
        acc = 100 * accuracy_score(y_enc, preds[m])
        macro = 100 * f1_score(y_enc, preds[m], average="macro", zero_division=0)
        fam = f1_score(y_enc, preds[m], average=None,
                       labels=range(len(names)), zero_division=0)
        res["methods"][m] = {
            "accuracy": round(acc, 2), "macro_f1": round(macro, 2),
            "k_per_fold": ks[m], "k_mean": round(float(np.mean(ks[m])), 1),
            "selection_seconds": round(secs[m], 1),
            "per_family_f1": {names[i]: round(100 * float(v), 2)
                              for i, v in enumerate(fam)},
        }
        print(f"  {m:<20}{acc:7.2f}%{macro:9.2f}%{np.mean(ks[m]):10,.0f}")

    print(f"\n  {'family':<18}" + "".join(f"{m[:14]:>16}" for m in methods))
    for i, fam_name in enumerate(names):
        row = "".join(f"{res['methods'][m]['per_family_f1'][fam_name]:16.2f}"
                      for m in methods)
        print(f"  {fam_name:<18}{row}")

    res["mcnemar"] = {}
    for m in ("class-conditional", "chi2-matched"):
        r = bl.mcnemar(preds["corpus-level"], preds[m], y_enc)
        res["mcnemar"][f"corpus-level vs {m}"] = r
        sig = "significant" if r["p"] < 0.05 else "not significant"
        print(f"\n  McNemar corpus-level vs {m:<20} p = {r['p']:.4g} ({sig})")

    out = Path(out_dir) / f"class_conditional_{n}gram.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nWritten to {out}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    X, names = bl.build_full_matrix(args.manifest, args.n)
    y = bl.load_labels(args.labels, names)
    run(X, y, args.n, out_dir)
