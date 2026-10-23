"""
Step 12 -- the unreduced baseline: no selection at all (Section 6.4,
Table 4).

WHY
---
The fitted threshold reduces 408,654 3-gram features to some thousands. A
comparison of one selector against another does not establish what the
reduction itself costs; for that the classifier must also be trained on the
UNREDUCED matrix.
 For each order it trains the same LightGBM on the full
vocabulary and on the power-law-retained subset, over identical folds, and
compares the paired predictions with McNemar. Three outcomes are possible and
all three are worth reporting:

  full ~= reduced   the reduction is free
  full >  reduced   the reduction costs accuracy
  full <  reduced   the discarded columns were hurting

Orders are run smallest first so that a partial result is still useful if the
largest order does not finish. n = 2 is 37,291 features and quick; n = 3 is
408,654 and is the main order; n = 4 is 1,474,526 and was not attempted.

Usage
-----
    python 12_no_selection_baseline.py --manifest ../results/malware/manifest.json \
        --labels /path/to/trainLabels.csv --orders 2 3 --out-dir ../results/malware
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

_spec = importlib.util.spec_from_file_location(
    "bl", Path(__file__).resolve().parent / "05_baseline_comparison.py")
bl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bl)

SEED = 42


def run_order(X, y, n, out_dir):
    from lightgbm import LGBMClassifier

    def clf():
        return LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                              verbosity=-1, random_state=SEED, n_jobs=-1)

    enc = LabelEncoder().fit(y)
    y_enc = enc.transform(y)
    names = [bl.FAMILY[c] for c in enc.classes_]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    preds = {m: np.zeros_like(y_enc) for m in ("full", "power-law")}
    secs = {m: 0.0 for m in preds}
    ks = {"full": [], "power-law": []}

    for fold, (tr, te) in enumerate(skf.split(X, y_enc), 1):
        X_tr, X_te, y_tr = X[tr], X[te], y_enc[tr]

        # --- no selection -------------------------------------------------
        t0 = time.time()
        c = clf()
        c.fit(X_tr, y_tr)
        preds["full"][te] = c.predict(X_te)
        secs["full"] += time.time() - t0
        ks["full"].append(int(X.shape[1]))
        print(f"  n={n} fold {fold}: full {X.shape[1]:,} feats "
              f"({secs['full']/60:.1f} min cumulative)", flush=True)

        # --- power-law threshold, identical fold ---------------------------
        t0 = time.time()
        idx, p = bl.select_powerlaw(X_tr, y_tr)
        c = clf()
        c.fit(X_tr[:, idx], y_tr)
        preds["power-law"][te] = c.predict(X_te[:, idx])
        secs["power-law"] += time.time() - t0
        ks["power-law"].append(int(len(idx)))
        print(f"  n={n} fold {fold}: power-law {len(idx):,} feats "
              f"(xmin={p['xmin']:.0f}, {secs['power-law']/60:.1f} min cumulative)",
              flush=True)

    res = {"n": n, "n_samples": int(X.shape[0]),
           "full_vocabulary": int(X.shape[1]), "methods": {}}
    for m in preds:
        res["methods"][m] = {
            "accuracy": round(100 * accuracy_score(y_enc, preds[m]), 2),
            "macro_f1": round(100 * f1_score(y_enc, preds[m], average="macro",
                                             zero_division=0), 2),
            "weighted_f1": round(100 * f1_score(y_enc, preds[m],
                                                average="weighted",
                                                zero_division=0), 2),
            "k_per_fold": ks[m],
            "seconds_total": round(secs[m], 1),
            "per_family_f1": {
                names[i]: round(100 * float(v), 2)
                for i, v in enumerate(f1_score(y_enc, preds[m], average=None,
                                               labels=range(len(names)),
                                               zero_division=0))},
        }
    res["mcnemar_full_vs_powerlaw"] = bl.mcnemar(preds["full"],
                                                 preds["power-law"], y_enc)

    print(f"\n  n={n}  {'method':<12}{'acc':>9}{'macro F1':>11}{'sec':>10}")
    for m in ("full", "power-law"):
        v = res["methods"][m]
        print(f"        {m:<12}{v['accuracy']:>8.2f}%{v['macro_f1']:>10.2f}%"
              f"{v['seconds_total']:>10,.0f}")
    mc = res["mcnemar_full_vs_powerlaw"]
    sig = "significant" if mc["p"] < 0.05 else "not significant"
    print(f"        McNemar full vs power-law: p = {mc['p']:.4g} ({sig})")
    d_acc = res["methods"]["full"]["accuracy"] - res["methods"]["power-law"]["accuracy"]
    print(f"        reduction costs {d_acc:+.2f} points of accuracy "
          f"for a {X.shape[1]/np.mean(ks['power-law']):.1f}x cut")

    out = Path(out_dir) / f"no_selection_{n}gram.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"  written to {out}\n", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in sorted(args.orders):
        X, names = bl.build_full_matrix(args.manifest, n)
        y = bl.load_labels(args.labels, names)
        print(f"Order n={n}: matrix {X.shape[0]:,} x {X.shape[1]:,}", flush=True)
        run_order(X, y, n, out_dir)
        del X
