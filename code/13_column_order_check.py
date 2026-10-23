"""
Step 13 -- why do 'power-law' and 'top-k-freq' score differently in Table 5?
(Section 6.5)

In 05_baseline_comparison.py the power-law selector returns
    idx = np.where(freqs >= xmin)[0]            (ascending column index)
and top-k-freq, given k = len(idx), returns
    np.argsort(freqs)[::-1][:k]                  (descending frequency)
Every column with f >= xmin outranks every column with f < xmin, so the two
index lists hold the SAME set; only the column order differs. This script
(a) checks the set identity on every fold, and (b) refits LightGBM on the
power-law set in both column orders with everything else fixed, to measure how
much of the Table 5 gap is LightGBM's sensitivity to column order alone.

Usage
-----
    python 13_column_order_check.py --manifest ../results/malware/manifest.json \
        --labels /path/to/trainLabels.csv --out-dir ../results/malware

Output
------
    <out-dir>/column_order_3gram.json
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

_spec = importlib.util.spec_from_file_location(
    "bc", Path(__file__).with_name("05_baseline_comparison.py"))
bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bc)


def main():
    from lightgbm import LGBMClassifier

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    X, names = bc.build_full_matrix(args.manifest, args.n)
    y = bc.load_labels(args.labels, names)
    y_enc = LabelEncoder().fit_transform(y)
    classes = [bc.FAMILY[c] for c in sorted(set(y))]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=bc.SEED)

    orders = ["index-order", "frequency-order"]
    preds = {o: np.zeros_like(y_enc) for o in orders}
    folds = []
    # Per-fold checkpoint: on a loaded machine the run can be killed, and each
    # fold costs tens of minutes. Completed folds are reloaded, not refitted.
    ckpt = Path(args.out_dir) / f"column_order_{args.n}gram.ckpt.npz"
    done = dict(np.load(ckpt, allow_pickle=True)) if ckpt.exists() else {}
    for fold, (tr, te) in enumerate(skf.split(X, y_enc), 1):
        if f"f{fold}_meta" in done:
            for o in orders:
                preds[o][te] = done[f"f{fold}_{o}"]
            folds.append(json.loads(str(done[f"f{fold}_meta"])))
            print(f"  fold {fold}: restored from checkpoint", flush=True)
            continue
        X_tr, X_te, y_tr = X[tr], X[te], y_enc[tr]
        idx_pl, params = bc.select_powerlaw(X_tr, y_tr)
        idx_top, _ = bc.select_top_freq(X_tr, y_tr, len(idx_pl))
        same = bool(set(idx_pl.tolist()) == set(idx_top.tolist()))
        folds.append({"xmin": params["xmin"], "k": int(len(idx_pl)),
                      "same_set": same})
        print(f"  fold {fold}: xmin={params['xmin']:.0f} k={len(idx_pl):,} "
              f"same set: {same}", flush=True)
        for o, idx in zip(orders, [idx_pl, idx_top]):
            clf = LGBMClassifier(n_estimators=100, learning_rate=0.1,
                                 max_depth=6, verbosity=-1,
                                 random_state=bc.SEED, n_jobs=-1)
            clf.fit(X_tr[:, idx], y_tr)
            preds[o][te] = clf.predict(X_te[:, idx])
        for o in orders:
            done[f"f{fold}_{o}"] = preds[o][te]
        done[f"f{fold}_meta"] = json.dumps(folds[-1])
        np.savez(ckpt, **done)

    out = {"n": args.n, "folds": folds, "methods": {}}
    for o in orders:
        per = f1_score(y_enc, preds[o], average=None,
                       labels=range(len(classes)), zero_division=0)
        out["methods"][o] = {
            "accuracy": round(100 * accuracy_score(y_enc, preds[o]), 2),
            "macro_f1": round(100 * f1_score(y_enc, preds[o], average="macro",
                                             zero_division=0), 2),
            "per_family_f1": {classes[i]: round(100 * float(v), 2)
                              for i, v in enumerate(per)},
        }
        print(o, out["methods"][o])
    out["mcnemar"] = bc.mcnemar(preds[orders[0]], preds[orders[1]], y_enc)
    print("McNemar", out["mcnemar"])
    p = Path(args.out_dir) / f"column_order_{args.n}gram.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Written to", p)


if __name__ == "__main__":
    main()
