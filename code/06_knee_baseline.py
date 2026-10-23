"""
Step 6 -- knee detection: another parameter-free choice of k (knee row of
Table 5, Section 6.5).

WHY
---
The selectors of 05_baseline_comparison.py are either supervised (and need k)
or a fixed cutoff (arbitrary by construction). None of them chooses k
automatically, so on their own they can only show that "a data-chosen cutoff
beats an arbitrary one". The natural question is whether any *other*
automatic rule does as well.

This script answers it with knee detection, the standard parameter-free way to
find a cutoff on a decaying curve. Sort the corpus frequencies in descending
order, take logs of both rank and frequency, normalise each to [0, 1], and cut
at the point furthest from the chord joining the first and last points. No
constant enters, exactly as for x_min.

It also reports per-family F1 for the two smallest families. Aggregate accuracy
on BIG 2015 saturates near 99.5% for any sensible feature set, so the headline
numbers cannot separate selectors. Simda (42 samples) and Kelihos_ver1 (398)
are where feature quality actually shows.

Usage
-----
    python 06_knee_baseline.py --manifest ../results/malware/manifest.json \
                               --labels /path/to/trainLabels.csv \
                               --n 3 --out-dir ../results/malware

Output
------
    <out-dir>/knee_comparison_<n>gram.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "bl", Path(__file__).resolve().parent / "05_baseline_comparison.py")
bl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bl)

SEED = 42
CHEN_K = 5000
FAMILY = {1: "Ramnit", 2: "Lollipop", 3: "Kelihos_ver3", 4: "Vundo",
          5: "Simda", 6: "Tracur", 7: "Kelihos_ver1", 8: "Obfuscator.ACY",
          9: "Gatak"}
SMALL = ["Simda", "Kelihos_ver1", "Vundo"]


def select_knee(X_tr, y_tr, k=None):
    """Parameter-free cutoff by maximum distance from the log-log chord.

    Standard elbow/knee detection (Satopaa et al. style), with no tuning
    constant, which is what makes it a fair comparison for x_min.
    """
    freqs = np.asarray(X_tr.sum(axis=0)).ravel()
    nz = np.sort(freqs[freqs > 0])[::-1]

    x = np.log(np.arange(1, len(nz) + 1, dtype=float))
    y = np.log(nz.astype(float))
    x = (x - x[0]) / (x[-1] - x[0])
    y = (y - y[-1]) / (y[0] - y[-1])

    # perpendicular distance to the chord from (0,1) to (1,0) is |x + y - 1|/sqrt2
    idx_knee = int(np.argmax(np.abs(x + y - 1.0)))
    cutoff = nz[idx_knee]
    keep = np.where(freqs >= cutoff)[0]
    return keep, {"cutoff": float(cutoff), "k": int(len(keep))}


def per_family_f1(y_true, y_pred, classes):
    f1s = f1_score(y_true, y_pred, average=None, labels=range(len(classes)),
                   zero_division=0)
    return {classes[i]: round(100 * float(v), 2) for i, v in enumerate(f1s)}


def run(X, y, n, out_dir):
    from lightgbm import LGBMClassifier

    def clf():
        return LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                              verbosity=-1, random_state=SEED, n_jobs=-1)

    enc = LabelEncoder().fit(y)
    y_enc = enc.transform(y)
    classes = [FAMILY[c] for c in enc.classes_]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    methods = ["power-law", "knee", f"top-{CHEN_K}-freq"]
    preds = {m: np.zeros_like(y_enc) for m in methods}
    ks = {m: [] for m in methods}
    sel_s = {m: 0.0 for m in methods}

    for fold, (tr, te) in enumerate(skf.split(X, y_enc), 1):
        X_tr, X_te, y_tr = X[tr], X[te], y_enc[tr]

        t0 = time.time()
        idx_pl, p_pl = bl.select_powerlaw(X_tr, y_tr)
        sel_s["power-law"] += time.time() - t0

        t0 = time.time()
        idx_kn, p_kn = select_knee(X_tr, y_tr)
        sel_s["knee"] += time.time() - t0

        t0 = time.time()
        idx_tk, _ = bl.select_top_freq(X_tr, y_tr, CHEN_K)
        sel_s[f"top-{CHEN_K}-freq"] += time.time() - t0

        print(f"  fold {fold}: power-law k={len(idx_pl):,} "
              f"(xmin={p_pl['xmin']:.0f})   knee k={len(idx_kn):,} "
              f"(cutoff={p_kn['cutoff']:.0f})", flush=True)

        for m, idx in (("power-law", idx_pl), ("knee", idx_kn),
                       (f"top-{CHEN_K}-freq", idx_tk)):
            ks[m].append(int(len(idx)))
            c = clf()
            c.fit(X_tr[:, idx], y_tr)
            preds[m][te] = c.predict(X_te[:, idx])

    res = {"n": n, "n_samples": int(X.shape[0]),
           "full_vocabulary": int(X.shape[1]), "methods": {}}

    print(f"\n  {'method':<18}{'acc':>8}{'macro F1':>10}   small-family F1")
    for m in methods:
        acc = 100 * accuracy_score(y_enc, preds[m])
        macro = 100 * f1_score(y_enc, preds[m], average="macro", zero_division=0)
        fam = per_family_f1(y_enc, preds[m], classes)
        res["methods"][m] = {
            "accuracy": round(acc, 2),
            "macro_f1": round(macro, 2),
            "k_per_fold": ks[m],
            "k_mean": round(float(np.mean(ks[m])), 1),
            "selection_seconds": round(sel_s[m], 1),
            "per_family_f1": fam,
        }
        small = "  ".join(f"{s}={fam[s]:.1f}" for s in SMALL if s in fam)
        print(f"  {m:<18}{acc:7.2f}%{macro:9.2f}%   {small}")

    res["mcnemar_vs_powerlaw"] = {}
    for m in methods:
        if m != "power-law":
            r = bl.mcnemar(preds["power-law"], preds[m], y_enc)
            res["mcnemar_vs_powerlaw"][m] = r
            sig = "significant" if r["p"] < 0.05 else "not significant"
            print(f"  McNemar power-law vs {m:<16} p = {r['p']:.4g} ({sig})")

    out = Path(out_dir) / f"knee_comparison_{n}gram.json"
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
