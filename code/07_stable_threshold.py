"""
Step 7 -- a candidate remedy for the threshold instability, measured
(Section 6.7).

WHY
---
On one of five training folds the CSN fit lands on a second KS minimum:
x_min = 9 instead of ~780, retaining 65,250 features instead of ~6,500. That
is a one-in-five chance of retaining an order of magnitude more than
intended, and it is the most serious objection to a method meant to remove
arbitrary choices. Resampling is the obvious mitigation; this script
evaluates it.

THE REMEDY
----------
Fit x_min on B independent document subsamples of the training fold and take
the median. The median is robust to a minority of collapsed fits: with an
observed failure rate near 20%, the probability that three or more of five
resamples collapse is about 6%, so the median would land on the stable mode
nearly always if collapses were rare and independent. Section 6.7 reports
that they are not: the low mode won on most subsamples, and the median of a
few draws from a two-point distribution is itself a draw from it.

This adds one free parameter, B, the number of subsamples. B is a compute
budget rather than a modelling choice, but it is a knob and is reported.

Reported against the single fit on identical folds: accuracy, macro F1, the
per-fold thresholds, and McNemar.

Usage
-----
    python 07_stable_threshold.py --manifest ../results/malware/manifest.json \
                                  --labels /path/to/trainLabels.csv \
                                  --n 3 --B 5 --out-dir ../results/malware
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

import powerlaw  # noqa: E402

SEED = 42
FAMILY = {1: "Ramnit", 2: "Lollipop", 3: "Kelihos_ver3", 4: "Vundo",
          5: "Simda", 6: "Tracur", 7: "Kelihos_ver1", 8: "Obfuscator.ACY",
          9: "Gatak"}


def xmin_single(X_tr):
    freqs = np.asarray(X_tr.sum(axis=0)).ravel()
    fit = powerlaw.Fit(freqs[freqs > 0], verbose=False)
    return float(fit.xmin), freqs


def xmin_median(X_tr, B, rng, frac=0.8):
    """Median x_min over B document subsamples of the training fold."""
    n = X_tr.shape[0]
    size = max(2, int(round(frac * n)))
    xs = []
    for _ in range(B):
        rows = rng.choice(n, size=size, replace=False)
        f = np.asarray(X_tr[rows].sum(axis=0)).ravel()
        nz = f[f > 0]
        if nz.size < 50:
            continue
        xs.append(float(powerlaw.Fit(nz, verbose=False).xmin))
    return float(np.median(xs)), xs


def run(X, y, n, B, out_dir):
    from lightgbm import LGBMClassifier

    def clf():
        return LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                              verbosity=-1, random_state=SEED, n_jobs=-1)

    enc = LabelEncoder().fit(y)
    y_enc = enc.transform(y)
    classes = [FAMILY[c] for c in enc.classes_]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rng = np.random.default_rng(SEED)

    methods = ["single", f"median-of-{B}"]
    preds = {m: np.zeros_like(y_enc) for m in methods}
    info = {m: {"xmin": [], "k": [], "seconds": 0.0} for m in methods}
    draws = []

    for fold, (tr, te) in enumerate(skf.split(X, y_enc), 1):
        X_tr, X_te, y_tr = X[tr], X[te], y_enc[tr]

        t0 = time.time()
        xs_single, freqs = xmin_single(X_tr)
        info["single"]["seconds"] += time.time() - t0

        t0 = time.time()
        xs_med, xs_all = xmin_median(X_tr, B, rng)
        info[f"median-of-{B}"]["seconds"] += time.time() - t0
        draws.append([round(v, 1) for v in xs_all])

        print(f"  fold {fold}: single x_min={xs_single:.0f}   "
              f"median-of-{B} x_min={xs_med:.0f}   draws={[int(v) for v in xs_all]}",
              flush=True)

        for m, xm in (("single", xs_single), (f"median-of-{B}", xs_med)):
            idx = np.where(freqs >= xm)[0]
            info[m]["xmin"].append(xm)
            info[m]["k"].append(int(len(idx)))
            c = clf()
            c.fit(X_tr[:, idx], y_tr)
            preds[m][te] = c.predict(X_te[:, idx])

    res = {"n": n, "B": B, "n_samples": int(X.shape[0]),
           "full_vocabulary": int(X.shape[1]),
           "median_draws_per_fold": draws, "methods": {}}

    print(f"\n  {'variant':<14}{'acc':>8}{'macro F1':>10}{'k spread':>22}")
    for m in methods:
        acc = 100 * accuracy_score(y_enc, preds[m])
        macro = 100 * f1_score(y_enc, preds[m], average="macro", zero_division=0)
        ks = info[m]["k"]
        res["methods"][m] = {
            "accuracy": round(acc, 2), "macro_f1": round(macro, 2),
            "xmin_per_fold": [round(v, 1) for v in info[m]["xmin"]],
            "k_per_fold": ks,
            "k_min": min(ks), "k_max": max(ks),
            "k_ratio": round(max(ks) / max(min(ks), 1), 1),
            "selection_seconds": round(info[m]["seconds"], 1),
        }
        print(f"  {m:<14}{acc:7.2f}%{macro:9.2f}%   "
              f"{min(ks):>7,} to {max(ks):<9,} ({max(ks)/max(min(ks),1):.1f}x)")

    r = bl.mcnemar(preds["single"], preds[f"median-of-{B}"], y_enc)
    res["mcnemar_single_vs_median"] = r
    sig = "significant" if r["p"] < 0.05 else "not significant"
    print(f"\n  McNemar single vs median-of-{B}: p = {r['p']:.4g} ({sig})")

    out = Path(out_dir) / f"stable_threshold_{n}gram.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Written to {out}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--B", type=int, default=5)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    X, names = bl.build_full_matrix(args.manifest, args.n)
    y = bl.load_labels(args.labels, names)
    run(X, y, args.n, args.B, out_dir)
