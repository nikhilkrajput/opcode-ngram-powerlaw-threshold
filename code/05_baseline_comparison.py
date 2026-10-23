"""
Step 5 -- the selector comparison (Section 6.5, Tables 5 and 6).

Compares the power-law threshold against the selection rules it claims to
replace, at matched dimensionality:

    power-law   f_g >= xmin-hat        unsupervised, parameter-free
    chi-squared top-k by chi2(X, y)    supervised, k must be chosen
    mutual info top-k by MI(X, y)      supervised, k must be chosen
    top-K freq  top-k by f_g           unsupervised, k must be chosen
    top-5000    top-5000 by f_g        the Chen et al. (2021) rule

DESIGN
------
Two pitfalls make such a comparison meaningless, and both are avoided here.

1. VACUOUS SELECTION.  Selecting the top k features by chi2 or MI from a
   matrix that has ALREADY been reduced to the k power-law columns returns
   every column: the "baselines" are then the power-law feature set itself,
   reordered, and any comparison is guaranteed to come out identical.  The
   baselines must select from the FULL vocabulary, which is what this script
   does -- so it needs the unreduced count matrix, not the reduced one.

2. LABEL LEAKAGE.  chi2 and mutual_info_classif are supervised.  Fitting them
   on the whole dataset and then cross-validating leaks test-fold labels into
   feature selection and inflates the baselines.  Here every selector is
   refitted inside each training fold; nothing sees the held-out labels.

The power-law selector is unsupervised, so it is immune to point 2 by
construction and can be fitted on unlabelled corpora.

Matched dimensionality: the power-law fit on each training fold yields k_fold
retained features; every other selector is given exactly k_fold on that fold.
The comparison is therefore like-for-like within each fold.

Usage
-----
    python 05_baseline_comparison.py --manifest ../results/malware/manifest.json \
                                     --labels /path/to/trainLabels.csv \
                                     --n 3 --out-dir ../results/malware

Output
------
    <out-dir>/selector_comparison_<n>gram.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import powerlaw
from scipy.stats import binom
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

FAMILY = {1: "Ramnit", 2: "Lollipop", 3: "Kelihos_ver3", 4: "Vundo",
          5: "Simda", 6: "Tracur", 7: "Kelihos_ver1",
          8: "Obfuscator.ACY", 9: "Gatak"}

SEED = 42
TOKEN_PATTERN = r"(?u)\b[a-z]{2,}\b"
CHEN_K = 5000  # the fixed cutoff of Chen et al. (2021)


# ---------------------------------------------------------------------------
# Selectors.  Each takes the training fold and returns column indices.
# ---------------------------------------------------------------------------
def select_powerlaw(X_tr, y_tr, k=None):
    """Unsupervised. k is determined by the fit, not supplied."""
    freqs = np.asarray(X_tr.sum(axis=0)).ravel()
    nz = freqs[freqs > 0]
    fit = powerlaw.Fit(nz, verbose=False)
    idx = np.where(freqs >= fit.xmin)[0]
    return idx, {"xmin": float(fit.xmin), "alpha": float(fit.alpha)}


def select_chi2(X_tr, y_tr, k):
    scores, _ = chi2(X_tr, y_tr)
    scores = np.nan_to_num(scores, nan=-np.inf)
    return np.argsort(scores)[::-1][:k], {}


def select_mutual_info(X_tr, y_tr, k):
    scores = mutual_info_classif(X_tr, y_tr, random_state=SEED)
    return np.argsort(scores)[::-1][:k], {}


def select_top_freq(X_tr, y_tr, k):
    freqs = np.asarray(X_tr.sum(axis=0)).ravel()
    return np.argsort(freqs)[::-1][:k], {}


# ---------------------------------------------------------------------------
def mcnemar(pred_a, pred_b, y_true):
    """Exact two-sided (binomial) McNemar test between two prediction vectors."""
    a_ok, b_ok = (pred_a == y_true), (pred_b == y_true)
    b = int(np.sum(~a_ok & b_ok))
    c = int(np.sum(a_ok & ~b_ok))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0, "chi2": 0.0}
    p = float(2 * binom.cdf(min(b, c), n, 0.5))
    chi_stat = float((abs(b - c) - 1) ** 2 / n)
    return {"b": b, "c": c, "p": min(p, 1.0), "chi2": chi_stat}


def build_full_matrix(manifest_path, n):
    """Full, UNREDUCED n-gram count matrix -- the baselines need it."""
    from sklearn.feature_extraction.text import CountVectorizer

    records = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    seqs = [Path(r["seq_file"]).read_text(encoding="utf-8").strip()
            for r in records]
    names = [r["name"] for r in records]
    cv = CountVectorizer(ngram_range=(n, n), lowercase=False,
                         token_pattern=TOKEN_PATTERN)
    # chi2 needs the raw counts, but LightGBM >= 4 rejects integer sparse
    # input; float32 is exact at these magnitudes.
    X = cv.fit_transform(seqs).astype('float32')
    print(f"Full {n}-gram matrix: {X.shape[0]:,} x {X.shape[1]:,}")
    return X, names


def load_labels(labels_csv, names):
    lbls = pd.read_csv(labels_csv)
    id_col = "Id" if "Id" in lbls.columns else "ID"
    lbls[id_col] = lbls[id_col].astype(str)
    lbls = lbls.set_index(id_col)
    missing = [i for i in names if i not in lbls.index]
    if missing:
        raise SystemExit(f"{len(missing)} manifest ids absent from labels, "
                         f"first: {missing[:3]}")
    return lbls.loc[names, "Class"].to_numpy()


def run(X, y, n, out_dir, classifier_name="LightGBM"):
    from lightgbm import LGBMClassifier

    def make_clf():
        return LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                              verbosity=-1, random_state=SEED, n_jobs=-1)

    y_enc = LabelEncoder().fit_transform(y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    methods = ["power-law", "chi2", "mutual-info", "top-k-freq",
               f"top-{CHEN_K}-freq"]
    preds = {m: np.zeros_like(y_enc) for m in methods}
    sel_time = {m: 0.0 for m in methods}
    k_per_fold, fit_params = [], []

    for fold, (tr, te) in enumerate(skf.split(X, y_enc), 1):
        X_tr, X_te = X[tr], X[te]
        y_tr = y_enc[tr]

        # Power-law first: it decides k for this fold.
        t0 = time.time()
        idx_pl, params = select_powerlaw(X_tr, y_tr)
        sel_time["power-law"] += time.time() - t0
        k = len(idx_pl)
        k_per_fold.append(k)
        fit_params.append(params)
        print(f"  fold {fold}: power-law kept {k:,} features "
              f"(xmin={params['xmin']:.0f}, alpha={params['alpha']:.3f})")

        fold_selectors = {
            "power-law": (lambda *_: (idx_pl, {})),
            "chi2": select_chi2,
            "mutual-info": select_mutual_info,
            "top-k-freq": select_top_freq,
            f"top-{CHEN_K}-freq": lambda a, b, _k: select_top_freq(a, b, CHEN_K),
        }

        for method, fn in fold_selectors.items():
            t0 = time.time()
            idx, _ = fn(X_tr, y_tr, k)
            if method != "power-law":
                sel_time[method] += time.time() - t0
            clf = make_clf()
            clf.fit(X_tr[:, idx], y_tr)
            preds[method][te] = clf.predict(X_te[:, idx])

    results = {
        "n": n,
        "classifier": classifier_name,
        "n_samples": int(X.shape[0]),
        "full_vocabulary": int(X.shape[1]),
        "k_per_fold": k_per_fold,
        "k_mean": float(np.mean(k_per_fold)),
        "powerlaw_fit_per_fold": fit_params,
        "methods": {},
        "mcnemar_vs_powerlaw": {},
    }

    # Macro F1 and per-family F1 are recorded for EVERY selector (Tables 5
    # and 6), because accuracy on BIG 2015 saturates near 99.5% for any
    # sensible feature set; macro F1 weights the 42-sample Simda family
    # equally with the 2,938-sample Kelihos_ver3.
    classes = [FAMILY[c] for c in sorted(set(y))]
    print('')
    print(f"  {'method':<18}{'acc':>8}{'macro F1':>10}{'wtd F1':>9}{'sel (s)':>10}")
    for m in methods:
        acc = 100 * accuracy_score(y_enc, preds[m])
        f1w = 100 * f1_score(y_enc, preds[m], average='weighted',
                             zero_division=0)
        f1m = 100 * f1_score(y_enc, preds[m], average='macro',
                             zero_division=0)
        per_fam = f1_score(y_enc, preds[m], average=None,
                           labels=range(len(classes)), zero_division=0)
        results['methods'][m] = {
            'accuracy': round(acc, 2),
            'macro_f1': round(f1m, 2),
            'weighted_f1': round(f1w, 2),
            'per_family_f1': {classes[i]: round(100 * float(v), 2)
                              for i, v in enumerate(per_fam)},
            'selection_seconds_total': round(sel_time[m], 1),
            'supervised': m in ('chi2', 'mutual-info'),
            'k_free_parameter': m != 'power-law',
        }
        print(f'  {m:<18}{acc:7.2f}%{f1m:9.2f}%{f1w:8.2f}%{sel_time[m]:9.1f}')

    for m in methods:
        if m != "power-law":
            results["mcnemar_vs_powerlaw"][m] = mcnemar(
                preds["power-law"], preds[m], y_enc)
            r = results["mcnemar_vs_powerlaw"][m]
            verdict = "significant" if r["p"] < 0.05 else "not significant"
            print(f"  McNemar power-law vs {m:<16} p = {r['p']:.4g} ({verdict})")

    out = Path(out_dir) / f"selector_comparison_{n}gram.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWritten to {out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True,
                    help="malware manifest from 01_extract_opcodes.py")
    ap.add_argument("--labels", required=True, help="BIG 2015 trainLabels.csv")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    X, names = build_full_matrix(args.manifest, args.n)
    y = load_labels(args.labels, names)
    run(X, y, args.n, out_dir)
