"""
Step 11 -- partially labelled training sets, repeated draws (Section 8.2,
Table 10).

WHY
---
Labels are kept on only a fraction rho of each training fold. A single
labelled-subset draw per rho carries no error bar, and the differences between
selectors turn out to be smaller than the spread across draws. This script
therefore repeats each rho over REPEATS independent labelled draws and reports
mean, spread and a paired test.

DESIGN
------

  power-law   fitted on ALL training documents, once per fold. Independent of
              rho AND of the draw, so it is reused across every repeat.
  chi-squared recomputed on each labelled draw, at the power-law's k
  top-5,000   label-free fixed-size control, also once per fold

The classifier is trained on the labelled subset alone in every case and the
classifier seed is held FIXED across repeats, so the only thing varying within
a (method, rho) cell is which documents kept their labels.

rho = 1.0 has no draw to vary -- the labelled set is the whole training fold --
so it is run once and its repeats would be bit-identical.

WHAT COUNTS AS SUPPORT
----------------------
Reported per rho:
  - mean +/- sd macro F1 over the repeats
  - paired t-test and Wilcoxon over the repeats, power-law vs chi-squared
  - how many repeats the threshold wins

A low-label advantage for the threshold is established only if its lead at
low rho is larger than the spread across draws. Section 8.2 reports that it is
not. (The closing VERDICT line printed by the script refers to an earlier
section number; the relevant section of the article is 8.2.)

Usage
-----
    python 11_unlabelled_repeats.py --manifest ../results/malware/manifest.json \
        --labels /path/to/trainLabels.csv --n 3 --out-dir ../results/malware
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

_spec = importlib.util.spec_from_file_location(
    "bl", Path(__file__).resolve().parent / "05_baseline_comparison.py")
bl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bl)

SEED = 42
RHOS = [0.05, 0.10, 0.25, 0.50, 1.00]
REPEATS = 5
CHEN_K = 5000
METHODS = ["power-law", "chi2", "top-5000"]


def n_reps(rho):
    """rho = 1.0 has no random draw, so repeating it would be bit-identical."""
    return 1 if rho >= 1.0 else REPEATS


def run(X, y, n, out_dir):
    from lightgbm import LGBMClassifier

    def clf():
        return LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                              verbosity=-1, random_state=SEED, n_jobs=-1)

    enc = LabelEncoder().fit(y)
    y_enc = enc.transform(y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    cells = [(m, r, rep) for m in METHODS for r in RHOS
             for rep in range(n_reps(r))]
    preds = {c: np.zeros_like(y_enc) for c in cells}
    t0 = time.time()

    for fold, (tr, te) in enumerate(skf.split(X, y_enc), 1):
        X_tr, X_te, y_tr = X[tr], X[te], y_enc[tr]

        idx_pl, p_pl = bl.select_powerlaw(X_tr, y_tr)
        k = len(idx_pl)
        idx_top, _ = bl.select_top_freq(X_tr, y_tr, CHEN_K)
        print("  fold {}: power-law k={:,} (xmin={:.0f})".format(
            fold, k, p_pl["xmin"]), flush=True)

        for rho in RHOS:
            for rep in range(n_reps(rho)):
                if rho >= 1.0:
                    lab = np.arange(len(y_tr))
                else:
                    lab, _ = train_test_split(
                        np.arange(len(y_tr)), train_size=rho, stratify=y_tr,
                        random_state=SEED + 1000 * rep)
                y_lab = y_tr[lab]
                idx_chi, _ = bl.select_chi2(X_tr[lab], y_lab, k)

                for m, idx in (("power-law", idx_pl), ("chi2", idx_chi),
                               ("top-5000", idx_top)):
                    c = clf()
                    c.fit(X_tr[lab][:, idx], y_lab)
                    preds[(m, rho, rep)][te] = c.predict(X_te[:, idx])
                print("    rho={:<5} rep {}/{}  labelled={:>5,}  ({:.1f} min)"
                      .format(rho, rep + 1, n_reps(rho), len(lab),
                              (time.time() - t0) / 60), flush=True)

    # ---- scoring -----------------------------------------------------------
    def macro(c):
        return 100 * f1_score(y_enc, preds[c], average="macro", zero_division=0)

    def acc(c):
        return 100 * accuracy_score(y_enc, preds[c])

    res = {"n": n, "rhos": RHOS, "repeats": REPEATS,
           "n_samples": int(X.shape[0]), "full_vocabulary": int(X.shape[1]),
           "note": "classifier seed fixed across repeats; only the labelled "
                   "draw varies. rho=1.0 has a single deterministic draw.",
           "per_repeat": {}, "summary": {}, "paired_tests": {}}

    for rho in RHOS:
        key = str(rho)
        res["per_repeat"][key] = {
            m: {"macro_f1": [round(macro((m, rho, r)), 2)
                             for r in range(n_reps(rho))],
                "accuracy": [round(acc((m, rho, r)), 2)
                             for r in range(n_reps(rho))]}
            for m in METHODS}
        res["summary"][key] = {}
        for m in METHODS:
            v = np.array(res["per_repeat"][key][m]["macro_f1"], dtype=float)
            a = np.array(res["per_repeat"][key][m]["accuracy"], dtype=float)
            res["summary"][key][m] = {
                "macro_f1_mean": round(float(v.mean()), 2),
                "macro_f1_sd": round(
                    float(v.std(ddof=1)) if v.size > 1 else 0.0, 2),
                "accuracy_mean": round(float(a.mean()), 2),
                "accuracy_sd": round(
                    float(a.std(ddof=1)) if a.size > 1 else 0.0, 2),
            }

    print("\n  macro F1, mean +/- sd over {} labelled draws".format(REPEATS))
    print("  {:>6}".format("rho") + "".join("{:>20}".format(m)
                                            for m in METHODS))
    for rho in RHOS:
        s = res["summary"][str(rho)]
        row = "".join("{:14.2f} +-{:4.2f}".format(s[m]["macro_f1_mean"],
                                                  s[m]["macro_f1_sd"])
                      for m in METHODS)
        print("  {:>6}".format(rho) + row)

    # ---- paired tests: threshold vs chi-squared, over draws ----------------
    print("\n  power-law vs chi-squared, paired over draws")
    for rho in RHOS:
        key = str(rho)
        pl = np.array(res["per_repeat"][key]["power-law"]["macro_f1"], float)
        ch = np.array(res["per_repeat"][key]["chi2"]["macro_f1"], float)
        d = pl - ch
        entry = {"mean_difference": round(float(d.mean()), 3),
                 "threshold_wins": int((d > 0).sum()), "n_draws": int(d.size)}
        if d.size > 1 and np.ptp(d) > 0:
            entry["t_test_p"] = round(float(stats.ttest_rel(pl, ch).pvalue), 4)
            try:
                entry["wilcoxon_p"] = round(
                    float(stats.wilcoxon(pl, ch).pvalue), 4)
            except ValueError:
                entry["wilcoxon_p"] = None
        else:
            entry["t_test_p"] = None
            entry["wilcoxon_p"] = None
            entry["note"] = "single deterministic draw; no test possible"
        res["paired_tests"][key] = entry
        p = entry["t_test_p"]
        ptxt = "n/a  (single draw)" if p is None else "t-test p = {:.4f}".format(p)
        print("    rho={:<5} mean diff {:+.3f}  threshold wins {}/{}  {}"
              .format(rho, entry["mean_difference"], entry["threshold_wins"],
                      entry["n_draws"], ptxt))

    out = Path(out_dir) / "unlabelled_repeats_{}gram.json".format(n)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nWritten to {}".format(out))

    lo = str(RHOS[0])
    e = res["paired_tests"][lo]
    sd = max(res["summary"][lo]["power-law"]["macro_f1_sd"],
             res["summary"][lo]["chi2"]["macro_f1_sd"])
    print("\n  VERDICT at rho={}: lead {:+.3f} points, largest sd {:.2f}"
          .format(lo, e["mean_difference"], sd))
    print("  The low-label advantage is established only if the lead exceeds "
          "the spread\n  AND the paired test agrees. Otherwise the claim "
          "must be softened.")
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
