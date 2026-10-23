"""
Step 3 -- classification on the power-law-retained features.

Reproduces Table 3 of the article: five classifiers x three n-gram orders,
stratified five-fold cross-validation, weighted accuracy / precision /
recall / F1.  Hyper-parameters are library defaults except for the values
stated in Section 5.2; no search is performed.

Usage
-----
    python 03_classify.py --results-dir ../results/malware \
                          --labels /path/to/trainLabels.csv \
                          --orders 2 3 4

`--labels` must be the BIG 2015 label CSV with columns Id,Class.  Rows are
aligned to the manifest order, so pass the same manifest used in step 2.

Output
------
    <results-dir>/classification_results.json    every number in Table 3
    <results-dir>/per_family_<n>gram_<clf>.json  per-family report
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

FAMILY_NAMES = {
    1: "Ramnit", 2: "Lollipop", 3: "Kelihos_ver3", 4: "Vundo", 5: "Simda",
    6: "Tracur", 7: "Kelihos_ver1", 8: "Obfuscator.ACY", 9: "Gatak",
}

SEED = 42


def make_classifiers():
    """The five classifiers of Table 3, with the settings of Section 5.2."""
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "KNN(k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "LinearSVM": LinearSVC(C=1.0, max_iter=5000, random_state=SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            eval_metric="mlogloss", verbosity=0, random_state=SEED, n_jobs=-1),
        "LightGBM": LGBMClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            verbosity=-1, random_state=SEED, n_jobs=-1),
    }


def load_labels(labels_csv, manifest_path):
    """Return the label vector in manifest order."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    order = [rec["name"] for rec in manifest]
    lbls = pd.read_csv(labels_csv)
    id_col = "Id" if "Id" in lbls.columns else "ID"
    lbls[id_col] = lbls[id_col].astype(str)
    lbls = lbls.set_index(id_col)
    missing = [i for i in order if i not in lbls.index]
    if missing:
        raise SystemExit(f"{len(missing)} manifest ids absent from {labels_csv}, "
                         f"first: {missing[:3]}")
    return lbls.loc[order, "Class"].to_numpy()


def evaluate(X, y, n, out_dir, skip=()):
    y_enc = LabelEncoder().fit_transform(y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    table, preds_by_clf = {}, {}

    print(f"\n=== {n}-gram, {X.shape[0]:,} samples x {X.shape[1]:,} features ===")
    for name, clf in make_classifiers().items():
        if name in skip:
            print(f'  {name:<14} SKIPPED (requested)')
            continue
        t0 = time.time()
        preds = cross_val_predict(clf, X, y_enc, cv=skf)
        table[name] = {
            "accuracy": round(100 * accuracy_score(y_enc, preds), 2),
            "precision": round(100 * precision_score(
                y_enc, preds, average="weighted", zero_division=0), 2),
            "recall": round(100 * recall_score(
                y_enc, preds, average="weighted", zero_division=0), 2),
            "f1": round(100 * f1_score(
                y_enc, preds, average="weighted", zero_division=0), 2),
            "fit_predict_seconds": round(time.time() - t0, 1),
        }
        preds_by_clf[name] = preds
        m = table[name]
        print(f"  {name:<14} acc={m['accuracy']:6.2f}  P={m['precision']:6.2f}"
              f"  R={m['recall']:6.2f}  F1={m['f1']:6.2f}"
              f"  ({m['fit_predict_seconds']}s)")

    # Per-family breakdown for the best classifier by accuracy.
    best = max(table, key=lambda k: table[k]["accuracy"])
    report = classification_report(
        y_enc, preds_by_clf[best],
        target_names=[FAMILY_NAMES[c] for c in sorted(set(y))],
        output_dict=True, zero_division=0)
    cm = confusion_matrix(y_enc, preds_by_clf[best]).tolist()
    (out_dir / f"per_family_{n}gram_{best}.json").write_text(
        json.dumps({"classifier": best, "report": report,
                    "confusion_matrix": cm}, indent=2), encoding="utf-8")
    print(f"  per-family report written for best classifier: {best}")

    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 3, 4])
    # Omit a classifier only when it is computationally infeasible at that
    # width -- and report the omission. Linear SVM took 7.7 h at n=3 on 7,144
    # features; at the 150,808 features retained for n=4 it is impractical.
    ap.add_argument("--skip", nargs="*", default=[],
                    help="classifier names to omit, e.g. LinearSVM")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    manifest = args.manifest or str(results_dir / "manifest.json")
    y = load_labels(args.labels, manifest)

    # Merge into any existing results rather than overwriting: running with
    # --orders 4 must not discard the n=2 and n=3 rows of Table 3.
    out = results_dir / "classification_results.json"
    all_results = (json.loads(out.read_text(encoding="utf-8"))
                   if out.exists() else {})
    for n in args.orders:
        npz = results_dir / f"opcode_{n}gram_reduced.npz"
        if not npz.exists():
            print(f"skipping n={n}: {npz.name} not found (run step 2 first)")
            continue
        # LightGBM >= 4 rejects integer sparse data; CountVectorizer emits
        # int64. float32 is exact for these counts and halves the memory.
        X = load_npz(str(npz)).astype('float32')
        if X.shape[0] != len(y):
            raise SystemExit(f"n={n}: matrix has {X.shape[0]} rows but "
                             f"{len(y)} labels were loaded")
        all_results[str(n)] = {
            "n_features": int(X.shape[1]),
            "classifiers": evaluate(X, y, n, results_dir, set(args.skip)),
            "skipped": list(args.skip),
        }

    all_results = {k: all_results[k] for k in sorted(all_results)}
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    # "Table 2" in the message below is Table 3 of the article.
    print(f"\nTable 3 numbers written to {out}")
