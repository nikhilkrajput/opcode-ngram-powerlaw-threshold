"""
Shared pieces for the 20 Newsgroups replication of the power-law threshold.

The corpus, the feature matrix, the selectors, the classifier and the tests
live here so that 01-03 use one definition of each. The selectors mirror
code/05_baseline_comparison.py (power-law, chi-squared, top-k by frequency)
and code/06_knee_baseline.py (knee), and the test mirrors
code/23_kcurve_repcv.py (Nadeau--Bengio + Holm). Nothing in code/ is imported,
so this folder runs on its own.

Corpus:   sklearn fetch_20newsgroups(subset="all",
                                     remove=("headers", "footers", "quotes"))
Features: CountVectorizer(lowercase, token_pattern (?u)\\b[a-z]{2,}\\b,
                          ngram_range (1, 2), no stop words, no min_df)
Classifier (fixed, untuned): TfidfTransformer() defaults on the selected
          columns (fitted on the training fold), then LinearSVC(C=1.0).
"""

import json
import time
from pathlib import Path

import numpy as np
import powerlaw
import scipy.sparse as sp
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CACHE = ROOT / "cache"

TOKEN_PATTERN = r"(?u)\b[a-z]{2,}\b"
CLASSIFIER_NOTE = ("TfidfTransformer() with sklearn defaults (l2 norm, smooth idf), "
                   "fitted on the training fold's selected columns, then "
                   "LinearSVC(C=1.0, random_state=42), sklearn defaults otherwise; "
                   "no tuning")


def load_corpus():
    from sklearn.datasets import fetch_20newsgroups
    d = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"),
                           data_home=str(DATA))
    return d.data, np.asarray(d.target), list(d.target_names)


def build_matrix():
    """Unigram+bigram count matrix, cached as .npz. The vocabulary is built on
    the whole corpus (unsupervised, as the opcode matrix in the article); every
    selector then uses only in-fold counts, and terms absent from a training
    fold have frequency 0 there."""
    CACHE.mkdir(exist_ok=True)
    xf, yf = CACHE / "X_counts.npz", CACHE / "y.npy"
    meta_f = CACHE / "meta.json"
    if xf.exists() and yf.exists() and meta_f.exists():
        return sp.load_npz(xf).tocsr(), np.load(yf), json.loads(meta_f.read_text())
    docs, y, names = load_corpus()
    cv = CountVectorizer(lowercase=True, token_pattern=TOKEN_PATTERN,
                         ngram_range=(1, 2), dtype=np.int32)
    X = cv.fit_transform(docs).tocsr()
    vocab = cv.get_feature_names_out()
    meta = {"n_docs": int(X.shape[0]), "n_features": int(X.shape[1]),
            "n_classes": len(names), "class_names": names,
            "n_unigrams": int(sum(" " not in v for v in vocab)),
            "n_bigrams": int(sum(" " in v for v in vocab)),
            "nnz": int(X.nnz), "total_tokens_counted": int(X.sum()),
            "empty_docs": int((np.diff(X.indptr) == 0).sum()),
            "class_counts": np.bincount(y).tolist()}
    sp.save_npz(xf, X)
    np.save(yf, y)
    meta_f.write_text(json.dumps(meta, indent=1))
    return X, y, meta


def corpus_freqs(X):
    return np.asarray(X.sum(axis=0)).ravel()


# ------------------------------------------------------------------ selectors
def select_powerlaw(X_tr, y_tr=None):
    """Unsupervised; k is set by the fit (continuous default, as in the article)."""
    freqs = corpus_freqs(X_tr)
    fit = powerlaw.Fit(freqs[freqs > 0], verbose=False)
    idx = np.where(freqs >= fit.xmin)[0]
    return idx, {"xmin": float(fit.xmin), "alpha": float(fit.alpha)}


def select_chi2(X_tr, y_tr, k):
    scores, _ = chi2(X_tr, y_tr)
    scores = np.nan_to_num(scores, nan=-np.inf)
    return np.argsort(scores)[::-1][:k], {}


def select_top_freq(X_tr, y_tr, k):
    freqs = corpus_freqs(X_tr)
    return np.argsort(freqs, kind="stable")[::-1][:k], {}


def select_min_df(X_tr, y_tr, min_df):
    """Keep terms occurring in at least min_df training documents."""
    df = np.bincount(X_tr.indices, minlength=X_tr.shape[1])
    return np.where(df >= min_df)[0], {"min_df": min_df}


def select_knee(X_tr, y_tr=None):
    """Max distance from the chord of the sorted log-log rank-frequency curve,
    both axes scaled to [0, 1] (as code/06_knee_baseline.py)."""
    freqs = corpus_freqs(X_tr)
    nz = np.sort(freqs[freqs > 0])[::-1]
    x = np.log(np.arange(1, len(nz) + 1, dtype=float))
    y = np.log(nz.astype(float))
    x = (x - x[0]) / (x[-1] - x[0])
    y = (y - y[-1]) / (y[0] - y[-1])
    cutoff = nz[int(np.argmax(np.abs(x + y - 1.0)))]
    keep = np.where(freqs >= cutoff)[0]
    return keep, {"cutoff": float(cutoff)}


# ----------------------------------------------------------------- classifier
def fit_score(X_tr, y_tr, X_te, y_te, idx=None):
    t0 = time.time()
    if idx is not None:
        idx = np.sort(np.asarray(idx))
        X_tr, X_te = X_tr[:, idx], X_te[:, idx]
    tf = TfidfTransformer().fit(X_tr)
    clf = LinearSVC(C=1.0, random_state=SEED).fit(tf.transform(X_tr), y_tr)
    pred = clf.predict(tf.transform(X_te))
    return {"acc": 100 * accuracy_score(y_te, pred),
            "macro_f1": 100 * f1_score(y_te, pred, average="macro",
                                       zero_division=0),
            "k": int(X_tr.shape[1]),
            "seconds": round(time.time() - t0, 1),
            "pred": pred.tolist()}


def pooled(rows, tes, y):
    """Accuracy and macro F1 over pooled out-of-fold predictions."""
    pred = np.empty_like(y)
    for r, te in zip(rows, tes):
        pred[te] = r["pred"]
    return {"acc": round(100 * accuracy_score(y, pred), 2),
            "macro_f1": round(100 * f1_score(y, pred, average="macro",
                                             zero_division=0), 2)}


# ---------------------------------------------------------------------- tests
def nb_test(d, n_tr, n_te):
    """Nadeau--Bengio corrected resampled t-test on per-split differences."""
    d = np.asarray(d, float)
    J = len(d)
    var = np.var(d, ddof=1) * (1.0 / J + n_te / n_tr)
    t = d.mean() / np.sqrt(var) if var > 0 else 0.0
    p = 2 * stats.t.sf(abs(t), df=J - 1) if var > 0 else 1.0
    half = stats.t.ppf(0.975, df=J - 1) * np.sqrt(var)
    return {"mean_diff": round(float(d.mean()), 3), "t": round(float(t), 3),
            "p": float(p), "ci95": [round(float(d.mean() - half), 3),
                                    round(float(d.mean() + half), 3)]}


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.d = json.loads(self.path.read_text()) if self.path.exists() else {}

    def get(self, key):
        return self.d.get(key)

    def put(self, key, val):
        self.d[key] = val
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d))
        tmp.replace(self.path)
