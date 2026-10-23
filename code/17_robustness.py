"""
Step 17 -- robustness checks (Sections 6.1 and 7.3).

(a) KS profile. For each malware order n, the KS distance D(x_min) over every
    candidate x_min, its global minimum, and the best candidate in the low
    (< 100) and high (>= 100) frequency ranges, so the reader can see whether
    the reported x_min is the high or the low mode of a bimodal objective.
(b) Discrete fit. The article's fits use powerlaw's default continuous form on
    integer counts; refit with discrete=True and report x_min and alpha.
(c) Tokenisation. TOKEN_PATTERN \\b[a-z]{2,}\\b silently drops mnemonics that
    contain a digit (int3, ud2, cvtsi2sd, ...). Count them in the benign
    sequences, and refit the benign 2- and 3-grams with a pattern that keeps
    them.

Usage
-----
    python 17_robustness.py --results ../results [--orders 2 3 4] [--discrete-orders 2 3]

Output
------
    <results>/robustness.json
"""

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import powerlaw
from sklearn.feature_extraction.text import CountVectorizer

DIGIT_PATTERN = r"(?u)\b[a-z][a-z0-9]+\b"


def ks_profile(freqs):
    fit = powerlaw.Fit(freqs, verbose=False)
    r = fit.xmin_fitting_results
    xs, Ds = np.asarray(r["xmins"], float), np.asarray(r["distances"], float)
    ok = np.isfinite(Ds) & np.asarray(r["valid_fits"], bool)
    xs, Ds = xs[ok], Ds[ok]
    lo, hi = xs < 100, xs >= 100
    best = lambda m: ({"xmin": float(xs[m][np.argmin(Ds[m])]),
                       "D": float(Ds[m].min())} if m.any() else None)
    # thinned profile for plotting / inspection
    keep = np.unique(np.linspace(0, len(xs) - 1, 400).astype(int))
    return {"xmin": float(fit.xmin), "alpha": float(fit.alpha), "D": float(fit.D),
            "best_low": best(lo), "best_high": best(hi),
            "profile": [[float(xs[i]), float(Ds[i])] for i in keep]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="../results")
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--discrete-orders", type=int, nargs="+", default=[2, 3])
    args = ap.parse_args()
    res = Path(args.results)
    out_path = res / "robustness.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}

    def save():
        out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")

    # (c) tokenisation, benign -- cheap, first
    recs = json.loads((res / "benign_matched/manifest.json").read_text())
    seqs = [Path(r["seq_file"]).read_text(encoding="utf-8") for r in recs]
    cnt = Counter(t for s in seqs for t in s.split())
    total = sum(cnt.values())
    dig = {t: v for t, v in cnt.items() if re.search(r"\d", t)}
    tok = {"benign_tokens": total,
           "digit_tokens": sum(dig.values()),
           "digit_pct": round(100 * sum(dig.values()) / total, 2),
           "top_digit_mnemonics": sorted(dig.items(), key=lambda x: -x[1])[:10]}
    for n in (2, 3):
        X = CountVectorizer(ngram_range=(n, n), lowercase=False,
                            token_pattern=DIGIT_PATTERN).fit_transform(seqs)
        f = np.asarray(X.sum(axis=0)).ravel()
        f = f[f > 0].astype(float)
        fit = powerlaw.Fit(f, verbose=False)
        tok[f"{n}gram"] = {"vocabulary": int(len(f)), "xmin": float(fit.xmin),
                           "alpha": float(fit.alpha),
                           "retained": int((f >= fit.xmin).sum()),
                           "retention_pct": round(100 * float((f >= fit.xmin).mean()), 2)}
        print("benign digits kept", n, tok[f"{n}gram"], flush=True)
    out["benign_digit_tokens"] = tok
    save()

    # (a) KS profile, malware
    out.setdefault("ks_profile", {})
    for n in args.orders:
        f = np.load(res / f"malware/opcode_{n}gram_freqs.npy")
        f = f[f > 0].astype(float)
        t0 = time.time()
        out["ks_profile"][str(n)] = ks_profile(f)
        p = out["ks_profile"][str(n)]
        print(f"KS n={n}: xmin={p['xmin']} low={p['best_low']} high={p['best_high']} "
              f"({time.time() - t0:.0f}s)", flush=True)
        save()

    # (b) discrete fit, malware
    out.setdefault("discrete_fit", {})
    for n in args.discrete_orders:
        f = np.load(res / f"malware/opcode_{n}gram_freqs.npy")
        f = f[f > 0].astype(float)
        t0 = time.time()
        fit = powerlaw.Fit(f, discrete=True, verbose=False)
        out["discrete_fit"][str(n)] = {
            "xmin": float(fit.xmin), "alpha": float(fit.alpha),
            "retained": int((f >= fit.xmin).sum()),
            "seconds": round(time.time() - t0, 1)}
        print(f"discrete n={n}: {out['discrete_fit'][str(n)]}", flush=True)
        save()


if __name__ == "__main__":
    main()
