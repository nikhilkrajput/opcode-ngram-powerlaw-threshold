"""
Step 19 -- nop sensitivity on ONE extraction (Section 7.3).

nop is removed from the benign sequences themselves (results/benign_matched)
and the 2- and 3-gram fits are redone, so the only change is nop. Comparing
against a separately extracted corpus would confound nop with every other
difference between the two extractions.

Usage
-----
    python 19_benign_nop_refit.py --results ../results

Output
------
    adds "benign_nop_removed" to <results>/robustness.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import powerlaw
from sklearn.feature_extraction.text import CountVectorizer

TOKEN_PATTERN = r"(?u)\b[a-z]{2,}\b"   # as in 02_powerlaw_reduce.py


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="../results")
    args = ap.parse_args()
    res = Path(args.results)
    recs = json.loads((res / "benign_matched/manifest.json").read_text())
    seqs = [" ".join(t for t in Path(r["seq_file"]).read_text(encoding="utf-8").split()
                     if t != "nop") for r in recs]
    out = {}
    for n in (2, 3):
        X = CountVectorizer(ngram_range=(n, n), lowercase=False,
                            token_pattern=TOKEN_PATTERN).fit_transform(seqs)
        f = np.asarray(X.sum(axis=0)).ravel()
        f = f[f > 0].astype(float)
        fit = powerlaw.Fit(f, verbose=False)
        out[f"{n}gram"] = {"vocabulary": int(len(f)), "xmin": float(fit.xmin),
                           "alpha": float(fit.alpha),
                           "retained": int((f >= fit.xmin).sum()),
                           "retention_pct": round(100 * float((f >= fit.xmin).mean()), 2)}
        print(n, out[f"{n}gram"], flush=True)
    p = res / "robustness.json"
    rob = json.loads(p.read_text())
    rob["benign_nop_removed"] = out
    p.write_text(json.dumps(rob, indent=1))


if __name__ == "__main__":
    main()
