"""
Step 18 -- retention at the competing KS minima and at the discrete-fit
threshold (from robustness.json), so the fragility of the Table 1 thresholds
discussed in Section 6.1 is traceable.

Usage
-----
    python 18_alt_thresholds.py --results ../results

Output
------
    adds "alt_thresholds" to <results>/robustness.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def local_minima(profile):
    pr = np.asarray(profile)
    x, d = pr[:, 0], pr[:, 1]
    idx = [i for i in range(1, len(d) - 1) if d[i] < d[i - 1] and d[i] <= d[i + 1]]
    return sorted(((float(x[i]), float(d[i])) for i in idx), key=lambda t: t[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="../results")
    args = ap.parse_args()
    res = Path(args.results)
    rob = json.loads((res / "robustness.json").read_text())
    out = {}
    for n, prof in rob["ks_profile"].items():
        f = np.load(res / f"malware/opcode_{n}gram_freqs.npy")
        f = f[f > 0]
        # local minima of the thinned profile, plus the exact best candidate
        # below and above 100 from the full scan
        mins = local_minima(prof["profile"]) + [
            (b["xmin"], b["D"]) for b in (prof["best_low"], prof["best_high"]) if b]
        mins.sort(key=lambda t: t[1])
        # runner-up: best local minimum more than a factor 5 away from the global one
        runner = next((m for m in mins
                       if max(m[0], prof["xmin"]) / min(m[0], prof["xmin"]) > 5), None)
        entry = {"global": {"xmin": prof["xmin"], "D": prof["D"]},
                 "runner_up": None}
        if runner:
            k = int((f >= runner[0]).sum())
            entry["runner_up"] = {"xmin": runner[0], "D": runner[1], "retained": k,
                                  "factor": round(len(f) / k, 1)}
        disc = rob.get("discrete_fit", {}).get(n)
        if disc:
            k = int((f >= disc["xmin"]).sum())
            entry["discrete"] = {"xmin": disc["xmin"], "retained": k,
                                 "factor": round(len(f) / k, 1)}
        out[n] = entry
        print(n, entry)
    rob["alt_thresholds"] = out
    (res / "robustness.json").write_text(json.dumps(rob, indent=1))


if __name__ == "__main__":
    main()
