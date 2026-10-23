"""
Step 25 -- Holm adjustments quoted in the text for Table 5 and Table 10.

Table 5 (Section 6.5): Holm over the five McNemar tests against the
power-law selector (knee, chi-squared, mutual information, top-k, top-5,000).
Table 10 (Section 8.2): Holm over the four paired t-tests (power-law against
chi-squared) at rho = 0.05, 0.10, 0.25 and 0.50.

Reads only shipped result files; needs no raw data.

Usage
-----
    python 25_holm_adjustments.py --results ../results
"""

import argparse
import json
from pathlib import Path


def holm(pvals):
    """Holm step-down adjusted p-values, in the input order."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="../results")
    args = ap.parse_args()
    res = Path(args.results) / "malware"

    sel = json.loads((res / "selector_comparison_3gram.json").read_text())
    knee = json.loads((res / "knee_comparison_3gram.json").read_text())
    tests = dict(sel["mcnemar_vs_powerlaw"])
    tests["knee"] = knee["mcnemar_vs_powerlaw"]["knee"]
    names = list(tests)
    adj = holm([tests[n]["p"] for n in names])
    print("Table 5: McNemar against power-law, Holm over five tests")
    for n, a in zip(names, adj):
        print(f"  {n:<16} p = {tests[n]['p']:.3g}   Holm p = {a:.3g}")

    unl = json.loads((res / "unlabelled_repeats_3gram.json").read_text())
    rhos = [r for r in ("0.05", "0.1", "0.25", "0.5")]
    ps = [unl["paired_tests"][r]["t_test_p"] for r in rhos]
    adj = holm(ps)
    print("\nTable 10: paired t-test power-law vs chi-squared, Holm over four rho")
    for r, p, a in zip(rhos, ps, adj):
        print(f"  rho = {r:<5} p = {p:.3g}   Holm p = {a:.3g}")


if __name__ == "__main__":
    main()
