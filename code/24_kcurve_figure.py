"""
Step 24 -- Fig. 2 of the article (Section 6.3): LightGBM accuracy and
macro F1 against the number of
3-grams kept (top-K by in-fold frequency), with the power-law threshold, knee
detection, top-5,000 and the unreduced matrix marked. Every value is read
from results/; nothing is typed in.

Usage
-----
    python 24_kcurve_figure.py --results ../results --out ../figures/fig2_kcurve.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

GREY, BLUE, ORANGE, GREEN = "#444444", "#2166AC", "#D6604D", "#1B7837"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def per_fold_mean(rows, key):
    return float(np.mean(rows[key]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default="../figures/fig2_kcurve.pdf")
    args = ap.parse_args()
    res = Path(args.results) / "malware"
    kc = json.loads((res / "kcurve_3gram.json").read_text())
    sel = json.loads((res / "selector_comparison_3gram.json").read_text())
    knee = json.loads((res / "knee_comparison_3gram.json").read_text())
    nos = json.loads((res / "no_selection_3gram.json").read_text())

    K = [r["K"] for r in kc["curve"]]
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.75), sharex=True)
    for ax, met, lab, pooled_key in ((axes[0], "acc", "accuracy (%)", "accuracy"),
                                     (axes[1], "macro_f1", r"macro $F_1$ (%)", "macro_f1")):
        ax.semilogx(K, [r[met] for r in kc["curve"]], "o-", color=GREY,
                    markersize=2.5, linewidth=1, label="top-$K$ by frequency")
        pl_k = float(np.median(sel["k_per_fold"]))
        ax.plot([pl_k], [sel["methods"]["power-law"][pooled_key]], "D",
                color=BLUE, markersize=4, label=r"$\hat{x}_{\min}$ (median $k$)")
        ax.plot([knee["methods"]["knee"]["k_mean"]],
                [knee["methods"]["knee"][pooled_key]], "s", color=ORANGE,
                markersize=3.5, label="knee")
        ax.axhline(nos["methods"]["full"][pooled_key], color=GREEN,
                   linestyle="--", linewidth=0.9, label="unreduced")
        ax.set_ylabel(lab)
        ax.set_xlabel("features kept $K$")
        ax.set_xticks([1e2, 1e3, 1e4, 1e5])
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.84))
    fig.legend(h, l, frameon=False, loc="upper center", ncol=4,
               handlelength=1.4, columnspacing=0.8, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
