"""
Step 4 -- figures, plotted from measured frequency data only.

This script plots measured data only.  It reads the
`opcode_<n>gram_freqs.npy` arrays written by 02_powerlaw_reduce.py -- the
actual corpus frequencies -- and draws their empirical CCDF with the fitted
tails; it never draws samples from the fitted model.  It builds whichever
figures it has data for and reports what it had to skip; it never substitutes
simulated data for missing measurements.

Usage
-----
    python 04_make_figures.py --malware-dir ../results/malware \
                              --benign-dir  ../results/benign_matched \
                              --out-dir     ../figures --n 3

Either corpus directory may be omitted or absent; the figures that depend on
it are then skipped with a message.

Output
------
    fig1_<n>gram_ccdf.pdf              malware CCDF with fit and x_min
                                       (with --n 3: Fig. 1 of the article)
    fig2_benign_<n>gram_ccdf.pdf       benign CCDF with fit and x_min
    fig3_benign_vs_malware_ccdf.pdf    both corpora overlaid

Only the first is in the article; the benign and overlay plots are
supplementary.  (Fig. 2 of the article is drawn by 24_kcurve_figure.py.)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import powerlaw                   # noqa: E402

BLUE, ORANGE, GREY, GREEN = "#2166AC", "#D6604D", "#666666", "#1B7837"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 300, "savefig.dpi": 600,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight",
})


def load_freqs(results_dir, n):
    """Measured corpus frequencies, or None if step 2 has not been run."""
    if results_dir is None:
        return None
    path = Path(results_dir) / f"opcode_{n}gram_freqs.npy"
    if not path.exists():
        print(f"  SKIP: {path} not found -- run 02_powerlaw_reduce.py. "
              f"This script will not simulate the distribution.")
        return None
    freqs = np.load(path)
    return freqs[freqs > 0].astype(float)


def empirical_ccdf(x):
    xs = np.sort(x)
    return xs, 1.0 - np.arange(len(xs)) / len(xs)


def fig_ccdf(freqs, out_path, n, corpus_label, with_truncated=True):
    """Empirical CCDF with the fitted tails.

    Layout note: the legend sits ABOVE the axes, not inside them. At these
    exponent values the data curve sweeps through the whole lower-left region,
    which is where a default legend lands, and the x_min rule crosses it as
    well. Placing the legend outside keeps every label readable at column
    width.

    Both tails are drawn because the likelihood-ratio tests prefer the
    truncated form (Table 2); showing only the pure fit would hide that the
    line does not track the data at the top of the range.
    """
    fit = powerlaw.Fit(freqs, verbose=False)
    fig, ax = plt.subplots(figsize=(3.4, 2.55))

    xs, ccdf = empirical_ccdf(freqs)
    # rasterize=True is not cosmetic: this layer is ~400k markers, and left as
    # vector paths it produces a >1 MB page that several PDF readers fail to
    # render at all (they show a blank page). Rasterising the data collapses it
    # to one image while axes, fitted lines and all text stay vector.
    ax.loglog(xs, ccdf, ".", color=GREY, markersize=1.4, rasterized=True,
              label=f"{corpus_label}, {len(freqs):,} {n}-grams")
    # The fitted CCDFs are conditional on f >= xmin, so they start at 1. The
    # data curve is over ALL n-grams. Plotting the fits unscaled (as
    # powerlaw's plot_ccdf does) floats them ~log10(1/tail_frac) decades above
    # the data they describe. Scale by the tail fraction so both are
    # P(F >= f) over the whole vocabulary and the overlay can be read.
    tail_frac = float(np.mean(freqs >= fit.xmin))
    grid = np.unique(np.logspace(np.log10(fit.xmin), np.log10(freqs.max()), 200))
    ax.loglog(grid, tail_frac * fit.power_law.ccdf(grid), color=ORANGE,
              linestyle="--", linewidth=1.3,
              label=rf"power law $\hat{{\alpha}}$={fit.alpha:.2f}")
    if with_truncated:
        try:
            tpl = fit.truncated_power_law
            ax.loglog(grid, tail_frac * tpl.ccdf(grid), color=GREEN,
                      linestyle="-.", linewidth=1.3,
                      label=rf"truncated $\hat{{\alpha}}$={tpl.alpha:.2f}")
            print(f"    truncated power law: alpha={tpl.alpha:.4f}, "
                  f"lambda={tpl.Lambda:.4e}")
        except Exception as exc:
            print(f"    truncated power law not drawn: {exc}")
    ax.axvline(fit.xmin, color=BLUE, linewidth=1.0,
               label=rf"$\hat{{x}}_{{\min}}$={fit.xmin:.0f}")

    # The truncated fit plunges far below the smallest observable probability
    # (1/len(freqs)); clip there so the data occupy the panel.
    ax.set_ylim(bottom=0.3 / len(freqs))
    ax.set_xlabel(r"corpus frequency $f_g$")
    ax.set_ylabel(r"$P(F \geq f_g)$")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=2, fontsize=6.2, handlelength=2.6, handletextpad=0.4,
              columnspacing=0.9, borderaxespad=0.0, labelspacing=0.35)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path.name}  (xmin={fit.xmin:.0f}, alpha={fit.alpha:.4f})")



def fig_overlay(mal_freqs, ben_freqs, out_path, n):
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for freqs, colour, label in (
            (mal_freqs, ORANGE, f"BIG 2015 malware ({len(mal_freqs):,})"),
            (ben_freqs, BLUE, f"System32 benign ({len(ben_freqs):,})")):
        xs, ccdf = empirical_ccdf(freqs)
        ax.loglog(xs, ccdf, "-", color=colour, linewidth=1.2, rasterized=True,
                  label=label)
    ax.set_xlabel(rf"corpus frequency of {n}-gram $f_g$")
    ax.set_ylabel(r"$P(F \geq f_g)$")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=1, fontsize=7, handlelength=1.9, handletextpad=0.5,
              borderaxespad=0.0)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path.name}")



def report_fit(results_dir, n, tag):
    if results_dir is None:
        return
    path = Path(results_dir) / f"powerlaw_{n}gram.json"
    if not path.exists():
        return
    s = json.loads(path.read_text(encoding="utf-8"))
    print(f"  {tag}: xmin={s['xmin']:.0f} alpha={s['alpha']:.4f} "
          f"retained={s['retained']:,} ({s['retention_pct']}%)")
    for alt, cmp_ in s.get("distribution_comparisons", {}).items():
        if "loglikelihood_ratio" in cmp_:
            verdict = ("power law favoured" if cmp_["loglikelihood_ratio"] > 0
                       else "ALTERNATIVE favoured")
            sig = "significant" if cmp_["p"] < 0.05 else "not significant"
            print(f"      vs {alt:<20} R={cmp_['loglikelihood_ratio']:+.3f} "
                  f"p={cmp_['p']:.4f}  ({verdict}, {sig})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--malware-dir", default=None)
    ap.add_argument("--benign-dir", default=None)
    ap.add_argument("--out-dir", default="../figures")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    n = args.n

    print("Loading measured frequency vectors:")
    mal = load_freqs(args.malware_dir, n)
    ben = load_freqs(args.benign_dir, n)

    print("\nBuilding figures:")
    built, skipped = [], []

    if mal is not None:
        fig_ccdf(mal, out_dir / f"fig1_{n}gram_ccdf.pdf", n, "BIG 2015 malware")
        built.append("Fig. 1")
    else:
        skipped.append("Fig. 1 (malware CCDF)")

    if ben is not None:
        fig_ccdf(ben, out_dir / f"fig2_benign_{n}gram_ccdf.pdf", n,
                 "System32 benign")
        built.append("Fig. 2")
    else:
        skipped.append("Fig. 2 (benign CCDF)")

    if mal is not None and ben is not None:
        fig_overlay(mal, ben, out_dir / f"fig3_benign_vs_malware_ccdf.pdf", n)
        built.append("Fig. 3")
    else:
        skipped.append("Fig. 3 (overlay -- needs both corpora)")

    print("\nFit summaries:")
    report_fit(args.malware_dir, n, "malware")
    report_fit(args.benign_dir, n, "benign")

    print(f"\nBuilt: {', '.join(built) if built else 'nothing'}")
    if skipped:
        print(f"Skipped: {'; '.join(skipped)}")
