"""
Step 5 helper -- derived numbers quoted in Section 7.4, from the saved
results only.

Reads results/powerlaw_fit.json, kcurve.json, repcv.json and repcv.ckpt.json
and writes results/summary.json with:
  * the spread of the fold-wise power-law x_min and k,
  * the Spearman correlation between the fold's power-law k and its accuracy
    (is the power-law score just a point on the K curve?),
  * the K-curve accuracy linearly interpolated in log K at the mean
    power-law k, for comparison with the power-law pooled accuracy,
  * the share of the top-5,000 -> full-vocabulary accuracy gap that the
    power-law selection recovers.
"""

import json

import numpy as np
from scipy import stats

import common as C


def main():
    kc = json.loads((C.RESULTS / "kcurve.json").read_text())
    rc = json.loads((C.RESULTS / "repcv.json").read_text())
    ck = json.loads((C.RESULTS / "repcv.ckpt.json").read_text())
    pf = json.loads((C.RESULTS / "powerlaw_fit.json").read_text())

    J = rc["J"]
    k = np.array([ck[f"s{j}_power-law"]["k"] for j in range(J)])
    acc = np.array([ck[f"s{j}_power-law"]["acc"] for j in range(J)])
    xm = np.array([p["xmin"] for p in rc["powerlaw_per_split"]])
    rho, p = stats.spearmanr(k, acc)

    Ks = np.array([c["K"] for c in kc["curve"]], float)
    curve_acc = np.array([c["acc"] for c in kc["curve"]])
    k_mean_kc = kc["powerlaw_k_mean"]
    interp = float(np.interp(np.log(k_mean_kc), np.log(Ks), curve_acc))

    s = rc["summary"]
    gap = s["full"]["acc"]["mean"] - s["top-5000"]["acc"]["mean"]
    rec = (s["power-law"]["acc"]["mean"] - s["top-5000"]["acc"]["mean"]) / gap

    out = {
        "whole_corpus": {"xmin": pf["continuous"]["xmin"],
                         "k": pf["continuous"]["k_retained"],
                         "second_ks_minimum": pf["continuous"]["ks_profile_local_minima"][1],
                         "discrete_xmin": pf["discrete"]["xmin"],
                         "discrete_k": pf["discrete"]["k_retained"]},
        "repcv_powerlaw_xmin": {"min": float(xm.min()), "median": float(np.median(xm)),
                                "max": float(xm.max()),
                                "splits_xmin_below_20": int((xm < 20).sum()),
                                "splits_xmin_above_100": int((xm > 100).sum())},
        "repcv_powerlaw_k": {"min": int(k.min()), "median": float(np.median(k)),
                             "max": int(k.max()),
                             "max_over_min": round(float(k.max() / k.min()), 1)},
        "spearman_k_vs_acc_powerlaw": {"rho": round(float(rho), 3), "p": float(p)},
        "kcurve_acc_interpolated_at_mean_powerlaw_k": {
            "k": k_mean_kc, "acc": round(interp, 2),
            "powerlaw_pooled_acc": kc["powerlaw_pooled"]["acc"]},
        "repcv_share_of_top5000_to_full_gap_recovered_by_powerlaw": round(float(rec), 3),
        "repcv_fraction_of_full_vocab_kept_by_powerlaw_mean":
            round(s["power-law"]["k_mean"] / s["full"]["k_mean"], 4),
    }
    (C.RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
