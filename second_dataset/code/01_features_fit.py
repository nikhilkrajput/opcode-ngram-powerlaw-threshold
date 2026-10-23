"""
Step 1-2 -- features and the power-law fit on the whole 20 Newsgroups corpus.

Builds the unigram+bigram count matrix (see common.py), then fits the
power-law tail to the corpus term frequencies with powerlaw.Fit(freqs), the
continuous default used in the article (Section 7.4). Also reports:
  * the discrete fit (discrete=True) and the k it would retain;
  * the KS profile over candidate x_min (fit.xmin_fitting_results): its local
    minima, ranked by KS distance, with the k each would retain;
  * likelihood-ratio comparisons of the tail against exponential, lognormal
    and truncated power law (distribution_compare(..., normalized_ratio=True));
  * the knee cut-off and the document-frequency cut-offs, for scale.

Output: results/features.json, results/powerlaw_fit.json
"""

import json
import time
import warnings

import numpy as np
import powerlaw

import common as C


def local_minima(xs, ds, valid):
    """Indices of strict-or-flat local minima of the KS profile among valid fits."""
    ds = np.where(valid, ds, np.inf)
    out = []
    for i in range(len(ds)):
        if not np.isfinite(ds[i]):
            continue
        left = ds[i - 1] if i > 0 else np.inf
        right = ds[i + 1] if i + 1 < len(ds) else np.inf
        if ds[i] <= left and ds[i] <= right:
            out.append(i)
    return sorted(out, key=lambda i: ds[i])


def describe(fit, freqs, label):
    r = fit.xmin_fitting_results
    xs, ds, al, valid = (np.asarray(r[k]) for k in
                         ("xmins", "distances", "alphas", "valid_fits"))
    mins = local_minima(xs, ds, valid)[:6]
    return {
        "form": label,
        "xmin": float(fit.xmin), "alpha": float(fit.alpha),
        "sigma": float(fit.sigma), "KS_D": float(fit.D),
        "n_tail": int(fit.n_tail),
        "k_retained": int((freqs >= fit.xmin).sum()),
        "n_candidate_xmin": int(len(xs)),
        "ks_profile_local_minima": [
            {"xmin": float(xs[i]), "KS_D": float(ds[i]), "alpha": float(al[i]),
             "k_retained": int((freqs >= xs[i]).sum()),
             "D_minus_global_min": float(ds[i] - np.min(ds[valid]))}
            for i in mins],
        # coarse profile for plotting: every candidate up to 1000, then sparse
        "ks_profile": [[float(x), float(d)] for x, d, v in zip(xs, ds, valid)
                       if v and (x <= 1000 or int(x) % 50 == 0)],
    }


def main():
    C.RESULTS.mkdir(exist_ok=True)
    X, y, meta = C.build_matrix()
    freqs = C.corpus_freqs(X)
    nz = freqs[freqs > 0].astype(float)
    df = np.bincount(X.indices, minlength=X.shape[1])

    feats = dict(meta)
    feats.update({
        "vectorizer": {"lowercase": True, "token_pattern": C.TOKEN_PATTERN,
                       "ngram_range": [1, 2], "stop_words": None, "min_df": 1},
        "corpus": "fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))",
        "n_unique_frequency_values": int(len(np.unique(nz))),
        "features_freq_eq_1": int((freqs == 1).sum()),
        "max_freq": int(freqs.max()),
        "k_at_min_df": {str(m): int((df >= m).sum()) for m in (2, 5, 10)},
    })
    (C.RESULTS / "features.json").write_text(json.dumps(feats, indent=1))
    print(json.dumps({k: v for k, v in feats.items() if k != "class_names"}, indent=1))

    out = {"data": "corpus term frequencies of all 1,139,171-ish features (see features.json)",
           "powerlaw_version": "2.0.0"}
    t0 = time.time()
    fit = powerlaw.Fit(nz, verbose=False)
    out["continuous"] = describe(fit, freqs, "continuous (powerlaw.Fit default; the article's rule)")
    out["continuous"]["seconds"] = round(time.time() - t0, 1)
    print("continuous", {k: v for k, v in out["continuous"].items() if k != "ks_profile"}, flush=True)

    comps = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for alt in ("exponential", "lognormal", "truncated_power_law"):
            R, p = fit.distribution_compare("power_law", alt, normalized_ratio=True)
            comps[alt] = {"R": float(R), "p": float(p),
                          "favours": "power_law" if R > 0 else alt,
                          "significant_at_0.05": bool(p < 0.05)}
            print(alt, comps[alt], flush=True)
    out["continuous"]["likelihood_ratio_vs_power_law"] = comps
    out["continuous"]["lognormal_params"] = {"mu": float(fit.lognormal.mu),
                                             "sigma": float(fit.lognormal.sigma)}
    out["continuous"]["truncated_power_law_params"] = {
        "alpha": float(fit.truncated_power_law.alpha),
        "lambda": float(fit.truncated_power_law.Lambda)}

    t0 = time.time()
    fit_d = powerlaw.Fit(nz, discrete=True, verbose=False)
    out["discrete"] = describe(fit_d, freqs, "discrete=True")
    out["discrete"]["seconds"] = round(time.time() - t0, 1)
    print("discrete", {k: v for k, v in out["discrete"].items() if k != "ks_profile"}, flush=True)

    knee_idx, knee = C.select_knee(X)
    out["for_scale"] = {"knee_cutoff": knee["cutoff"], "knee_k": int(len(knee_idx)),
                        "k_at_min_df": feats["k_at_min_df"],
                        "full_vocabulary": int(X.shape[1])}
    (C.RESULTS / "powerlaw_fit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["for_scale"], indent=1))


if __name__ == "__main__":
    main()
