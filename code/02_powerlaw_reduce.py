"""
Step 2 -- n-gram counts, power-law fit, parameter-free reduction.

This is the method of the article (Section 4), end to end:

  1. CountVectorizer over the whole corpus              -> X  (N x |G_n|)
  2. corpus frequency  f_g = sum_i X[i, g]              -> Eq. (1)
  3. Clauset-Shalizi-Newman fit via `powerlaw.Fit`      -> Eqs. (2), (3)
       alpha-hat  : MLE exponent
       xmin-hat   : KS-minimising lower bound  (NO tuning constant)
  4. keep columns with f_g >= xmin-hat                  -> Eq. (4)

The fit is done by powerlaw.Fit with default settings (continuous form,
discrete=False). On the directive-filtered corpus (opcode_filter.py) it gives
the 3-gram values of Table 1, xmin = 794 and alpha = 1.632. The
likelihood-ratio comparisons written to powerlaw_<n>gram.json are those of
Table 2.

Usage
-----
    python 02_powerlaw_reduce.py --manifest ../results/malware/manifest.json \
                                 --n 3 --out-dir ../results/malware

Output
------
    <out-dir>/opcode_<n>gram_reduced.npz    reduced sparse count matrix
    <out-dir>/opcode_<n>gram_kept.txt       retained n-grams, one per line
    <out-dir>/opcode_<n>gram_freqs.npy      full corpus-frequency vector
    <out-dir>/powerlaw_<n>gram.json         fit + reduction summary
"""

import argparse
import json
from pathlib import Path

import numpy as np
import powerlaw
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import CountVectorizer

# Mnemonics are lowercase ASCII words; the token pattern must not split them
# and must not lowercase-fold anything else.  Keep this identical across the
# malware and benign corpora, otherwise the two vocabularies are not
# comparable.
TOKEN_PATTERN = r"(?u)\b[a-z]{2,}\b"


def load_sequences(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    seqs, names = [], []
    for rec in manifest:
        seqs.append(Path(rec["seq_file"]).read_text(encoding="utf-8").strip())
        names.append(rec["name"])
    return names, seqs


def fit_and_reduce(seqs, n):
    cv = CountVectorizer(ngram_range=(n, n), lowercase=False,
                         token_pattern=TOKEN_PATTERN)
    X = cv.fit_transform(seqs)
    vocab = np.array(cv.get_feature_names_out())
    print(f"{n}-gram vocabulary: {X.shape[1]:,} features over {X.shape[0]:,} samples")

    freqs = np.asarray(X.sum(axis=0)).ravel()
    nonzero = freqs[freqs > 0]

    fit = powerlaw.Fit(nonzero, verbose=False)
    xmin, alpha = float(fit.xmin), float(fit.alpha)
    print(f"  CSN fit: xmin = {xmin:.0f}, alpha = {alpha:.4f}")

    keep = np.where(freqs >= xmin)[0]
    X_red = X[:, keep]
    factor = X.shape[1] / len(keep) if len(keep) else float("nan")
    print(f"  retained {len(keep):,} / {X.shape[1]:,} features "
          f"({100 * len(keep) / X.shape[1]:.2f}%, {factor:.1f}x reduction)")

    # Goodness of fit against the standard alternatives.  A positive R with
    # small p favours the power law; report both, do not round p away.
    comparisons = {}
    for alt in ("exponential", "lognormal", "truncated_power_law"):
        try:
            R, p = fit.distribution_compare("power_law", alt,
                                            normalized_ratio=True)
            comparisons[alt] = {"loglikelihood_ratio": float(R), "p": float(p)}
            print(f"  vs {alt:<20} R = {R:+.3f}  p = {p:.4f}")
        except Exception as exc:  # pragma: no cover - alt fit can fail
            comparisons[alt] = {"error": str(exc)}
            print(f"  vs {alt:<20} FAILED: {exc}")

    summary = {
        "n": n,
        "n_samples": int(X.shape[0]),
        "vocabulary_size": int(X.shape[1]),
        "xmin": xmin,
        "alpha": alpha,
        "retained": int(len(keep)),
        "retention_pct": round(100 * len(keep) / X.shape[1], 3),
        "reduction_factor": round(factor, 2),
        "token_pattern": TOKEN_PATTERN,
        "distribution_comparisons": comparisons,
    }
    return X_red, freqs, vocab[keep], summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _, seqs = load_sequences(args.manifest)
    X_red, freqs, kept, summary = fit_and_reduce(seqs, args.n)

    save_npz(str(out_dir / f"opcode_{args.n}gram_reduced.npz"), X_red)
    np.save(out_dir / f"opcode_{args.n}gram_freqs.npy", freqs)
    (out_dir / f"opcode_{args.n}gram_kept.txt").write_text(
        "\n".join(kept), encoding="utf-8")
    (out_dir / f"powerlaw_{args.n}gram.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWritten to {out_dir}")
