# Reproducing the article's results

Each item lists the script that produces it and the output file it writes.
Scripts are in `code/` unless stated otherwise and are run from that folder;
outputs are under `results/` unless stated otherwise. Scripts marked
"raw data" need BIG 2015 (`train.7z`, `trainLabels.csv`) or a System32
directory; the others run from the files shipped here.

| Article item | Script | Output |
|---|---|---|
| Opcode sequences, BIG 2015 (raw data) | `01b_stream_opcodes_from_archive.py --archive train.7z --out-dir ../results/malware` (or `01_extract_opcodes.py malware --asm-dir` on unpacked listings); filter in `opcode_filter.py` | `malware/seqs/`, `malware/manifest.json` (not shipped), `malware/blocks.json` |
| Opcode sequences, System32 (raw data) | `01_extract_opcodes.py benign --sys32 ... --out-dir ../results/benign_matched` | `benign_matched/seqs/`, `benign_matched/manifest.json` (not shipped) |
| Section 4.1: 96.2% directive share in block 1 (123 samples), `int 3` | `21_block1_inventory.py` (raw data) | `malware/block1_inventory.json`, `.log` |
| Sections 4.1 and 5.1: token and sample counts, 50 excluded samples, unrecognised words, benign corpus size | `14_corpus_stats.py` (raw data; its `benign_nop_dropped` part needs a separate extraction not shipped, see its docstring) | `corpus_stats.json` |
| Section 5.1: 199 of 200 benign files 64-bit; block-1 segments 32-bit | `20_benign_arch.py` (System32); `21_block1_inventory.py` | `benign_matched/arch.json`; `malware/block1_inventory.json` |
| Section 5, environment | `22_environment.py` (Windows) | `environment.json` |
| Table 1; Table 2 (malware rows); exponential comparisons (Section 6.1) | `02_powerlaw_reduce.py --n 2/3/4` on the malware manifest (raw data) | `malware/powerlaw_{2,3,4}gram.json`, `malware/opcode_{2,3,4}gram_freqs.npy` |
| Table 2 (benign rows); Section 7.1 | `02_powerlaw_reduce.py --n 2/3` on the benign manifest | `benign_matched/powerlaw_{2,3}gram.json`, `benign_matched/opcode_{2,3}gram_freqs.npy` |
| Section 6.1: second KS minima, discrete fit | `17_robustness.py` (its KS profile and discrete fit use only the shipped `malware/*_freqs.npy`, but the script starts with a part that needs the benign sequences), then `18_alt_thresholds.py` | `robustness.json` (`ks_profile`, `discrete_fit`, `alt_thresholds`) |
| Fig. 1 | `04_make_figures.py --malware-dir ../results/malware --n 3` | `figures/fig1_3gram_ccdf.pdf` |
| Table 3; Simda F1 at each order (Section 6.2) | `03_classify.py` (raw data; `--orders 4 --skip LinearSVM` for n = 4) | `malware/classification_results.json`, `malware/per_family_{2gram_LightGBM,3gram_LightGBM,4gram_XGBoost}.json`, `malware/n4_classify.log` |
| Section 6.3: accuracy against K | `23_kcurve_repcv.py --parts kcurve` (raw data) | `malware/kcurve_3gram.json`, `malware/kcurve_repcv.log` |
| Fig. 2 | `24_kcurve_figure.py` | `figures/fig2_kcurve.pdf` (reads `kcurve_3gram.json`, `selector_comparison_3gram.json`, `knee_comparison_3gram.json`, `no_selection_3gram.json`) |
| Table 4 (Section 6.4) | `12_no_selection_baseline.py --orders 2 3` (raw data) | `malware/no_selection_{2,3}gram.json`, `malware/no_selection.log` |
| Tables 5 and 6, except the knee row (Section 6.5) | `05_baseline_comparison.py` (raw data) | `malware/selector_comparison_3gram.json`, `malware/selector_full.log` |
| Table 5, knee row | `06_knee_baseline.py` (raw data) | `malware/knee_comparison_3gram.json`, `malware/knee.log` |
| Section 6.5: same feature set in two column orders | `13_column_order_check.py` (raw data) | `malware/column_order_3gram.json`, `malware/column_order.log` |
| Table 7; low-mode count and k range over 25 splits (Section 6.6) | `23_kcurve_repcv.py --parts repcv` (raw data) | `malware/repcv_3gram.json`, `malware/kcurve_repcv.log` |
| Section 6.7: fold thresholds, 25 subsample fits, median-of-5 variant | `07_stable_threshold.py --B 5` (raw data) | `malware/stable_threshold_3gram.json`, `malware/stable.log` |
| Table 8 (Section 7.2) | `08_byte_ngrams.py` (raw data; reads `malware/blocks.json`) | `bytes/byte_{2,3}gram_powerlaw.json`, `bytes/bytes_full.log`, `bytes/byte_2gram_freqs.npy` (the 129 MB `byte_3gram_freqs.npy` is not shipped) |
| Section 7.3: `int3` share, digit-keeping tokeniser | `17_robustness.py` (needs the benign sequences) | `robustness.json` (`benign_digit_tokens`) |
| Section 7.3: nop removed | `19_benign_nop_refit.py` (needs the benign sequences) | `robustness.json` (`benign_nop_removed`) |
| Section 7.4, Table 9 (20 Newsgroups) | `second_dataset/code/run_all.sh`: `01_features_fit.py`, `02_kcurve.py`, `03_repcv.py`, `04_summary.py` | `second_dataset/results/features.json`, `powerlaw_fit.json`, `kcurve.json`, `repcv.json`, `summary.json`, `*.log` |
| Section 8.1, Eq. (5) | `09_class_conditional.py` (raw data) | `malware/class_conditional_3gram.json`, `malware/classcond.log` |
| Table 10 (Section 8.2) | `11_unlabelled_repeats.py` (raw data) | `malware/unlabelled_repeats_3gram.json`, `malware/unlabelled_repeats.log` |

## Notes

- All BIG 2015 experiments use `StratifiedKFold(5, shuffle=True,
  random_state=42)` on the 10,818 samples in manifest order, so Tables 4 to
  6, the knee row, Section 6.7, Section 8 and the first repeat of Table 7
  share the same folds. The folds depend on the row order, which is the
  manifest order. `01b_stream_opcodes_from_archive.py` writes the manifest
  sorted by `pathlib.Path`, which compares case-insensitively on Windows
  (where the article's runs were made) and case-sensitively elsewhere; BIG
  2015 sample names mix upper and lower case, so on Linux or macOS the row
  order, and hence the folds, differ and the scores can differ slightly.
  `01_extract_opcodes.py malware` sorts plain strings (case-sensitively) and
  so also gives a different order from `01b` on Windows.
- The Holm adjustments quoted in the text for Table 5 (over five McNemar
  tests) and Table 10 (over four values of rho) are computed by
  `code/25_holm_adjustments.py` from the shipped p-values. Those of Tables 7
  and 9 are computed by `23_kcurve_repcv.py` and the text-corpus scripts.
- When given `--benign-dir`, `04_make_figures.py` also writes two
  supplementary benign plots that it labels "Fig. 2" and "Fig. 3"; they
  are not the article's figures.
- Absolute output paths printed in the logs have been replaced by `<repo>`
  and `<python>`, and a few labels that referred to an earlier manuscript
  numbering were updated in `n4_classify.log`, `unlabelled_repeats.log`,
  `kcurve_3gram.json` and the text-corpus fit outputs. No number was
  changed. Running `04_make_figures.py` or `24_kcurve_figure.py` rewrites
  the figure PDFs with a new creation date only.
- `malware/table3.log` is an earlier run of `05_baseline_comparison.py`
  that recorded accuracy and weighted F1 only; its accuracies and McNemar
  p-values equal those of `selector_full.log`, but its selection times were
  measured on a more heavily loaded machine. The article reports
  `selector_full.log` / `selector_comparison_3gram.json`.
- Timings are indicative: the machine was shared with other jobs during
  some runs (Section 5). The longest runs were the linear SVM at n = 3
  (7.7 h), mutual information in `05_baseline_comparison.py` (about 1.5 h of
  selection) and the repeated cross-validation of `23_kcurve_repcv.py`.
  `13` and `23` checkpoint per fold or per fit and resume when killed.
- LightGBM, XGBoost and the random forest use all cores; scores can differ
  in the last decimal across library versions and thread counts.
- `opcode_{n}gram_freqs.npy` holds the corpus frequency of every n-gram in
  the full vocabulary, in `CountVectorizer` column order, so every fit on
  the full corpus (Tables 1, 2 and 8 at n = 2, Fig. 1, Section 6.1) can be
  redone without the raw data as `powerlaw.Fit(f[f > 0])`, which is what
  `04_make_figures.py`, `17_robustness.py` and `18_alt_thresholds.py` do.
