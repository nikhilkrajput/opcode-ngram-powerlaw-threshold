# Label-free reduction of opcode n-gram feature spaces

Code and derived outputs for the article

> S. Verma, N. K. Rajput, *Label-free reduction of opcode n-gram feature
> spaces for malware classification by power-law threshold estimation*,
> submitted to *Annals of Telecommunications*.

The method fits a power-law tail to the corpus frequencies of opcode
n-grams with the Clauset-Shalizi-Newman estimator and keeps the n-grams at or
above the fitted lower bound x_min. It reads no labels and has no tuning
constant. Every number, table and figure in the article is produced by a
script in this repository. `REPRODUCE.md` maps each one to its script and
output file.

## Contents

| Folder | What it holds |
|---|---|
| `code/` | Opcode extraction (BIG 2015 IDA listings and System32 executables), the power-law fit and reduction, classifiers, selector comparisons, repeated cross-validation, stability, byte n-grams, the two variants of Section 8, corpus statistics, robustness checks and the figure scripts |
| `results/malware/` | BIG 2015 opcode results: fits, corpus-frequency vectors for n = 2, 3, 4, classification, selector comparisons, K curve, repeated CV, stability, variants (JSON and run logs) |
| `results/benign_matched/` | System32 opcode fits and corpus-frequency vectors (n = 2, 3), machine types |
| `results/bytes/` | BIG 2015 byte n-gram fits and the byte 2-gram frequency vector |
| `results/` (top level) | Corpus statistics, robustness checks, software environment |
| `second_dataset/` | The 20 Newsgroups experiment of Section 7.4: `code/` and `results/` |
| `figures/` | Figures as used in the article (PDF) |

## Data

None of the raw data are redistributed here.

- **BIG 2015** (Microsoft Malware Classification Challenge): R. Ronen,
  M. Radu, C. Feuerstein, E. Yom-Tov, M. Ahmadi, *Microsoft Malware
  Classification Challenge*, arXiv:1802.10135, 2018. Download `train.7z` and
  `trainLabels.csv` from the challenge page on Kaggle,
  <https://www.kaggle.com/c/malware-classification>, after accepting its
  terms. `code/01b_stream_opcodes_from_archive.py` reads the opcode
  sequences directly from `train.7z` (a solid archive of 197.9 GB when unpacked) one block at a
  time, so the archive never has to be unpacked.
- **System32 executables**: Microsoft software, not redistributed.
  `python code/01_extract_opcodes.py benign --sys32 C:/Windows/System32
  --out-dir results/benign_matched` rebuilds a comparable corpus from a
  Windows installation: the first 200 executables, in alphabetical order,
  that yield at least 50 instructions. The article's corpus came from
  Windows 11 Pro build 26300 with Capstone 5.0.7; another build gives a
  different set of files.
- **20 Newsgroups** (K. Lang, *NewsWeeder: Learning to Filter Netnews*,
  ICML 1995), by-date version: downloaded by scikit-learn
  (`fetch_20newsgroups`) into `second_dataset/data/` on first use.

The shipped outputs are statistics derived from these data: corpus
frequencies of each n-gram (the `*_freqs.npy` vectors, in the column order
of scikit-learn's `CountVectorizer`), fitted parameters, scores and
per-family results. The per-sample opcode sequences (`seqs/`), manifests
and reduced count matrices are not included; they are rebuilt from the raw
data by `01b_stream_opcodes_from_archive.py`, `01_extract_opcodes.py` and
`02_powerlaw_reduce.py`. `results/malware/blocks.json` is the block map of
`train.7z` (member names only), cached so that the byte n-gram and block-1
scripts do not have to rescan the archive.

## Environment

All runs used one Windows 11 machine (Intel Core i7-1255U, 15.7 GB RAM), as
recorded in `results/environment.json`:

| Package | Version |
|---|---|
| Python | 3.12.10 |
| scikit-learn | 1.9.0 |
| powerlaw | 2.0.0 |
| LightGBM | 4.7.0 |
| XGBoost | 3.4.1 |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| Capstone | 5.0.7 |
| pefile | 2024.8.26 |

`pip install -r requirements.txt` installs these, plus pandas, matplotlib
and py7zr. `code/20_benign_arch.py` and the benign mode of
`code/01_extract_opcodes.py` need a Windows System32 directory, and
`code/22_environment.py` queries the hardware through PowerShell; the other
scripts are platform-independent.

## Quick start

The scripts are run from `code/`, as their relative default paths assume.
These need only the files shipped here:

```bash
pip install -r requirements.txt
cd code
python 04_make_figures.py --malware-dir ../results/malware --n 3   # Fig. 1
python 24_kcurve_figure.py                                         # Fig. 2
python 18_alt_thresholds.py          # Section 6.1 runner-up and discrete thresholds
cd ../second_dataset/code && sh run_all.sh    # Section 7.4, Table 9
```

`run_all.sh` downloads 20 Newsgroups and rebuilds the count matrix. The
shipped `second_dataset/results/*.ckpt.json` files are the per-fit
checkpoints of the article's run, so with them in place the K curve and the
repeated cross-validation are read back rather than refitted; delete them to
recompute everything (about 1.5 h on one core). `04_summary.py` reads
`repcv.ckpt.json`, so keep a copy if you only want to rerun the summary.

The full pipeline from the raw data (BIG 2015 paths are placeholders):

```bash
cd code
python 01b_stream_opcodes_from_archive.py --archive /path/to/BIG2015/train.7z --out-dir ../results/malware
python 01_extract_opcodes.py benign --sys32 C:/Windows/System32 --out-dir ../results/benign_matched
for n in 2 3 4; do python 02_powerlaw_reduce.py --manifest ../results/malware/manifest.json --n $n --out-dir ../results/malware; done
for n in 2 3; do python 02_powerlaw_reduce.py --manifest ../results/benign_matched/manifest.json --n $n --out-dir ../results/benign_matched; done
python 03_classify.py --results-dir ../results/malware --labels /path/to/trainLabels.csv --orders 2 3
python 03_classify.py --results-dir ../results/malware --labels /path/to/trainLabels.csv --orders 4 --skip LinearSVM
```

and then the scripts listed in `REPRODUCE.md`, each of which takes
`--manifest ../results/malware/manifest.json --labels /path/to/trainLabels.csv`.

## Licence

Code: MIT licence (`LICENSE`). The derived outputs in `results/`,
`second_dataset/results/` and `figures/` are statistics computed from BIG
2015, from System32 executables and from 20 Newsgroups, provided for
reproducing the article; cite the datasets listed above when using them.

## Citation

If you use this code, please cite the article above and this repository,
<https://github.com/nikhilkrajput/opcode-ngram-powerlaw-threshold>.
