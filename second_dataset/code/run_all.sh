#!/usr/bin/env sh
# Re-run the 20 Newsgroups replication end to end (about 1.5 h on one core).
# Downloads the corpus to ../data on first use; caches the count matrix in
# ../cache. Delete ../results/*.ckpt.json to recompute instead of resuming.
set -e
cd "$(dirname "$0")"
python -u 01_features_fit.py > ../results/01_features_fit.log 2>&1
python -u 02_kcurve.py       > ../results/02_kcurve.log 2>&1
python -u 03_repcv.py        > ../results/03_repcv.log 2>&1
python -u 04_summary.py     > ../results/04_summary.log 2>&1
