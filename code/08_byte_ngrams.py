"""
Step 8 -- does the fit behave the same on byte n-grams? (Section 7.2,
Table 8)

WHY
---
Whether the threshold transfers to other heavy-tailed static feature spaces
is an empirical question. BIG 2015 ships a .bytes hex dump beside every .asm
listing, so the experiment costs no new data: the same 25 solid blocks that
opcode extraction skips.

WHAT IT MEASURES
----------------
Corpus frequency distribution of byte 2-grams and 3-grams, the CSN fit
(x_min, alpha), the retention that follows, and the likelihood-ratio tests
against exponential, lognormal and truncated power law.

DESIGN NOTES
------------
Counting is done with np.bincount over an integer code per n-gram
(b0*256 + b1, or b0*65536 + b1*256 + b2), not with a Python Counter. The
vocabulary is bounded by construction -- 65,536 and 16,777,216 -- so the
accumulator is a fixed 134 MB array and the whole corpus streams through it.
A Counter over billions of tokens would not finish.

BYTES_PER_FILE caps how much of each sample is read. The full corpus is
51.7 GB of hex text, roughly 14.5 billion bytes; parsing all of it in Python
is not practical. The cap keeps every sample represented rather than sampling
files, which matters because the families differ in size. The cap is
recorded in the output JSON.

Usage
-----
    python 08_byte_ngrams.py --archive /path/to/BIG2015/train.7z \\
        --out-dir ../results/bytes --limit-blocks 1     # time one block first
    python 08_byte_ngrams.py --archive /path/to/BIG2015/train.7z --out-dir ../results/bytes

Reads the block map ../malware/blocks.json relative to --out-dir (written by
01b_stream_opcodes_from_archive.py and shipped in results/malware/).
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import py7zr

BYTES_PER_FILE = 50_000   # hex pairs read per sample; see design notes
HEXMAP = np.full(256, -1, dtype=np.int16)
for i, c in enumerate(b"0123456789ABCDEF"):
    HEXMAP[c] = i
for i, c in enumerate(b"0123456789abcdef"):
    HEXMAP[c] = i


def parse_bytes_file(raw, cap=BYTES_PER_FILE):
    """Hex dump -> uint8 array, dropping the address column and '??' holes.

    Lines look like:  00401000 4D 5A 90 00 03 00 ...
    The first token is an address; '??' marks an unmapped byte.
    """
    out = []
    for line in raw.split(b"\n"):
        parts = line.split()
        if len(parts) < 2:
            continue
        for tok in parts[1:]:                      # skip the address
            if len(tok) != 2:
                continue
            hi, lo = HEXMAP[tok[0]], HEXMAP[tok[1]]
            if hi < 0 or lo < 0:                   # '??' and friends
                continue
            out.append((hi << 4) | lo)
        if len(out) >= cap:
            break
    return np.asarray(out[:cap], dtype=np.uint8)


def accumulate(arr, counts2, counts3):
    if arr.size < 3:
        return
    a = arr.astype(np.int64)
    np.add.at(counts2, a[:-1] * 256 + a[1:], 1)
    np.add.at(counts3, a[:-2] * 65536 + a[1:-1] * 256 + a[2:], 1)


def fit_and_report(counts, n, out_dir, n_files, total_tokens):
    import powerlaw

    freqs = counts[counts > 0].astype(float)
    print(f"\n  byte {n}-gram vocabulary: {freqs.size:,} of {len(counts):,} possible")
    fit = powerlaw.Fit(freqs, verbose=False)
    keep = int(np.sum(counts >= fit.xmin))
    print(f"    CSN fit: xmin = {fit.xmin:.0f}, alpha = {fit.alpha:.4f}")
    print(f"    retained {keep:,} / {freqs.size:,} "
          f"({100*keep/freqs.size:.2f}%, {freqs.size/max(keep,1):.1f}x)")

    comparisons = {}
    for alt in ("exponential", "lognormal", "truncated_power_law"):
        try:
            R, p = fit.distribution_compare("power_law", alt,
                                            normalized_ratio=True)
            comparisons[alt] = {"loglikelihood_ratio": float(R), "p": float(p)}
            print(f"    vs {alt:<20} R = {R:+.3f}  p = {p:.4f}")
        except Exception as exc:
            comparisons[alt] = {"error": str(exc)}

    summary = {
        "feature": f"byte {n}-gram", "n_files": n_files,
        "bytes_per_file_cap": BYTES_PER_FILE, "total_bytes_read": total_tokens,
        "vocabulary_size": int(freqs.size), "xmin": float(fit.xmin),
        "alpha": float(fit.alpha), "retained": keep,
        "retention_pct": round(100 * keep / freqs.size, 3),
        "reduction_factor": round(freqs.size / max(keep, 1), 2),
        "distribution_comparisons": comparisons,
    }
    (out_dir / f"byte_{n}gram_powerlaw.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    np.save(out_dir / f"byte_{n}gram_freqs.npy", counts)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit-blocks", type=int, default=None)
    args = ap.parse_args()

    archive = Path(args.archive).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_scratch"

    blocks = json.loads(
        (Path(args.out_dir).parent / "malware" / "blocks.json").read_text())
    byte_blocks = [b for b in blocks
                   if any(m.lower().endswith(".bytes") for m in b["members"])]
    if args.limit_blocks:
        byte_blocks = byte_blocks[:args.limit_blocks]
    gb = sum(b["uncompressed"] for b in byte_blocks) / 1e9
    print(f"{len(byte_blocks)} .bytes blocks, {gb:.1f} GB to decompress")

    counts2 = np.zeros(256 ** 2, dtype=np.int64)
    counts3 = np.zeros(256 ** 3, dtype=np.int64)
    n_files = 0
    total = 0
    t0 = time.time()

    for i, block in enumerate(byte_blocks, 1):
        members = [m for m in block["members"] if m.lower().endswith(".bytes")]
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True, exist_ok=True)
        with py7zr.SevenZipFile(archive, mode="r") as z:
            z.extract(path=str(scratch), targets=members)
        for path in scratch.rglob("*.bytes"):
            arr = parse_bytes_file(path.read_bytes())
            path.unlink()
            accumulate(arr, counts2, counts3)
            n_files += 1
            total += int(arr.size)
        shutil.rmtree(scratch, ignore_errors=True)
        el = time.time() - t0
        print(f"  block {i}/{len(byte_blocks)}  {n_files:,} files  "
              f"{total/1e6:.0f}M bytes  {el/60:.1f} min", flush=True)

    print(f"\nread {total:,} bytes from {n_files:,} files "
          f"in {(time.time()-t0)/60:.1f} min")
    for n, c in ((2, counts2), (3, counts3)):
        fit_and_report(c, n, out_dir, n_files, total)
    print(f"\nWritten to {out_dir}")
