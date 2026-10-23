"""
Step 1b -- opcode sequences from train.7z, one solid block at a time.

WHY THIS SHAPE
--------------
BIG 2015's train.7z is a SOLID archive: 197.9 GB uncompressed across 95 LZMA
blocks of ~2.14 GB each. Two consequences drive the design.

1. It cannot be unpacked. 197.9 GB, or 148.4 GB for the .asm listings alone,
   exceeds any sensible free-space budget. Listings are therefore extracted one
   block at a time, converted to opcode sequences, and deleted. Peak scratch
   use is one block, ~2.2 GB.

2. Work is aligned to block boundaries, not to arbitrary file batches. In a
   solid archive a block is the unit of decompression, so touching members
   from k different blocks costs k block decompressions. Processing
   block-by-block decompresses each exactly once -- a single sequential pass.

The .asm and .bytes files are almost perfectly segregated across blocks: 70
blocks hold .asm, 25 hold .bytes, 1 is mixed. Skipping the .bytes-only blocks
avoids 49.5 GB of decompression that opcode extraction has no use for
(08_byte_ngrams.py reads those blocks separately).

Measured on py7zr 1.1.3: extract(targets=...) seeks to the containing block
rather than re-decompressing from the archive start, which is what makes the
per-block approach O(n) instead of O(n^2).

Usage
-----
    python 01b_stream_opcodes_from_archive.py \
        --archive /path/to/BIG2015/train.7z --out-dir ../results/malware

    # one block only, to measure throughput before committing
    python 01b_stream_opcodes_from_archive.py --archive ... --out-dir ... --limit-blocks 1

Resumable: rerun and it skips blocks whose sequences already exist.

Output
------
    <out-dir>/seqs/<sample-id>.txt   one opcode sequence per sample
    <out-dir>/manifest.json          rewritten after every block
    <out-dir>/blocks.json            cached block map
"""

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import py7zr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opcode_filter import filter_opcodes  # noqa: E402

# Matches "whitespace, two hex digits, optional +, whitespace, lowercase word".
# NOTE: this matches data lines as well as instruction lines -- IDA gives them
# the same column layout -- so the capture MUST be passed through
# filter_opcodes(). Without it, 96% of the tokens are `db`/`dd` padding rather
# than instructions. See opcode_filter.py for the measurement.
OPCODE_RGX = re.compile(r"\s[\dA-F]{2}(?:\+)?\s+([a-z]+)\s")


def build_block_map(archive, cache_path):
    """Group members into solid blocks. A non-None `compressed` starts a block."""
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    print("mapping solid blocks ...", flush=True)
    with py7zr.SevenZipFile(archive, mode="r") as z:
        info = z.list()

    blocks, cur = [], None
    for f in info:
        if f.compressed is not None:
            cur = {"start": f.filename, "members": [], "uncompressed": 0}
            blocks.append(cur)
        if cur is None:
            continue
        cur["members"].append(f.filename)
        cur["uncompressed"] += f.uncompressed or 0

    for b in blocks:
        b["asm"] = [m for m in b["members"] if m.lower().endswith(".asm")]

    cache_path.write_text(json.dumps(blocks, indent=1), encoding="utf-8")
    total = sum(b["uncompressed"] for b in blocks)
    asm_total = sum(b["uncompressed"] for b in blocks if b["asm"])
    print(f"  {len(blocks)} blocks, {total/1e9:.1f} GB uncompressed")
    print(f"  {sum(1 for b in blocks if b['asm'])} blocks hold .asm "
          f"({asm_total/1e9:.1f} GB) -- the rest are skipped")
    return blocks


def process_block(archive, block, scratch, seq_dir):
    """Extract one block's .asm members, convert, delete. Returns records."""
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)

    with py7zr.SevenZipFile(archive, mode="r") as z:
        z.extract(path=str(scratch), targets=block["asm"])

    records = []
    for path in scratch.rglob("*.asm"):
        text = path.read_text(encoding="latin-1", errors="ignore")
        opcodes = filter_opcodes(OPCODE_RGX.findall(text))
        path.unlink()  # free the listing immediately
        if not opcodes:
            continue
        seq_path = seq_dir / f"{path.stem}.txt"
        seq_path.write_text(" ".join(opcodes), encoding="utf-8")
        records.append({"name": path.stem, "n_opcodes": len(opcodes),
                        "seq_file": str(seq_path)})

    shutil.rmtree(scratch, ignore_errors=True)
    return records


def write_manifest(seq_dir, out_dir):
    manifest = [{"name": p.stem,
                 "n_opcodes": len(p.read_text(encoding="utf-8").split()),
                 "seq_file": str(p)}
                for p in sorted(seq_dir.glob("*.txt"))]
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scratch", default=None,
                    help="temp dir for one block (default: <out-dir>/_scratch)")
    ap.add_argument("--limit-blocks", type=int, default=None,
                    help="process only the first N .asm blocks (for timing)")
    args = ap.parse_args()

    archive = Path(args.archive).resolve()
    if not archive.exists():
        sys.exit(f"archive not found: {archive}")

    out_dir = Path(args.out_dir).resolve()
    seq_dir = out_dir / "seqs"
    seq_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch) if args.scratch else out_dir / "_scratch"

    blocks = build_block_map(archive, out_dir / "blocks.json")
    asm_blocks = [b for b in blocks if b.get("asm")]

    done = {p.stem for p in seq_dir.glob("*.txt")}
    todo = [b for b in asm_blocks
            if any(Path(m).stem not in done for m in b["asm"])]
    if done:
        print(f"resuming: {len(done):,} sequences present, "
              f"{len(todo)} of {len(asm_blocks)} blocks remain")
    if args.limit_blocks:
        todo = todo[:args.limit_blocks]
        print(f"--limit-blocks: processing {len(todo)} block(s) only")

    gb_todo = sum(b["uncompressed"] for b in todo) / 1e9
    print(f"{len(todo)} blocks to process, {gb_todo:.1f} GB to decompress\n")

    t_start = time.time()
    gb_done = 0.0
    total_records = len(done)

    for i, block in enumerate(todo, 1):
        t0 = time.time()
        recs = process_block(archive, block, scratch, seq_dir)
        dt = time.time() - t0
        gb = block["uncompressed"] / 1e9
        gb_done += gb
        total_records += len(recs)

        elapsed = time.time() - t_start
        rate = gb_done / max(elapsed, 1e-9)
        eta_h = (gb_todo - gb_done) / max(rate, 1e-9) / 3600
        print(f"  block {i:>2}/{len(todo)}  {len(recs):>4} files  "
              f"{gb:5.2f} GB in {dt:6.1f}s ({gb/max(dt,1e-9)*1000:5.0f} MB/s)  "
              f"| total {total_records:,} seqs  ETA {eta_h:5.2f} h", flush=True)

        write_manifest(seq_dir, out_dir)

    manifest = write_manifest(seq_dir, out_dir)
    counts = [r["n_opcodes"] for r in manifest]
    print(f"\nsequences: {len(manifest):,}  (BIG 2015 expects 10,868)")
    if len(manifest) != 10868 and not args.limit_blocks:
        print("  *** COUNT MISMATCH -- rerun to resume before using this. ***")
    if counts:
        print(f"total opcodes: {sum(counts):,}  "
              f"min {min(counts):,}  mean {sum(counts)/len(counts):,.0f}  "
              f"max {max(counts):,}")
    print(f"elapsed: {(time.time()-t_start)/60:.1f} min")
