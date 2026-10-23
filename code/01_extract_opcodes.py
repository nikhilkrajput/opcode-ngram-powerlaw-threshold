"""
Step 1 -- Opcode extraction.

Two corpora, one extraction contract: every sample becomes a single
space-separated string of lowercase instruction mnemonics.

  (a) BIG 2015 malware:  parse the IDA Pro .asm listings shipped with the
      dataset.  This is the exact regular expression used to produce the
      counts reported in the article (Section 4.1).

  (b) Benign reference:  Windows System32 executables have no IDA listing,
      so they are disassembled with pefile + capstone: the first 200
      executables, in alphabetical order, that yield at least 50
      instructions (Section 5.1).

Usage
-----
    python 01_extract_opcodes.py malware --asm-dir  /path/to/BIG2015/train
    python 01_extract_opcodes.py benign  --sys32    C:/Windows/System32 \
                                         --out-dir  ../results/benign_matched

The benign corpus used in the article lives in results/benign_matched/, which
is where the later scripts look for it; pass --out-dir accordingly (the
default for `benign` is ../results/benign).  For BIG 2015 as distributed
(one solid train.7z archive), 01b_stream_opcodes_from_archive.py produces the
same output without unpacking the archive.

Output
------
    <out-dir>/seqs/<sample-id>.txt   one opcode sequence per sample
    <out-dir>/manifest.json          [{name, n_opcodes, seq_file}, ...]
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opcode_filter import DIRECTIVES, filter_opcodes  # noqa: E402

# ---------------------------------------------------------------------------
# (a) BIG 2015 -- IDA Pro .asm listings
# ---------------------------------------------------------------------------
# Matches "whitespace, two hex digits, optional + continuation, whitespace,
# lowercase word".  IDA lays data lines out exactly like instruction lines, so
# this ALSO matches `db`/`dd`/`dw`; every capture must go through
# filter_opcodes().  See opcode_filter.py.
OPCODE_RGX = re.compile(r"\s[\dA-F]{2}(?:\+)?\s+([a-z]+)\s")


def extract_from_asm(path):
    """Return the opcode sequence of one IDA Pro .asm listing."""
    with open(path, "r", encoding="latin-1", errors="ignore") as fh:
        text = fh.read()
    return filter_opcodes(OPCODE_RGX.findall(text))


def run_malware(asm_dir, out_dir, limit=None):
    asm_files = sorted(glob.glob(os.path.join(asm_dir, "*.asm")))
    if not asm_files:
        sys.exit(f"No .asm files found under {asm_dir}")
    print(f"Found {len(asm_files)} .asm listings")

    seq_dir = out_dir / "seqs"
    seq_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for i, path in enumerate(asm_files):
        if limit and len(manifest) >= limit:
            break
        sample_id = Path(path).stem
        opcodes = extract_from_asm(path)
        if not opcodes:
            continue
        seq_path = seq_dir / f"{sample_id}.txt"
        seq_path.write_text(" ".join(opcodes), encoding="utf-8")
        manifest.append({
            "name": sample_id,
            "n_opcodes": len(opcodes),
            "seq_file": str(seq_path),
        })
        if (i + 1) % 500 == 0:
            print(f"  {i + 1} / {len(asm_files)} ...")

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done: {len(manifest)} malware sequences -> {out_dir}")


# ---------------------------------------------------------------------------
# (b) Benign -- System32 PE files via pefile + capstone
# ---------------------------------------------------------------------------
# capstone emits genuine instructions, so the directive list barely applies;
# `data16` is an operand-size prefix rendered as a mnemonic.  `nop` is a real
# instruction and is KEPT -- dropping it here but not in the malware corpus
# would make the two incomparable.  See opcode_filter.py.
JUNK_MNEMONICS = {"data16"} | DIRECTIVES


def extract_from_pe(path, md_x86, md_x64, cap=200_000):
    """Return the opcode sequence of one PE file, or None if unusable."""
    import pefile

    try:
        pe = pefile.PE(path, fast_load=False)
    except Exception:
        return None

    machine = pe.FILE_HEADER.Machine
    if machine == 0x8664:
        md = md_x64
    elif machine == 0x14C:
        md = md_x86
    else:
        pe.close()
        return None  # skip ARM / MIPS / anything non-x86

    opcodes = []
    for section in pe.sections:
        if not (section.Characteristics & 0x20000000):  # IMAGE_SCN_MEM_EXECUTE
            continue
        data = section.get_data()
        vaddr = section.VirtualAddress + pe.OPTIONAL_HEADER.ImageBase
        for insn in md.disasm(data, vaddr):
            mn = insn.mnemonic.split()[0].lower()
            if mn and mn not in JUNK_MNEMONICS:
                opcodes.append(mn)
        if len(opcodes) > cap:
            break
    pe.close()
    return opcodes if len(opcodes) >= 50 else None


def run_benign(sys32, out_dir, limit=200):
    import capstone

    md_x86 = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
    md_x64 = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md_x86.detail = md_x64.detail = False

    all_exes = sorted(glob.glob(os.path.join(sys32, "*.exe")))
    print(f"Found {len(all_exes)} EXEs in {sys32}")

    seq_dir = out_dir / "seqs"
    seq_dir.mkdir(parents=True, exist_ok=True)
    manifest, failed = [], 0

    for path in all_exes:
        if len(manifest) >= limit:
            break
        opcodes = extract_from_pe(path, md_x86, md_x64)
        if opcodes is None:
            failed += 1
            continue
        name = os.path.basename(path)
        seq_path = seq_dir / (name.replace(".exe", "") + ".txt")
        seq_path.write_text(" ".join(opcodes), encoding="utf-8")
        manifest.append({
            "name": name,
            "n_opcodes": len(opcodes),
            "seq_file": str(seq_path),
        })

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    counts = [r["n_opcodes"] for r in manifest]
    print(f"Done: {len(manifest)} benign sequences, {failed} skipped")
    if counts:
        counts_sorted = sorted(counts)
        mid = len(counts_sorted) // 2
        median = (counts_sorted[mid] if len(counts_sorted) % 2
                  else (counts_sorted[mid - 1] + counts_sorted[mid]) / 2)
        print(f"  opcodes/file: min={min(counts)} max={max(counts)} "
              f"mean={sum(counts) / len(counts):.1f} median={median}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", choices=["malware", "benign"])
    ap.add_argument("--asm-dir", help="directory of BIG 2015 .asm listings")
    ap.add_argument("--sys32", default="C:/Windows/System32")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    default_out = "../results/malware" if args.corpus == "malware" else "../results/benign"
    out_dir = Path(args.out_dir or default_out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.corpus == "malware":
        if not args.asm_dir:
            sys.exit("--asm-dir is required for the malware corpus")
        run_malware(args.asm_dir, out_dir, args.limit)
    else:
        run_benign(args.sys32, out_dir, args.limit or 200)
