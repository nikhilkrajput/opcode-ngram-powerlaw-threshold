"""
Step 21 -- measure the block-1 facts quoted in Sections 4.1 and 5.1.

This script derives the 96.2% directive share (also tabulated in the
docstring of opcode_filter.py), and two further facts the article states,
from the raw IDA listings of the first solid block of train.7z:

  * raw captures of the extraction regex, and the share that are directives;
  * how IDA writes the one-byte breakpoint (does "int 3" occur?);
  * the segment bitness IDA declares (use32 / use64).

Usage
-----
    python 21_block1_inventory.py --archive /path/to/BIG2015/train.7z \
        --blocks ../results/malware/blocks.json --scratch /path/to/scratch \
        --out ../results/malware/block1_inventory.json
"""

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import py7zr

from opcode_filter import DIRECTIVES

OPCODE_RGX = re.compile(r"\s[\dA-F]{2}(?:\+)?\s+([a-z]+)\s")   # as in 01/01b
INT3_RGX = re.compile(r"\s[\dA-F]{2}(?:\+)?\s+int\s+3\s")
SEG_RGX = re.compile(r"\b(use32|use64)\b")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    block = json.loads(Path(args.blocks).read_text())[0]
    scratch = Path(args.scratch)
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    with py7zr.SevenZipFile(args.archive, mode="r") as z:
        z.extract(path=str(scratch), targets=block["asm"])

    caps, int3, seg = Counter(), 0, Counter()
    files = 0
    for path in scratch.rglob("*.asm"):
        text = path.read_text(encoding="latin-1", errors="ignore")
        caps.update(OPCODE_RGX.findall(text))
        int3 += len(INT3_RGX.findall(text))
        seg.update(set(SEG_RGX.findall(text)))
        files += 1
        path.unlink()
    shutil.rmtree(scratch, ignore_errors=True)

    total = sum(caps.values())
    dir_tok = sum(v for k, v in caps.items() if k in DIRECTIVES)
    four = sum(caps.get(k, 0) for k in ("db", "dd", "dw", "align"))
    out = {"files": files, "captured_tokens": total,
           "directive_tokens": dir_tok,
           "directive_pct": round(100 * dir_tok / total, 2),
           "db_dd_dw_align_pct": round(100 * four / total, 2),
           "int_3_lines": int3,
           "files_declaring_use32": seg.get("use32", 0),
           "files_declaring_use64": seg.get("use64", 0),
           "top_captures": caps.most_common(12)}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
