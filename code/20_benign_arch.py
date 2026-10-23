"""
Step 20 -- machine type of the benign corpus files (32- vs 64-bit),
quoted in Section 5.1.

Reads the PE header of each file named in the benign manifest from the
System32 directory it was extracted from.

Usage
-----
    python 20_benign_arch.py --results ../results --sys32 C:/Windows/System32

Output
------
    <results>/benign_matched/arch.json
"""

import argparse
import json
import os
from pathlib import Path

import pefile

MACHINE = {0x8664: "x64", 0x14C: "x86"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="../results")
    ap.add_argument("--sys32", default="C:/Windows/System32")
    args = ap.parse_args()
    res = Path(args.results)
    recs = json.loads((res / "benign_matched/manifest.json").read_text())
    counts = {}
    for r in recs:
        try:
            pe = pefile.PE(os.path.join(args.sys32, r["name"]), fast_load=True)
            k = MACHINE.get(pe.FILE_HEADER.Machine, hex(pe.FILE_HEADER.Machine))
            pe.close()
        except Exception:
            k = "missing"
        counts[k] = counts.get(k, 0) + 1
    (res / "benign_matched/arch.json").write_text(json.dumps(counts, indent=1))
    print(counts)


if __name__ == "__main__":
    main()
