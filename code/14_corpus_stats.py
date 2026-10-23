"""
Step 14 -- corpus statistics quoted in Sections 4.1 and 5.1 of the article.

Reads the extracted sequences (manifests from 01_extract_opcodes.py /
01b_stream_opcodes_from_archive.py) and the BIG 2015 labels, and writes every
corpus-level count the article states, so none of them is carried by hand:

  * malware: samples, excluded samples per family, instruction count,
    distinct tokens, and how many of those tokens Capstone's x86 instruction
    table (or an IDA spelling of one) does not recognise, with their mass;
  * benign (results/benign_matched, the extraction behind the benign fits of
    Table 2 and Section 7.1): files, tokens, distinct mnemonics, per-file
    median/mean/range, nop share;
  * benign_nop_dropped: a separate benign extraction without nop, read from
    <results>/benign/ (manifest.json and powerlaw_3gram.json). It is not
    used in the article -- the nop sensitivity of Section 7.3 comes from
    19_benign_nop_refit.py -- and its inputs are not shipped, so this part
    needs those files to run.

Usage
-----
    python 14_corpus_stats.py --results ../results --labels /path/to/trainLabels.csv

Output
------
    <results>/corpus_stats.json
"""

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path

import pandas as pd

FAMILY = {1: "Ramnit", 2: "Lollipop", 3: "Kelihos_ver3", 4: "Vundo",
          5: "Simda", 6: "Tracur", 7: "Kelihos_ver1",
          8: "Obfuscator.ACY", 9: "Gatak"}

# IDA Pro spells several instructions differently from Capstone (jz/je,
# retn/ret, string ops without size suffix, ...). These are mnemonics, not
# extraction residue, so they are not counted as unrecognised.
IDA_SPELLINGS = {
    "retn", "retf", "retnw", "retfw", "jz", "jnz", "jnb", "jnbe", "jna",
    "jnae", "jnl", "jnle", "jng", "jnge", "setz", "setnz", "setnb", "setnbe",
    "setnl", "setnle", "cmovz", "cmovnz", "cmovnb", "cmovnbe", "lods", "stos",
    "movs", "cmps", "scas", "outs", "ins", "pushf", "popf", "pusha", "popa",
    "pushfw", "popfw", "iretw", "enterw", "leavew", "loopw", "loopwe",
    "loopwne", "fcmovnbe", "fcmovnb", "fcmovnu", "fcmovne", "repe", "repne",
    "setalc", "xlat", "faddp", "fstsw", "fstcw", "fstenv", "fsave", "fclex",
    "fndisi", "fneni", "fucomip", "fcomip",
}


def read_manifest(path):
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    counts, lengths = Counter(), []
    for r in records:
        toks = Path(r["seq_file"]).read_text(encoding="utf-8").split()
        counts.update(toks)
        lengths.append(len(toks))
    return records, counts, lengths


def capstone_names():
    import capstone.x86_const as k
    return {n[len("X86_INS_"):].lower() for n in dir(k)
            if n.startswith("X86_INS_")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--labels", required=True)
    args = ap.parse_args()
    res = Path(args.results)
    out = {}

    # ---- malware ----------------------------------------------------------
    recs, counts, lengths = read_manifest(res / "malware" / "manifest.json")
    names = {r["name"] for r in recs}
    lbl = pd.read_csv(args.labels)
    lbl["excluded"] = ~lbl["Id"].astype(str).isin(names)
    per_fam = {FAMILY[c]: {"total": int(len(g)),
                           "excluded": int(g["excluded"].sum())}
               for c, g in lbl.groupby("Class")}
    known = capstone_names() | IDA_SPELLINGS
    unknown = {t: v for t, v in counts.items() if t not in known}
    total = sum(counts.values())
    out["malware"] = {
        "labelled_samples": int(len(lbl)),
        "samples_with_instructions": len(recs),
        "excluded_samples": int(lbl["excluded"].sum()),
        "excluded_pct_of_corpus": round(100 * lbl["excluded"].mean(), 2),
        "per_family": per_fam,
        "instructions": total,
        "distinct_tokens": len(counts),
        "unrecognised_distinct": len(unknown),
        "unrecognised_tokens": sum(unknown.values()),
        "unrecognised_pct": round(100 * sum(unknown.values()) / total, 4),
        "unrecognised_top": sorted(unknown.items(), key=lambda x: -x[1])[:25],
    }

    # ---- benign, nop retained (the extraction behind the reported fits) ---
    _, counts, lengths = read_manifest(res / "benign_matched" / "manifest.json")
    out["benign_matched"] = {
        "files": len(lengths),
        "tokens": sum(lengths),
        "distinct_mnemonics": len(counts),
        "median_per_file": st.median(lengths),
        "mean_per_file": round(st.mean(lengths), 1),
        "min_per_file": min(lengths),
        "max_per_file": max(lengths),
        "nop_pct": round(100 * counts.get("nop", 0) / sum(lengths), 2),
    }

    # ---- benign, nop dropped (separate extraction) ----------------------
    _, counts, lengths = read_manifest(res / "benign" / "manifest.json")
    fit3 = json.loads((res / "benign" / "powerlaw_3gram.json")
                      .read_text(encoding="utf-8"))
    out["benign_nop_dropped"] = {
        "tokens": sum(lengths),
        "distinct_mnemonics": len(counts),
        "3gram_xmin": fit3["xmin"],
        "3gram_retention_pct": fit3["retention_pct"],
    }

    p = res / "corpus_stats.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
