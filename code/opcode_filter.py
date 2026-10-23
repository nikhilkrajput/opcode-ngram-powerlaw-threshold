"""
Shared opcode extraction rules. Both extractors import from here so the two
corpora cannot drift apart -- an asymmetry between them silently invalidates
the malware-vs-benign comparison (Section 7.1 and Table 2 of the article).

THE PROBLEM THIS FIXES
----------------------
IDA Pro listings put data and instructions in the same column layout:

    .text:10001007 64 A1 00 00 00 00      mov     eax, large fs:0
    .text:10001310 CC                     db 0CCh
    .text:1000143E C7 74                  dw 74C7h

The extraction regex matches "whitespace, two hex digits, whitespace, lowercase
word", which describes both lines equally well. So `db` and `dd` -- assembler
data directives, not instructions -- were captured as opcodes.

Measured on the first solid block of BIG 2015 (123 samples, 46,749,564 tokens):

    db        42,989,109   91.96%
    dd         1,909,129    4.08%
    align         37,277    0.08%
    dw            17,246    0.04%
    unicode        2,344    0.01%
    extrn/end/offset  36    0.00%
    ------------------------------
    directives 44,955,141   96.16%

96% of the "opcode n-grams" were n-grams of data-section padding. Filtering
leaves 1,794,423 instruction tokens over 286 distinct mnemonics -- close to the
289 that capstone produces for the benign System32 corpus (Section 5.1), which
is the independent check that this list is neither too aggressive nor too lax.

NOTE ON `nop`
-------------
`nop` is a genuine x86 instruction and is KEPT on both sides. Dropping it from
one corpus only would be a second, smaller asymmetry between the corpora.
"""

# Assembler directives and IDA layout artefacts. These are not instructions.
# Derived by inventorying every distinct token in a full solid block and
# classifying it, not guessed -- see the counts above.
DIRECTIVES = frozenset({
    # data definition
    "db", "dd", "dw", "dq", "dt", "df", "do", "dup",
    # alignment and layout
    "align", "org", "even",
    # size and type keywords
    "byte", "word", "dword", "qword", "tbyte", "fword", "oword",
    "xmmword", "ymmword", "zmmword", "ptr", "short", "near", "far",
    "large", "small", "flat", "offset",
    # segment and procedure structure
    "segment", "ends", "proc", "endp", "assume", "end", "group",
    # linkage
    "public", "extrn", "extern", "global", "include", "includelib",
    # misc IDA / MASM
    "equ", "unicode", "model", "use16", "use32", "title", "subttl",
    "comment", "if", "else", "endif", "macro", "endm", "rept", "irp",
})


def is_instruction(token):
    """True if the token is a plausible instruction mnemonic."""
    return token not in DIRECTIVES


def filter_opcodes(tokens):
    """Drop assembler directives, keep instruction mnemonics (including nop)."""
    return [t for t in tokens if t not in DIRECTIVES]


# Mnemonics the benign (capstone) extractor discards. capstone only emits real
# instructions, so the only genuine directive it can produce is `data16` (an
# operand-size prefix IDA renders differently). `nop` is retained on both
# sides.
BENIGN_ONLY_DROP = frozenset({"data16"})
