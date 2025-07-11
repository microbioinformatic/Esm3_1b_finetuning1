# File: scripts\clean_and_prepare_final.py

import os
import sys

# ----- CONFIGURE PATHS RELATIVE TO PROJECT ROOT -----
ANNOT_PATH = r"data\salmonella_annotations.tsv"
FASTA_PATH  = r"data\salmonella_sequences.fa"
CLEANED_FASTA_OUT = r"outputs\salmonella_cleaned_final.fa"
MIN_LEN = 50
MAX_LEN = 1024
# ----------------------------------------------------

# 1. Read annotation headers
annot_headers = set()
with open(ANNOT_PATH, "r", encoding="utf8") as f:
    header_line = f.readline().rstrip("\n").split("\t")
    if "Header" not in header_line:
        print("ERROR: 'Header' column not found in annotation file.")
        sys.exit(1)
    hdr_idx = header_line.index("Header")
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) > hdr_idx:
            annot_headers.add(parts[hdr_idx])

print(f"Loaded {len(annot_headers)} unique headers from annotations.")

# 2. Parse FASTA, filter & dedupe
seq_to_headers = {}
current_header = None
buffer = []

with open(FASTA_PATH, "r", encoding="utf8") as f:
    for line in f:
        line = line.rstrip("\r\n")
        if line.startswith(">"):
            if current_header is not None:
                seq = "".join(buffer).upper()
                # Filtering conditions:
                # (a) no 'U'
                # (b) only valid AA letters
                # (c) length between MIN_LEN and MAX_LEN
                # (d) header must be in annotations
                if (
                    "U" not in seq
                    and all(c in "ACDEFGHIKLMNPQRSTVWYBXZ" for c in seq)
                    and (MIN_LEN <= len(seq) <= MAX_LEN)
                    and (current_header in annot_headers)
                ):
                    seq_to_headers.setdefault(seq, []).append(current_header)
            buffer = []
            current_header = line[1:].split()[0]
        else:
            buffer.append(line)

    # Last record
    if current_header is not None:
        seq = "".join(buffer).upper()
        if (
            "U" not in seq
            and all(c in "ACDEFGHIKLMNPQRSTVWYBXZ" for c in seq)
            and (MIN_LEN <= len(seq) <= MAX_LEN)
            and (current_header in annot_headers)
        ):
            seq_to_headers.setdefault(seq, []).append(current_header)

# Deduplicate: keep only the first header for each unique sequence
unique_seqs = [(hdrs[0], seq) for seq, hdrs in seq_to_headers.items()]
print(f"Sequences after filtering & deduplication: {len(unique_seqs)}")

# 3. Write cleaned FASTA
os.makedirs(os.path.dirname(CLEANED_FASTA_OUT), exist_ok=True)
with open(CLEANED_FASTA_OUT, "w", encoding="utf8") as out_f:
    for hdr, seq in unique_seqs:
        out_f.write(f">{hdr}\n")
        for i in range(0, len(seq), 80):
            out_f.write(seq[i : i + 80] + "\n")

print(f"Cleaned FASTA written to: {CLEANED_FASTA_OUT}")
