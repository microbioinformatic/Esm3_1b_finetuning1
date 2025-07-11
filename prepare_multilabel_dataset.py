#File: scripts\prepare_multilabel_dataset.py
#The prepare_multilabel_dataset.py script expects to read its input FASTA from
#outputs\salmonella_cleaned_final.fa and data\salmonella_annotations.tsv and generate outputs\multilabel_dataset.csv
#and runs 3rd on the scripts

import os
import csv
from collections import defaultdict

# ----- PATHS (relative to project root) -----
CLEANED_FASTA = r"outputs\salmonella_cleaned_final.fa"
ANNOT_PATH    = r"data\salmonella_annotations.tsv"
OUT_CSV       = r"outputs\multilabel_dataset.csv"
# ---------------------------------------------

# 1) Read GO annotations: build header → [GO list]
header2gos = defaultdict(list)
all_go_terms = set()
with open(ANNOT_PATH, "r", encoding="utf8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    if "Header" not in reader.fieldnames or "GO" not in reader.fieldnames:
        raise ValueError("Annotation TSV must have 'Header' and 'GO' columns.")
    for row in reader:
        hdr = row["Header"].strip()
        raw = row["GO"].strip()
        if raw == "":
            continue
        gos = [g.strip() for g in raw.split(";") if g.strip()]
        header2gos[hdr] = gos
        all_go_terms.update(gos)

print(f"Total unique headers with GO terms: {len(header2gos)}")
print(f"Total distinct GO terms found: {len(all_go_terms)}")

# 2) Read cleaned FASTA and keep only sequences present in header2gos
seq_records = []  # list of (header, sequence)
with open(CLEANED_FASTA, "r", encoding="utf8") as f:
    curr_hdr = None
    buffer = []
    for line in f:
        line = line.rstrip("\n")
        if line.startswith(">"):
            if curr_hdr:
                seq = "".join(buffer).upper()
                seq_records.append((curr_hdr, seq))
            curr_hdr = line[1:].split()[0]
            buffer = []
        else:
            buffer.append(line)
    # last record
    if curr_hdr:
        seq = "".join(buffer).upper()
        seq_records.append((curr_hdr, seq))

print(f"Total sequences in cleaned FASTA: {len(seq_records)}")

# 3) Filter to only those headers present in header2gos
filtered = [(h, s) for (h, s) in seq_records if h in header2gos]
print(f"Sequences after matching to annotations: {len(filtered)}")

# 4) Write out a CSV with columns: header,sequence,semicolon‐joined‐GO
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf8") as out_csv:
    writer = csv.writer(out_csv)
    writer.writerow(["header", "sequence", "GO_terms"])
    for hdr, seq in filtered:
        gos = header2gos.get(hdr, [])
        if not gos:
            continue
        writer.writerow([hdr, seq, ";".join(gos)])

print(f"Wrote {len(filtered)} rows to {OUT_CSV}")
