# File: scripts\group_labels.py

import pandas as pd
import os

# Paths (relative to E:\leyden finetunning esm3)
IN_TSV  = r"data\Salmonella_annotations.tsv"          # your original file
OUT_TSV = r"data\salmonella_annotations_grouped.tsv" # new, grouped file

# 1. Read the original TSV (Header + Label)
df = pd.read_csv(IN_TSV, sep="\t", dtype=str)

# 2. Group by Header, collect unique GO IDs, sort, and join with semicolons
grouped = (
    df
    .dropna(subset=["Header", "Label"])            # drop any empty lines
    .groupby("Header")["Label"]
    .apply(lambda go_list: ";".join(sorted(set(go_list))))
    .reset_index()
    .rename(columns={"Label": "GO"})
)

# 3. Write out the new TSV
os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)
grouped.to_csv(OUT_TSV, sep="\t", index=False)

print(f"Written grouped annotations to: {OUT_TSV}")
print(f"Total headers: {len(grouped)}; example rows:\n")
print(grouped.head(5).to_string(index=False))
