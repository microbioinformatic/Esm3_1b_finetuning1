# File: clean_multilabel_dataset.py

import csv
from pathlib import Path

def load_annotations(tsv_path):
    """
    Expects a TSV with at least columns:
      - sequence_id (or header)
      - GO_terms (semicolon-separated)
    Returns a dict: { sequence_id: [go1, go2, ...] }
    """
    annots = {}
    with open(tsv_path, "r", encoding="utf8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "sequence_id" not in reader.fieldnames or "GO_terms" not in reader.fieldnames:
            raise ValueError("TSV must have columns: sequence_id, GO_terms")
        for row in reader:
            sid = row["sequence_id"].strip()
            gos = [g.strip() for g in row["GO_terms"].split(";") if g.strip()]
            annots[sid] = gos
    return annots

def load_fasta(fasta_path):
    """
    Parses a FASTA file and returns a dict: { header (string after '>'): sequence_string }.
    """
    seqs = {}
    with open(fasta_path, "r", encoding="utf8") as f:
        curr_header = None
        curr_seq = []
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if curr_header is not None:
                    seqs[curr_header] = "".join(curr_seq)
                curr_header = line[1:].split()[0]  # take the first token as the ID
                curr_seq = []
            else:
                curr_seq.append(line.strip())
        # last one
        if curr_header is not None:
            seqs[curr_header] = "".join(curr_seq)
    return seqs

def write_cleaned_csv(fasta_dict, annots_dict, output_csv, min_terms=3):
    """
    Writes a CSV with columns: header,sequence,GO_terms
    Only include those sequences whose GO_terms list length >= min_terms.
    GO_terms in CSV will be semicolon-separated.
    """
    with open(output_csv, "w", newline="", encoding="utf8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["header", "sequence", "GO_terms"])
        kept = 0
        for header, seq in fasta_dict.items():
            if header not in annots_dict:
                continue
            gos = annots_dict[header]
            if len(gos) < min_terms:
                continue
            writer.writerow([header, seq, ";".join(gos)])
            kept += 1
    print(f"Written {kept} records (each ≥ {min_terms} GO terms) to {output_csv}")

def main():
    # Paths (adjust if your files live elsewhere)
    tsv_path = Path("salmonella_annotations.tsv")
    fasta_path = Path("salmonella_sequences.fa")
    out_csv   = Path("multilabel_dataset_cleaned.csv")

    print("Loading annotations…")
    annots = load_annotations(tsv_path)
    print(f"  {len(annots)} sequences with at least one annotation read from TSV.")

    print("Loading FASTA…")
    seqs = load_fasta(fasta_path)
    print(f"  {len(seqs)} sequences read from FASTA.")

    print("Writing cleaned CSV (min_terms=3)…")
    write_cleaned_csv(seqs, annots, out_csv, min_terms=3)

if __name__ == "__main__":
    main()
