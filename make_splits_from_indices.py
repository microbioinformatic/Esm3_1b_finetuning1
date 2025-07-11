#!/usr/bin/env python3
import os
import csv
import argparse
import numpy as np

def gen_split(csv_path, split_dir, out_dir, fold):
    # mapping from expected CSV basename to index-file prefix
    mapping = {
        "train":      "train_index",
        "validation": "val_index",
        "test":       "test_index",
    }

    # read full dataset once
    with open(csv_path, newline="", encoding="utf8") as rf:
        reader = list(csv.reader(rf))
    if not reader:
        raise ValueError(f"Empty master CSV: {csv_path}")
    header, rows = reader[0], reader[1:]

    os.makedirs(out_dir, exist_ok=True)
    for split_name, idx_prefix in mapping.items():
        idx_path = os.path.join(split_dir, f"{idx_prefix}_{fold}.npy")
        if not os.path.isfile(idx_path):
            raise FileNotFoundError(f"Missing index file: {idx_path}")
        indices = np.load(idx_path).astype(int).tolist()

        out_csv = os.path.join(out_dir, f"{split_name}_fold{fold}.csv")
        with open(out_csv, "w", newline="", encoding="utf8") as wf:
            writer = csv.writer(wf)
            writer.writerow(header)
            for i in indices:
                writer.writerow(rows[i])
        print(f"Wrote {len(indices)} rows → {out_csv}")

def main():
    p = argparse.ArgumentParser(
        description="Generate train/validation/test CSVs from master CSV + .npy indices"
    )
    p.add_argument("--csv_path",   required=True,
                   help="Master CSV with columns: header,sequence,GO_terms")
    p.add_argument("--split_dir",  required=True,
                   help="Folder containing train_index_*.npy, val_index_*.npy, test_index_*.npy")
    p.add_argument("--out_dir",    required=True,
                   help="Where to write train_fold{f}.csv etc (e.g. outputs/splits)")
    p.add_argument("--fold",       type=int, required=True,
                   help="Fold number (0–4)")
    args = p.parse_args()

    gen_split(args.csv_path, args.split_dir, args.out_dir, args.fold)

if __name__ == "__main__":
    main()

