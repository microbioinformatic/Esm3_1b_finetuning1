# File: scripts/dataset_multilabel.py

import csv
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import esm  # from fair-esm

class SalmonellaMultiLabelDataset(Dataset):
    def __init__(self, csv_path, go_vocab=None, max_len=1024):
        """
        Tokenize each sequence individually (with ESM-1b), then pad all to the same length.
        __getitem__ returns (tokens, lengths, targets) as torch.Tensors.

        Args:
            csv_path (str): Path to CSV with columns: header, sequence, GO_terms
            go_vocab (iterable[str], optional): pre-defined list of GO terms to use.
                If None, builds vocab from this split.
            max_len (int): maximum sequence length to include.
        """
        self.records = []   # each entry: (header, sequence, [GO terms])
        all_gos = set()

        # 1) Read CSV and collect sequences + GO lists
        with open(csv_path, "r", encoding="utf8") as f:
            reader = csv.DictReader(f, delimiter=",")
            required = {"header", "sequence", "GO_terms"}
            if not required.issubset(reader.fieldnames):
                raise ValueError(f"CSV must have columns: {required}")
            for row in reader:
                hdr = row["header"].strip()
                seq = row["sequence"].strip()
                gos = [g.strip() for g in row["GO_terms"].split(";") if g.strip()]
                if not seq or not gos or len(seq) > max_len:
                    continue
                self.records.append((hdr, seq, gos))
                all_gos.update(gos)

        # 2) Build GO‐term vocabulary (global if provided, else from this split)
        if go_vocab is None:
            self.go_vocab = sorted(all_gos)
        else:
            self.go_vocab = list(go_vocab)
        self.go_to_idx = {go: i for i, go in enumerate(self.go_vocab)}
        self.num_labels = len(self.go_vocab)

        print(f"Dataset: {len(self.records)} sequences; {self.num_labels} distinct GO terms")

        # 3) Load ESM‐1b alphabet and batch_converter
        _, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.batch_converter = self.alphabet.get_batch_converter()

        # 4) Tokenize each sequence individually, collecting token + length tensors
        token_tensors = []
        length_tensors = []
        for hdr, seq, _ in self.records:
            data = [(hdr, seq)]
            labels, tokens, lengths = self.batch_converter(data)

            # ─── Handle `tokens` ───
            if isinstance(tokens, torch.Tensor):
                # Normal case: Tensor of shape [1, L]
                tok1 = tokens.squeeze(0)   # → Tensor [L]
            else:
                # tokens is a Python list with one element
                first_row = tokens[0]

                # Case A: first_row is a list of token IDs
                if isinstance(first_row, list):
                    tok_ids = []
                    for x in first_row:
                        if isinstance(x, str):
                            try:
                                tok_ids.append(int(x))
                            except ValueError:
                                raise ValueError(
                                    f"Cannot convert token '{x}' to int in list {first_row[:5]}..."
                                )
                        else:
                            tok_ids.append(x)
                    tok1 = torch.tensor(tok_ids, dtype=torch.long)

                # Case B: first_row is a raw sequence string ("MAGLK…")
                elif isinstance(first_row, str):
                    tok_ids = []
                    for char in first_row:
                        idx = self.alphabet.get_idx(char)
                        if idx is None:
                            raise ValueError(f"Unknown token '{char}' in sequence '{first_row[:10]}...'")
                        tok_ids.append(idx)
                    tok1 = torch.tensor(tok_ids, dtype=torch.long)

                else:
                    raise ValueError(f"Unexpected tokens[0] type: {type(first_row)}")

            # record length
            len1 = torch.tensor(tok1.size(0), dtype=torch.long)

            token_tensors.append(tok1)
            length_tensors.append(len1)

        # 5) Pad all token tensors to the same length (max over dataset)
        padding_idx = self.alphabet.padding_idx
        self.tokens = pad_sequence(token_tensors, batch_first=True, padding_value=padding_idx)
        self.lengths = torch.stack(length_tensors)

        # 6) Build multi‐hot target tensor of shape (N, num_labels)
        N = len(self.records)
        targets = torch.zeros((N, self.num_labels), dtype=torch.float32)
        for i, (_, _, gos) in enumerate(self.records):
            for go in gos:
                idx = self.go_to_idx.get(go)
                if idx is not None:
                    targets[i, idx] = 1.0
        self.targets = targets

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        """
        Returns:
            tokens[idx]: LongTensor of size (L,)   (already padded)
            lengths[idx]: LongTensor scalar
            targets[idx]: FloatTensor of size (num_labels,)
        """
        return self.tokens[idx], self.lengths[idx], self.targets[idx]
