#!/usr/bin/env python3
import os, csv
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import esm
from Bio.PDB import PDBParser, PPBuilder
from Bio import pairwise2
from Bio.PDB.Polypeptide import three_to_one

# The 37 atom names as ESM expects
ATOM37 = [
    "N","CA","C","O","CB","CG","CG2","OG","OG1","SG",
    "CD","CD1","CD2","ND1","ND2","OD1","OD2","CE","CE1",
    "CE2","CZ","OH","NE","NH1","NH2","CZ2","CZ3","CH2",
    "OXT","H","HA","HB","1HG","2HG","3HG"
]

class SalmonellaMultiLabelDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        go_vocab=None,
        max_len: int = 1024,
        structure_dir: str = None,
        split_dir: str = None,
        split_type: str = None,  # "train","val","test"
        fold: int = None,
    ):
        if structure_dir is None:
            raise ValueError("Must pass --struct_dir to load PDBs")
        self.structure_dir = structure_dir

        # 1) Read CSV and subset by split indices
        raw = []
        with open(csv_path, encoding="utf8") as f:
            rdr = csv.DictReader(f, delimiter=",")
            for row in rdr:
                hdr = row["header"].strip()
                seq = row["sequence"].strip()
                gos = [g.strip() for g in row["GO_terms"].split(";") if g.strip()]
                if hdr and seq and gos and len(seq)<=max_len:
                    raw.append((hdr, seq, gos))
        if split_dir and split_type and fold is not None:
            idx_file = os.path.join(split_dir, f"{split_type}_index_{fold}.npy")
            inds = np.load(idx_file).astype(int).tolist()
            self.records = [raw[i] for i in inds]
        else:
            self.records = raw

        # 2) GO‐term vocab
        all_gos = set(g for _,_,gos in self.records for g in gos)
        self.go_vocab = sorted(all_gos) if go_vocab is None else list(go_vocab)
        self.go_to_idx = {g:i for i,g in enumerate(self.go_vocab)}
        self.num_labels = len(self.go_vocab)
        print(f"Dataset: {len(self.records)} seqs; {self.num_labels} GO terms")

        # 3) ESM tokenizer
        _, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.batch_converter = self.alphabet.get_batch_converter()

        # 4) BioPython PDB tools
        self.pdb_parser = PDBParser(QUIET=True)
        self.ppb = PPBuilder()

        tokens, lengths, coords, targets = [], [], [], []
        for hdr, seq, gos in self.records:
            # ─── Tokenize ─────────────────────────────────────
            _, tok_tensor, _ = self.batch_converter([(hdr, seq)])
            tok = tok_tensor.squeeze(0)
            tokens.append(tok)
            lengths.append(torch.tensor(tok.size(0),dtype=torch.long))

            # ─── Parse PDB & extract backbone coords ─────────
            pdb_fn = os.path.join(self.structure_dir, f"{hdr}.pdb")
            if not os.path.isfile(pdb_fn):
                raise FileNotFoundError(f"PDB not found: {pdb_fn}")
            stru = self.pdb_parser.get_structure(hdr, pdb_fn)
            # assume single model & chain
            model = next(stru.get_models())
            chain = next(model.get_chains())

            # build PDB one-letter sequence & residue list
            seqres = []
            residues = []
            for pp in self.ppb.build_peptides(chain):
                seqres.extend(list(pp.get_sequence()))
                residues.extend(pp)
            pdb_seq = "".join(seqres)

            # align pdb_seq ↔ raw seq
            aln = pairwise2.align.globalxx(seq, pdb_seq, one_alignment_only=True)[0]
            seq_aln, pdb_aln = aln.seqA, aln.seqB

            # map sequence positions → PDB residue index
            seq_to_pdb = {}
            si = pi = 0
            for a,b in zip(seq_aln, pdb_aln):
                if a!="-":
                    if b!="-":
                        seq_to_pdb[si] = pi
                        pi += 1
                    si += 1
                else:
                    if b!="-":
                        pi += 1

            # build coords in seq order
            mat_list = []
            for i in range(len(seq)):
                pidx = seq_to_pdb.get(i)
                atom_map = {}
                if pidx is not None:
                    res = residues[pidx]
                    atom_map = {a.get_name(): a.get_coord() for a in res.get_atoms()}
                # build [37,3]
                mat = [atom_map.get(n, np.zeros(3,dtype=np.float32)) for n in ATOM37]
                mat_list.append(mat)
            coords.append(torch.tensor(mat_list,dtype=torch.float32))

            # ─── GO targets ───────────────────────────────────
            tgt = torch.zeros(self.num_labels, dtype=torch.float32)
            for g in gos:
                tgt[self.go_to_idx[g]] = 1.0
            targets.append(tgt)

        # 5) Pad everything
        pad_idx = self.alphabet.padding_idx
        self.tokens = pad_sequence(tokens, batch_first=True, padding_value=pad_idx)
        self.lengths = torch.stack(lengths)
        self.coords = pad_sequence(coords, batch_first=True, padding_value=0.0)
        self.targets = torch.stack(targets)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return (
            self.tokens[i],
            self.lengths[i],
            self.coords[i],    # FloatTensor [L,37,3]
            self.targets[i],   # FloatTensor [num_labels]
        )

