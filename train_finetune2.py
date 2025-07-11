#!/usr/bin/env python3
"""train_finetune.py
Fine‑tune ESM‑1b on GO‑term prediction **with coordinates** (structure auxiliary loss).
Assumes `dataset_multilabel_with_structures.py` returns
(tokens, lengths, coords, targets).
Relies on the vendored legacy helper `esm.utils.structure.protein_chain.ProteinChain`.
"""

import os, time, csv, argparse, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import esm  # ‑ comes from `fair‑esm` wheel + patched esm_src
from dataset_multilabel_with_structures import SalmonellaMultiLabelDataset
from clean_multilabel_dataset import load_fasta
from sklearn.metrics import average_precision_score, precision_recall_curve

# ─── Hyper‑params & paths ──────────────────────────────────────────────────
MODEL_NAME   = "esm1b_t33_650M_UR50S"
ANNOT_PATH   = "../data/salmonella_annotations.tsv"
DEFAULT_SPLIT= "/proj/nobackup/hpc2nstor2025-042/Salmonella_fp/finetunning_esm3_Giulia_data/Salmonella_splits_foldseek"
OUTPUT_DIR   = "../outputs/checkpoints_multilabel"
METRICS_CSV  = os.path.join(OUTPUT_DIR, "validation_metrics_fold{fold}.csv")

NUM_EPOCHS   = 10
BATCH_SIZE   = 16
LR_HEAD      = 1e-4
WEIGHT_DECAY = 5e-3
WARMUP_PCT   = 0.10
MAX_GRAD_NORM= 1.0
LAMBDA_STRUCT= 1.0   # weight for structure MSE loss
NUM_WORKERS  = 12
PIN_MEM      = True
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ───────────────────────────────────────────────────────────────────────────

def load_annotations_adaptive(tsv_path:str):
    ann = {}
    with open(tsv_path, newline="", encoding="utf8") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        cols = rdr.fieldnames or []
        if {"sequence_id","GO_terms"}.issubset(cols):
            idc, gc = "sequence_id", "GO_terms"
        elif {"Header","GO"}.issubset(cols):
            idc, gc = "Header", "GO"
        else:
            raise ValueError("TSV must have sequence_id/GO_terms or Header/GO")
        for r in rdr:
            sid, raw = r[idc].strip(), r[gc].strip()
            if sid and raw:
                gos = [g.strip() for g in raw.split(";") if g.strip()]
                if gos:
                    ann[sid] = gos
    print(f"[Info] Loaded {len(ann)} annotations from {tsv_path}")
    return ann

# ─── Model ────────────────────────────────────────────────────────────────

def build_model(num_labels:int):
    print(f"[Info] Loading backbone {MODEL_NAME} …")
    trunk, _ = esm.pretrained.esm1b_t33_650M_UR50S()
    trunk = trunk.to(DEVICE)
    try:
        trunk.layers.gradient_checkpointing_enable()
    except Exception:  # older fair‑esm doesn't expose this
        pass
    D = trunk.args.embed_dim

    class Net(nn.Module):
        def __init__(self, backbone, dim, n_labels):
            super().__init__()
            self.backbone = backbone
            self.cls_head = nn.Sequential(
                nn.Dropout(0.5), nn.Linear(dim, n_labels))
            # simple MLP to predict per‑token coords (L,37,3) → (L,111)
            self.coord_head = nn.Linear(dim, 37*3)
        def forward(self, tokens, lengths, coords_gt=None):
            out = self.backbone(tokens, repr_layers=[33], return_contacts=False)
            reps = out["representations"][33]        # [B,L,D]
            cls = self.cls_head(reps[:,0])            # [B,n_labels]
            pred_flat = self.coord_head(reps)         # [B,L,111]
            pred_coords = pred_flat.view(reps.size(0), reps.size(1), 37, 3)
            return cls, pred_coords

    net = Net(trunk, D, num_labels).to(DEVICE)
    return net

# ─── Helpers ──────────────────────────────────────────────────────────────

def compute_pos_weights(ds):
    pos = torch.zeros(ds.num_labels, device=DEVICE)
    for *_, tgt in ds:
        pos += tgt.to(DEVICE)
    neg = len(ds) - pos
    return (neg/(pos+1e-6)).to(DEVICE)

def ensure_metrics_file(fold:int):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = METRICS_CSV.format(fold=fold)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","val_cls","val_struct","AUPRC","Fmax"])
    return path

def compute_fmax(y_true,y_scores):
    f_vals=[]
    for i in range(y_true.shape[0]):
        p,r,_ = precision_recall_curve(y_true[i],y_scores[i])
        f = 2*p*r/(p+r+1e-8)
        if len(f): f_vals.append(f.max())
    return float(sum(f_vals)/len(f_vals)) if f_vals else 0.0

# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    p=argparse.ArgumentParser(description="ESM‑1b fine‑tune with structures")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--split_dir", default=DEFAULT_SPLIT)
    p.add_argument("--struct_dir", required=True)
    args=p.parse_args()

    # CSV paths (already prepared by external script)
    base=args.split_dir
    train_csv=os.path.join(base,f"train_fold{args.fold}.csv")
    val_csv  =os.path.join(base,f"val_fold{args.fold}.csv")
    test_csv =os.path.join(base,f"test_fold{args.fold}.csv")

    vocab=sorted({g for gs in load_annotations_adaptive(ANNOT_PATH).values() for g in gs})
    train_ds=SalmonellaMultiLabelDataset(train_csv,vocab,max_len=1024,structure_dir=args.struct_dir)
    val_ds  =SalmonellaMultiLabelDataset(val_csv,  vocab,max_len=1024,structure_dir=args.struct_dir)
    test_ds =SalmonellaMultiLabelDataset(test_csv, vocab,max_len=1024,structure_dir=args.struct_dir)

    train_loader=DataLoader(train_ds,BATCH_SIZE,True ,num_workers=NUM_WORKERS,pin_memory=PIN_MEM)
    val_loader  =DataLoader(val_ds  ,BATCH_SIZE,False,num_workers=NUM_WORKERS,pin_memory=PIN_MEM)
    test_loader =DataLoader(test_ds ,BATCH_SIZE,False,num_workers=NUM_WORKERS,pin_memory=PIN_MEM)

    model=build_model(train_ds.num_labels)
    for p_ in model.parameters(): p_.requires_grad=False
    for p_ in list(model.cls_head.parameters())+list(model.coord_head.parameters()): p_.requires_grad=True

    crit_cls   = nn.BCEWithLogitsLoss(pos_weight=compute_pos_weights(train_ds))
    crit_struct= nn.MSELoss()
    opt=optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    total=NUM_EPOCHS*len(train_loader); warm=int(WARMUP_PCT*total)
    sch = LambdaLR(opt, lambda t: t/warm if t<warm else max(0,(total-t)/(total-warm)))
    scaler=torch.cuda.amp.GradScaler()
    metrics=ensure_metrics_file(args.fold)

    best_val=float("inf")
    for epoch in range(1,NUM_EPOCHS+1):
        # ── train ──
        model.train(); loss_acc=0
        for i,(tok,len_,coord,tgt) in enumerate(train_loader,1):
            tok,len_,coord,tgt=[x.to(DEVICE) for x in (tok,len_,coord,tgt)]
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                logits,pcoords=model(tok,len_,coord)
                l_cls   = crit_cls(logits,tgt)
                l_struct= crit_struct(pcoords,coord)
                loss    = l_cls + LAMBDA_STRUCT*l_struct
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),MAX_GRAD_NORM)
            scaler.step(opt); scaler.update(); sch.step(); loss_acc+=loss.item()
            if i%50==0:
                print(f"Ep{epoch} step{i}/{len(train_loader)} loss{loss_acc/50:.4f}"); loss_acc=0
        # ── val ──
        model.eval(); v_cls=v_struct=0; all_t,all_s=[],[]
        with torch.no_grad():
            for tok,len_,coord,tgt in val_loader:
                tok,len_,coord,tgt=[x.to(DEVICE) for x in (tok,len_,coord,tgt)]
                logit,pcoord=model(tok,len_,coord)
                v_cls   += crit_cls(logit,tgt).item()
                v_struct+= crit_struct(pcoord,coord).item()
                all_t.append(tgt.cpu().numpy()); all_s.append(torch.sigmoid(logit).cpu().numpy())
        v_cls/=len(val_loader); v_struct/=len(val_loader)
        y_true=np.vstack(all_t); y_scores=np.vstack(all_s)
        term_auc=float(np.mean([average_precision
