"""Self-supervised pretraining of the GVP encoder (LeJEPA: masked latent prediction with SIGReg).

Per protein a spatially contiguous block of 15% of the residues is masked (a random seed residue grown through its
16 nearest CA neighbours). The encoder embeds the full structure and, separately, the unmasked context; a predictor
estimates each masked residue's embedding from its 8 nearest context residues (their embeddings and RBF distances
only, so it stays rotation invariant). Loss = MSE(prediction, full-structure embedding of the masked residues; no
stop-gradient) + 25 x SIGReg on all full-structure embeddings of the batch (Balestriero & LeCun 2025: 32 random
projections, Epps-Pulley statistic on 32 points in [-5, 5]). AdamW (lr 3e-4, weight decay 0.05, constant), gradient
norm clipped at 1.0, batch 16, 20 epochs, bf16 autocast on GPUs that support it. Proteins: the stage-1 training split
with a structure of >= 10 residues and mean pLDDT >= 30.

  python -m siteseer.structural.pretrain --config configs/structural/pretrain_gvp.yaml --structure-root data/structures
"""
import argparse
import os
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from siteseer.structural.gvp import encoder_from_config
from siteseer.structural.graph import build_knn_graph, load_structure_frames, rbf_encode, resolve_pdb_path, slice_frames


def block_mask(ca, target_fraction=0.15, num_blocks=1, knn_k=16):
    """Bool [R], True = masked: random seed residue(s) grown by repeated 3D k-NN expansion to target_fraction."""
    r = ca.shape[0]
    target = max(1, int(round(min(max(target_fraction, 0.10), 0.20) * r)))
    k = min(knn_k, r - 1) if r > 1 else 0
    dist = torch.cdist(ca, ca)
    dist.fill_diagonal_(float("inf"))
    knn = dist.topk(k, largest=False, dim=-1)[1] if k > 0 else torch.empty((r, 0), dtype=torch.long)
    mask = torch.zeros(r, dtype=torch.bool)
    seeds = torch.randperm(r)[:min(num_blocks, r)]
    mask[seeds] = True
    frontier = seeds.tolist()
    while mask.sum().item() < target and frontier:
        nxt = []
        for node in frontier:
            for nb in knn[node].tolist():
                if not mask[nb]:
                    mask[nb] = True
                    nxt.append(nb)
                    if mask.sum().item() >= target:
                        break
            if mask.sum().item() >= target:
                break
        frontier = nxt
    return mask


class Predictor(nn.Module):
    """Masked residue embedding from the k nearest context residues (embedding + RBF distance), mean-pooled."""

    def __init__(self, embed_dim=256, hidden_dim=256, num_layers=2, query_knn_k=8, num_rbf=16):
        super().__init__()
        self.query_knn_k, self.num_rbf = query_knn_k, num_rbf
        self.neighbor_mlp = nn.Sequential(nn.Linear(embed_dim + num_rbf, hidden_dim), nn.ReLU(),
                                          nn.Linear(hidden_dim, hidden_dim))
        layers = []
        for _ in range(max(num_layers - 1, 0)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.output_mlp = nn.Sequential(*layers, nn.Linear(hidden_dim, embed_dim))

    def forward(self, context_emb, context_ca, query_ca):
        k = min(self.query_knn_k, context_ca.shape[0])
        knn_dist, knn_idx = torch.cdist(query_ca, context_ca).topk(k, largest=False, dim=-1)
        feat = torch.cat([context_emb[knn_idx], rbf_encode(knn_dist, num_rbf=self.num_rbf)], dim=-1)
        return self.output_mlp(self.neighbor_mlp(feat).mean(dim=1))


class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regularisation: Epps-Pulley distance of K random 1-D projections of the
    (centred, globally scaled) embeddings from N(0, 1), on a G-point grid in [-t_max, t_max]."""

    def __init__(self, embed_dim, K=32, G=32, t_max=5.0):
        super().__init__()
        self.embed_dim, self.K, self.G, self.t_max = embed_dim, K, G, t_max
        t = torch.linspace(-t_max, t_max, G)
        self.register_buffer("t_grid", t)
        self.register_buffer("gauss_weight", torch.exp(-0.5 * t ** 2))
        self.register_buffer("target_re", torch.exp(-0.5 * t ** 2))

    def forward(self, z):
        dirs = torch.randn(self.K, self.embed_dim, device=z.device)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        z = z - z.mean(dim=0, keepdim=True)
        z = z / z.std().clamp(min=1e-6)
        angle = (z @ dirs.t()).unsqueeze(-1) * self.t_grid.view(1, 1, -1)
        sq_err = (torch.cos(angle).mean(0) - self.target_re.view(1, -1)) ** 2 + torch.sin(angle).mean(0) ** 2
        return ((sq_err * self.gauss_weight.view(1, -1)).sum(-1) * (2 * self.t_max / self.G)).mean()


def effective_rank(z):
    z = z - z.mean(dim=0, keepdim=True)
    eig = torch.linalg.eigvalsh((z.t() @ z) / max(z.shape[0] - 1, 1)).clamp(min=0).flip(0)
    p = eig / eig.sum().clamp(min=1e-12)
    return torch.exp(-(p * torch.log(p.clamp(min=1e-12))).sum()).item()


class MaskedStructures(Dataset):
    def __init__(self, csv_path, structure_root, split_name="train", id_column="alphafolddb-id", masking=None,
                 min_ca=10, min_plddt=30.0):
        self.masking = masking or {}
        d = pd.read_csv(csv_path)
        d = d[d["dataset_flag"] == split_name].drop_duplicates(subset=id_column)
        self.frames, self.ids, skipped = [], [], {"unresolved": 0, "failed_qc": 0}
        for _, row in d.iterrows():
            path = resolve_pdb_path(row, structure_root, id_column)
            try:
                f = load_structure_frames(path) if path else None
            except Exception:
                f = None
            if f is None:
                skipped["unresolved"] += 1
                continue
            if len(f.residue_numbers) < min_ca or (f.plddt.numel() and f.plddt.mean().item() < min_plddt):
                skipped["failed_qc"] += 1
                continue
            self.frames.append(f)
            self.ids.append(str(row[id_column]).strip())
        print(f"[*] {len(self.frames)} structures ({skipped} of {len(d)})", flush=True)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        m = self.masking
        f = self.frames[i]
        return {"frames": f, "mask": block_mask(f.coords_ca, m.get("target_fraction", 0.15), m.get("num_blocks", 1),
                                                 m.get("knn_k_for_growth", 16)), "protein_id": self.ids[i]}


def train_step(batch, encoder, predictor, sigreg, cfg, device, autocast_dtype):
    k = cfg["encoder"]["k_neighbors"]
    pred_losses, full_embs = [], []
    for item in batch:
        frames, mask = item["frames"], item["mask"]
        n_context = int((~mask).sum())
        if mask.sum() == 0 or n_context < 2:
            continue
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            full_emb = encoder(build_knn_graph(frames, k=k).to(device))
            context = slice_frames(frames, ~mask)
            context_emb = encoder(build_knn_graph(context, k=min(k, n_context - 1)).to(device))
            pred = predictor(context_emb, frames.coords_ca[~mask].to(device), frames.coords_ca[mask].to(device))
            pred_losses.append(F.mse_loss(pred, full_emb[mask.to(device)]))
            full_embs.append(full_emb)
    if not pred_losses:
        return None
    pred_loss = torch.stack(pred_losses).mean()
    pooled = torch.cat(full_embs, dim=0).float()          # regulariser in fp32
    reg = sigreg(pooled)
    return pred_loss + cfg["objective"]["sigreg"]["lambda"] * reg, pred_loss.item(), reg.item(), pooled.detach()


def save(path, step, encoder, predictor, optimizer, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"step": step, "encoder_state_dict": encoder.state_dict(), "predictor_state_dict": predictor.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "config": cfg}, path)
    print(f"  [checkpoint] {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--structure-root", default=None)
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg.get("seed", 42))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    autocast_dtype = None
    if device.type == "cuda" and cfg["hardware"]["precision"] in ("bf16", "fp16"):
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() and cfg["hardware"]["precision"] == "bf16" \
            else torch.float16
    d = cfg["data"]
    ds = MaskedStructures(d["train_csv"], args.structure_root or d["structure_root"], d.get("split_name", "train"),
                          d["id_column"], cfg["masking"], d["qc"]["min_ca_atoms"], d["qc"]["min_avg_plddt"])
    loader = DataLoader(ds, batch_size=cfg["optim"]["batch_size"], shuffle=True, collate_fn=lambda b: b, num_workers=0)
    enc_cfg, pc = cfg["encoder"], cfg["predictor"]
    encoder = encoder_from_config(enc_cfg).to(device)
    predictor = Predictor(enc_cfg["out_dim"], pc["hidden_dim"], pc["num_layers"], pc["query_knn_k"]).to(device)
    s = cfg["objective"]["sigreg"]
    sigreg = SIGReg(enc_cfg["out_dim"], s["K"], s["G"], s["t_max"]).to(device)
    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    step = 0
    if args.resume_from:
        ck = torch.load(args.resume_from, map_location=device, weights_only=False)
        encoder.load_state_dict(ck["encoder_state_dict"])
        predictor.load_state_dict(ck["predictor_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        step = ck["step"]
    total = cfg["optim"]["epochs"] * len(loader)
    run_dir = os.path.join(args.checkpoint_dir, cfg["run_name"])
    log = cfg["logging"]
    print(f"[*] {len(loader)} steps per epoch, {total} steps, starting at {step}; device {device}", flush=True)
    t0 = time.time()
    for epoch in range(cfg["optim"]["epochs"]):
        for batch in loader:
            out = train_step(batch, encoder, predictor, sigreg, cfg, device, autocast_dtype)
            if out is None:
                continue
            loss, pred_val, reg_val, pooled = out
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            if step % log["log_every_n_steps"] == 0:
                print(f"step {step:5d} epoch {epoch:3d} loss {loss.item():.4f} pred {pred_val:.4f} reg {reg_val:.4f} "
                      f"({time.time() - t0:.1f}s)", flush=True)
            if step % log["collapse_diagnostics_every_n_steps"] == 0:
                print(f"  effective rank {effective_rank(pooled):.2f}/{pooled.shape[1]}", flush=True)
            if step > 0 and step % log["checkpoint_every_n_steps"] == 0:
                save(os.path.join(run_dir, f"step_{step}.pt"), step, encoder, predictor, optimizer, cfg)
            step += 1
            if step >= total or (args.max_steps is not None and step >= args.max_steps):
                save(os.path.join(run_dir, "final.pt"), step, encoder, predictor, optimizer, cfg)
                return
    save(os.path.join(run_dir, "final.pt"), step, encoder, predictor, optimizer, cfg)


if __name__ == "__main__":
    main()
