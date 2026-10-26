"""Train a probe on frozen embeddings.

  python -m siteseer.train.train_probe --config configs/main/stage2_mcsa.yaml --encoder esm_raw \\
      --init-probe-checkpoint checkpoints/main_stage15/best.pt [--set seed=43 run_name=main_stage2_mcsa_seed43]

AdamW with one step per protein and gradient clipping at 1.0; the learning rate halves after two validation checks
without improvement. best.pt has the best validation AUPRC of the main output; latest.pt (--checkpoint-every-n-steps)
allows --resume-from.
"""
import argparse
import os
import random
import subprocess
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader

from siteseer.data.datasets import build_dataset, identity_collate
from siteseer.data.site_types import TYPE_NAMES
from siteseer.embeddings.embedders import embedder_from_config
from siteseer.eval.metrics import binary_metrics
from siteseer.models.probe import probe_from_config
from siteseer.train.losses import build_probe_loss, masked_type_loss

ENCODERS = ("esm_raw", "esmc_raw")


def apply_overrides(cfg: dict, overrides: list) -> dict:
    """--set a.b=value (value parsed as YAML) on a loaded config, e.g. seed=43 or data.train_csv=path."""
    for item in overrides or []:
        key, _, value = item.partition("=")
        if not key or not _:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def load_config(path: str, overrides: list = None) -> dict:
    with open(path) as f:
        return apply_overrides(yaml.safe_load(f), overrides)


def embed_dim_for(data_cfg: dict, encoder: str) -> int:
    return data_cfg.get("embed_dim") or (2560 if encoder == "esmc_raw" else 1280)


def type_metrics(type_sink: list) -> dict:
    """Pooled AUPRC / AUROC per site type over the unmasked residues collected by run_epoch."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    logits = torch.cat([t[0] for t in type_sink]).float()
    targets = torch.cat([t[1] for t in type_sink])
    mask = torch.cat([t[2] for t in type_sink])
    out = {}
    for k, name in enumerate(TYPE_NAMES):
        m = mask[:, k] > 0
        y, s = targets[m, k].numpy(), logits[m, k].numpy()
        if m.sum() == 0 or y.min() == y.max():
            out[name] = None
            continue
        out[name] = {"auprc": float(average_precision_score(y, s)), "auroc": float(roc_auc_score(y, s)),
                     "n_residues": int(m.sum()), "n_positive": int(y.sum())}
    return out


def run_epoch(loader, probe, device, loss_fn, optimizer=None, to_input=None, type_loss_fn=None,
              type_loss_weight: float = 0.0, type_sink: list = None, log_every: int = None, log_prefix: str = "",
              save_callback=None, global_step: int = 0):
    """One pass over loader; optimizer=None means evaluation. to_input(item) -> the probe's input (default: the
    item's embedding). Returns (metrics, mean loss, per-protein logits, per-protein labels, global_step)."""
    is_train = optimizer is not None
    probe.train() if is_train else probe.eval()
    to_input = to_input or (lambda item: item["embedding"].to(device))
    all_logits, all_labels, losses = [], [], []
    count, start = 0, time.time()
    for batch in loader:
        for item in batch:
            labels = item["labels"].to(device)
            with torch.no_grad():
                x = to_input(item)
            use_types = getattr(probe, "n_site_types", 0) > 0 and "type_targets" in item
            with torch.set_grad_enabled(is_train):
                if use_types:
                    logits, type_logits = probe(x, return_types=True)
                else:
                    logits = probe(x)
                loss = loss_fn(logits, labels)
                if use_types:
                    targets, mask = item["type_targets"].to(device), item["type_mask"].to(device)
                    if is_train and type_loss_fn is not None and type_loss_weight:
                        loss = loss + type_loss_weight * masked_type_loss(type_logits, targets, mask, type_loss_fn)
                    if type_sink is not None:
                        type_sink.append((type_logits.detach().cpu(), targets.cpu(), mask.cpu()))
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
                optimizer.step()
                global_step += 1
            losses.append(loss.detach().item())
            all_logits.append(logits.detach())
            all_labels.append(labels.detach())
            count += 1
            if is_train and log_every and count % log_every == 0:
                recent = sum(losses[-log_every:]) / min(log_every, len(losses))
                print(f"  {log_prefix}protein {count}/{len(loader.dataset)} loss={recent:.4f} "
                      f"({time.time() - start:.1f}s)")
                if save_callback is not None:
                    save_callback(global_step)
    if not all_logits:
        return None, None, [], [], global_step
    metrics = binary_metrics(torch.cat(all_logits), torch.cat(all_labels))
    return metrics, sum(losses) / len(losses), all_logits, all_labels, global_step


def _git_hash() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=here,
                             timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "siteseer"],
                               capture_output=True, text=True, cwd=here, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def build_model_and_data(args, cfg, device):
    """-> (probe, train_ds, valid_ds, to_input). Load order matters for reproducibility: the language model is
    constructed (consuming the seeded RNG) before the datasets and the probe."""
    data_cfg, probe_cfg = cfg["data"], cfg["probe"]
    embed_dim = embed_dim_for(data_cfg, args.encoder)
    if args.probe_architecture == "graph_v2":
        from siteseer.structural.probe import build_structural   # structural ablation only
        return build_structural(args, cfg, device, embed_dim)
    embedder = None
    if data_cfg.get("dataset_class", "mcsa") == "mcsa":
        embedder = embedder_from_config(data_cfg, args.encoder).load(device)
    train_ds = build_dataset(cfg, "train", embedder)
    valid_ds = build_dataset(cfg, "valid", embedder)
    probe = probe_from_config(probe_cfg, embed_dim).to(device)
    return probe, train_ds, valid_ds, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config entries")
    ap.add_argument("--encoder", "--baseline-encoder", dest="encoder", required=True, choices=ENCODERS)
    ap.add_argument("--probe-architecture", default="flat", choices=["flat", "graph_v2"])
    ap.add_argument("--structure-root", default=None)
    ap.add_argument("--pretrained-gvp-checkpoint", default=None)
    ap.add_argument("--init-probe-checkpoint", default=None, help="warm start from a probe checkpoint")
    ap.add_argument("--resume-from", default=None, help="a latest.pt; restarts the epoch it was saved in")
    ap.add_argument("--checkpoint-every-n-steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    provenance = {"run_name": cfg["run_name"], "seed": seed, "config_path": args.config, "overrides": args.set,
                  "git_hash": _git_hash(), "argv": sys.argv[1:], "init_probe_checkpoint": args.init_probe_checkpoint,
                  "pretrained_gvp_checkpoint": args.pretrained_gvp_checkpoint,
                  "probe_architecture": args.probe_architecture}
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] run={cfg['run_name']} seed={seed} git={provenance['git_hash']} device={device}")

    probe, train_ds, valid_ds, to_input = build_model_and_data(args, cfg, device)
    if len(train_ds) == 0:
        raise RuntimeError("no usable labelled proteins for training")
    bs = cfg["optim"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=identity_collate, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=bs, shuffle=False, collate_fn=identity_collate, num_workers=0)

    if args.init_probe_checkpoint:
        init = torch.load(args.init_probe_checkpoint, map_location=device, weights_only=False)
        probe.load_state_dict(init["probe_state_dict"])
        print(f"[*] initialised probe from {args.init_probe_checkpoint}")

    probe_cfg = cfg["probe"]
    optimizer = torch.optim.AdamW(probe.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    loss_fn = build_probe_loss(probe_cfg, device)
    n_types = probe_cfg.get("n_site_types", 0)
    type_loss_fn, type_loss_weight = None, 0.0
    if n_types:
        if n_types != len(TYPE_NAMES) or args.probe_architecture != "flat":
            raise NotImplementedError(f"site types need the flat probe and n_site_types={len(TYPE_NAMES)}")
        type_loss_fn = build_probe_loss(probe_cfg, device, reduction="none")
        type_loss_weight = probe_cfg.get("site_type_loss_weight", 0.5)
        print(f"[*] site-type outputs {TYPE_NAMES}, masked loss weight {type_loss_weight}")

    encoder_ref = f"baseline:{args.encoder}"
    ckpt_dir = os.path.join(cfg["logging"]["checkpoint_dir"], cfg["run_name"])
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_auprc, start_epoch, global_step = -1.0, 0, 0
    if args.resume_from:
        r = torch.load(args.resume_from, map_location=device, weights_only=False)
        probe.load_state_dict(r["probe_state_dict"])
        optimizer.load_state_dict(r["optimizer_state_dict"])
        if r.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(r["scheduler_state_dict"])
        best_val_auprc, start_epoch, global_step = r.get("best_val_auprc", -1.0), r["epoch"], r.get("global_step", 0)
        print(f"[*] resumed from {args.resume_from}: epoch {start_epoch}, best valid AUPRC {best_val_auprc:.3f}")

    epochs = cfg["optim"]["epochs"]
    eval_every = cfg["logging"]["eval_every_n_epochs"]
    ckpt_every = args.checkpoint_every_n_steps or cfg["logging"].get("checkpoint_every_n_steps")
    log_every = ckpt_every or cfg["logging"].get("log_every_n_steps")
    epoch = start_epoch

    def save(name: str, **extra):
        state = {"probe_state_dict": probe.state_dict(), "config": cfg, "encoder_checkpoint": encoder_ref,
                 "provenance": provenance, **extra}
        torch.save(state, os.path.join(ckpt_dir, name))

    def save_latest(step: int):
        save("latest.pt", optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
             epoch=epoch, global_step=step, best_val_auprc=best_val_auprc)

    for epoch in range(start_epoch, epochs):
        tm, tl, _, _, global_step = run_epoch(
            train_loader, probe, device, loss_fn, optimizer=optimizer, to_input=to_input, type_loss_fn=type_loss_fn,
            type_loss_weight=type_loss_weight, log_every=log_every, log_prefix=f"epoch {epoch} ",
            save_callback=save_latest if ckpt_every else None, global_step=global_step)
        print(f"epoch {epoch:3d} train_loss={tl:.4f} P={tm['precision']:.3f} R={tm['recall']:.3f} "
              f"F1={tm['f1']:.3f} AUPRC={tm['auprc']:.3f}")
        if epoch % eval_every and epoch != epochs - 1:
            continue
        sink = [] if n_types else None
        vm, vl, _, _, _ = run_epoch(valid_loader, probe, device, loss_fn, to_input=to_input, type_sink=sink)
        if vm is None:
            print("  [valid] no usable validation proteins")
            continue
        print(f"  [valid] loss={vl:.4f} P={vm['precision']:.3f} R={vm['recall']:.3f} F1={vm['f1']:.3f} "
              f"AUPRC={vm['auprc']:.3f} (n_residues={vm['n_residues']}, n_positive={vm['n_positive']})")
        for name, t in (type_metrics(sink).items() if sink else []):
            if t:
                print(f"  [valid:{name}] AUPRC={t['auprc']:.3f} AUROC={t['auroc']:.3f}")
        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(vm["auprc"])
        if optimizer.param_groups[0]["lr"] != lr_before:
            print(f"  [scheduler] lr {lr_before:.2e} -> {optimizer.param_groups[0]['lr']:.2e}")
        if vm["auprc"] > best_val_auprc:
            best_val_auprc = vm["auprc"]
            save("best.pt", epoch=epoch, val_auprc=best_val_auprc, val_f1=vm["f1"])
            print(f"  [checkpoint] new best (AUPRC={best_val_auprc:.3f}) -> {ckpt_dir}/best.pt")

    save("final.pt", epoch=epochs - 1)
    print(f"[+] done. best valid AUPRC={best_val_auprc:.3f}; checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
