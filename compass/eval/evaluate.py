"""Evaluate a trained probe on one split: pooled and macro metrics with protein-level bootstrap CIs.

  python -m compass.eval.evaluate --config configs/main/stage2_mcsa.yaml --encoder esm_raw \\
      --probe-checkpoint checkpoints/<run>/best.pt --split test [--calibrate-threshold] \\
      [--filter-ids-file ids.txt] [--save-predictions results/<name>.pt]

--save-predictions writes {"protein_ids", "logits", "labels"} (per-protein tensors), the input format of
compass.eval.paired_bootstrap and compass.eval.ensemble. --calibrate-threshold additionally fits the pooled-F1
threshold on the validation split and reports the requested split at that threshold.
"""
import argparse

import torch
from torch.utils.data import DataLoader

from compass.data.datasets import build_dataset, identity_collate
from compass.embeddings.embedders import embedder_from_config
from compass.eval.metrics import bootstrap_ci, find_best_threshold, format_bootstrap_result
from compass.models.probe import load_probe
from compass.train.losses import build_probe_loss
from compass.train.train_probe import ENCODERS, embed_dim_for, load_config, run_epoch


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--encoder", "--baseline-encoder", dest="encoder", required=True, choices=ENCODERS)
    ap.add_argument("--probe-checkpoint", required=True)
    ap.add_argument("--probe-architecture", default="flat", choices=["flat", "graph_v2"])
    ap.add_argument("--structure-root", default=None)
    ap.add_argument("--pretrained-gvp-checkpoint", default=None)
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--filter-ids-file", default=None, help="restrict to these protein ids (one per line)")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--calibrate-threshold", action="store_true")
    ap.add_argument("--save-predictions", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    data_cfg = cfg["data"]
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filter_ids = None
    if args.filter_ids_file:
        filter_ids = {line.strip() for line in open(args.filter_ids_file) if line.strip()}
        print(f"[*] restricting to {len(filter_ids)} proteins from {args.filter_ids_file}")

    if args.probe_architecture == "graph_v2":
        from compass.structural.probe import load_structural_for_eval   # structural ablation only
        probe, make_dataset, to_input, probe_cfg = load_structural_for_eval(args, cfg, device)
    else:
        probe, ckpt = load_probe(args.probe_checkpoint, device)
        probe_cfg = ckpt["config"]["probe"]
        width = ckpt["probe_state_dict"]["head.0.weight"].shape[1]
        if width != embed_dim_for(data_cfg, args.encoder):
            raise ValueError(f"checkpoint expects {width}-d embeddings, --encoder/--config give "
                             f"{embed_dim_for(data_cfg, args.encoder)} (set data.embed_dim / data.esm_model)")
        prov = ckpt.get("provenance") or {}
        print(f"[*] checkpoint: run_name={prov.get('run_name')} seed={prov.get('seed')} epoch={ckpt.get('epoch')} "
              f"val_auprc={ckpt.get('val_auprc')}")
        embedder = None
        if data_cfg.get("dataset_class", "mcsa") == "mcsa":
            embedder = embedder_from_config(data_cfg, args.encoder).load(device)
        make_dataset = lambda split, ids: build_dataset(cfg, split, embedder, filter_ids=ids)
        to_input = None

    def predict(split, ids):
        ds = make_dataset(split, ids)
        loader = DataLoader(ds, batch_size=cfg["optim"]["batch_size"], shuffle=False, collate_fn=identity_collate)
        _, loss, logits, labels, _ = run_epoch(loader, probe, device, build_probe_loss(probe_cfg, device),
                                               to_input=to_input)
        return ds, loss, logits, labels

    ds, loss, logits, labels = predict(args.split, filter_ids)
    result = bootstrap_ci(logits, labels, n_bootstrap=args.n_bootstrap)
    print(f"[{args.split}] loss={loss:.4f}\n--- threshold=0.5 ---\n{format_bootstrap_result(result)}")

    if args.calibrate_threshold and args.split != "valid":
        _, _, v_logits, v_labels = predict("valid", None)
        t = find_best_threshold(v_logits, v_labels)
        calibrated = bootstrap_ci(logits, labels, n_bootstrap=args.n_bootstrap, threshold=t)
        print(f"--- threshold={t:.3f} (pooled-F1 optimum on valid) ---\n{format_bootstrap_result(calibrated)}")

    if args.save_predictions:
        if len(ds.protein_ids) != len(logits):
            raise RuntimeError("protein ids and predictions are not aligned; refusing to save")
        torch.save({"protein_ids": ds.protein_ids, "logits": [x.cpu() for x in logits],
                    "labels": [y.cpu() for y in labels]}, args.save_predictions)
        print(f"[*] saved {len(logits)} proteins' predictions to {args.save_predictions}")


if __name__ == "__main__":
    main()
