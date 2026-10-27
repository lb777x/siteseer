"""Structural probe: a frozen, self-supervised GVP encoder over the AlphaFold structure, fused with the untouched
ESMC embedding, then the Flat head. The structural ablation of the thesis (ESMC-6B, fix1 data).

  per residue: fusion = ReLU(Linear([ESMC (2560) || GVP (256)] -> 256)), dropout; head as the Flat probe
"""
import torch
import torch.nn as nn

from siteseer.embeddings.embedders import ESMCEmbedder
from siteseer.models.probe import ActiveSiteProbe
from siteseer.structural.datasets import StructuralCachedDataset, StructuralLiveDataset
from siteseer.structural.gvp import GVPEncoder, load_pretrained
from siteseer.structural.graph import build_knn_graph


class StructuralProbe(nn.Module):
    def __init__(self, gvp_encoder: nn.Module, esm_dim=2560, gvp_out_dim=256, fusion_hidden_dim=256,
                 probe_hidden_dim=128, dropout_rate=0.3):
        super().__init__()
        self.gvp_encoder = gvp_encoder
        self.fusion = nn.Sequential(nn.Linear(esm_dim + gvp_out_dim, fusion_hidden_dim), nn.ReLU(),
                                    nn.Dropout(dropout_rate))
        self.probe = ActiveSiteProbe(embed_dim=fusion_hidden_dim, hidden_dim=probe_hidden_dim,
                                     dropout_rate=dropout_rate)

    def forward(self, data) -> torch.Tensor:
        structural = self.gvp_encoder(data)
        return self.probe(self.fusion(torch.cat([data.side_channel_embedding, structural], dim=-1)))


def make_probe(cfg, encoder, gvp_out_dim, embed_dim, device):
    sp, pc = cfg.get("structural_probe", {}), cfg["probe"]
    return StructuralProbe(encoder, esm_dim=embed_dim, gvp_out_dim=gvp_out_dim,
                           fusion_hidden_dim=sp.get("fusion_hidden_dim", 256), probe_hidden_dim=pc["hidden_dim"],
                           dropout_rate=pc["dropout"]).to(device)


def make_dataset(cfg, split, structure_root, embedder=None, filter_ids=None):
    d = cfg["data"]
    if d.get("dataset_class", "mcsa") == "mcsa":
        return StructuralLiveDataset(d[f"{split}_csv"], structure_root, embedder, d["id_column"], filter_ids,
                                     label=f"structural:{split}")
    if filter_ids is not None:
        raise NotImplementedError("filter_ids needs dataset_class 'mcsa'")
    return StructuralCachedDataset(d["train_csv"], structure_root, d["embedding_dir"], split, d["id_column"],
                                   label=f"structural:{split}", resample_on_error=(split == "train"))


def graph_input(k, device):
    return lambda item: build_knn_graph(item["frames"], k=k, side_channel_embedding=item["esm_embedding"]).to(device)


def build_structural(args, cfg, device, embed_dim):
    """For train_probe: -> (probe, train_ds, valid_ds, to_input). The GVP encoder comes frozen from
    --pretrained-gvp-checkpoint."""
    if not args.pretrained_gvp_checkpoint:
        raise SystemExit("--probe-architecture graph_v2 needs --pretrained-gvp-checkpoint")
    root = args.structure_root or cfg["data"]["structure_root"]
    embedder = ESMCEmbedder().load(device) if cfg["data"].get("dataset_class", "mcsa") == "mcsa" else None
    train_ds, valid_ds = (make_dataset(cfg, s, root, embedder) for s in ("train", "valid"))
    encoder, enc_cfg = load_pretrained(args.pretrained_gvp_checkpoint, device)
    probe = make_probe(cfg, encoder, enc_cfg["out_dim"], embed_dim, device)
    return probe, train_ds, valid_ds, graph_input(cfg.get("structural_probe", {}).get("k_neighbors", 24), device)


def load_structural_for_eval(args, cfg, device):
    """For evaluate: -> (probe, make_dataset(split, ids), to_input, probe_cfg); graph settings from the checkpoint."""
    ckpt = torch.load(args.probe_checkpoint, map_location=device, weights_only=False)
    sd = ckpt["probe_state_dict"]
    enc = GVPEncoder(hidden_scalar=sd["gvp_encoder.embed.ws.weight"].shape[0],
                     hidden_vector=sd["gvp_encoder.embed.wv.weight"].shape[0],
                     num_layers=len({k.split(".")[2] for k in sd if k.startswith("gvp_encoder.layers.")}),
                     out_dim=sd["gvp_encoder.output.ws.weight"].shape[0])
    embed_dim = sd["fusion.0.weight"].shape[1] - sd["gvp_encoder.output.ws.weight"].shape[0]
    run_cfg = {**cfg, "probe": ckpt["config"]["probe"], "structural_probe": ckpt["config"].get("structural_probe", {})}
    probe = make_probe(run_cfg, enc, sd["gvp_encoder.output.ws.weight"].shape[0], embed_dim, device)
    probe.load_state_dict(sd)
    probe.eval()
    root = args.structure_root or cfg["data"]["structure_root"]
    embedder = ESMCEmbedder().load(device) if cfg["data"].get("dataset_class", "mcsa") == "mcsa" else None
    k = run_cfg["structural_probe"].get("k_neighbors", 24)
    return probe, (lambda split, ids: make_dataset(cfg, split, root, embedder, ids)), graph_input(k, device), \
        ckpt["config"]["probe"]
