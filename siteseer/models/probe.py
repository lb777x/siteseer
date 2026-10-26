"""Flat probe: Linear(d, 128) - ReLU - Dropout - Linear(128, 1) per residue (164,097 parameters for d = 1280),
with an optional linear output for the site types on the same hidden layer. Layer names match the trained checkpoints.
"""
import torch
import torch.nn as nn


class ActiveSiteProbe(nn.Module):
    def __init__(self, embed_dim: int = 1280, hidden_dim: int = 128, dropout_rate: float = 0.2,
                 n_site_types: int = 0, mode: str = "agnostic"):
        super().__init__()
        if mode != "agnostic":
            raise ValueError(f"only mode='agnostic' is supported (got {mode!r})")
        self.mode = mode
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )
        self.n_site_types = n_site_types
        if n_site_types > 0:
            self.type_head = nn.Linear(hidden_dim, n_site_types)

    def forward(self, residue_embeddings: torch.Tensor, return_hidden: bool = False, return_types: bool = False):
        """[R, d] -> logits [R]; with return_hidden also the hidden layer [R, hidden]; with return_types also the
        type logits [R, n_site_types]. Order: (logits, hidden, types)."""
        hidden = self.head[:-1](residue_embeddings)
        logits = self.head[-1](hidden).squeeze(-1)
        out = [logits]
        if return_hidden:
            out.append(hidden)
        if return_types:
            if self.n_site_types == 0:
                raise ValueError("return_types=True needs a probe built with n_site_types > 0")
            out.append(self.type_head(hidden))
        return out[0] if len(out) == 1 else tuple(out)


def probe_from_config(probe_cfg: dict, embed_dim: int) -> ActiveSiteProbe:
    return ActiveSiteProbe(embed_dim=embed_dim, hidden_dim=probe_cfg["hidden_dim"],
                           dropout_rate=probe_cfg["dropout"], n_site_types=probe_cfg.get("n_site_types", 0),
                           mode=probe_cfg.get("mode", "agnostic"))


def load_probe(path: str, device: torch.device):
    """A trained Flat probe from its checkpoint; the input width comes from the weights. -> (probe, checkpoint)"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    embed_dim = ckpt["probe_state_dict"]["head.0.weight"].shape[1]
    probe = probe_from_config(ckpt["config"]["probe"], embed_dim).to(device)
    probe.load_state_dict(ckpt["probe_state_dict"])
    return probe.eval(), ckpt
