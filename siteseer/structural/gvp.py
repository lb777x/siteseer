"""GVP-GNN encoder (Jing et al. 2021, Geometric Vector Perceptrons): residue graph -> per-residue embeddings.

Scalar outputs use only vector norms, vector outputs are gated linear mixes of input vectors, and the last layer
drops the vector channel, so the embedding is invariant to rotation and translation of the structure. Message
passing averages over the k neighbours (index_add_, no compiled scatter library).
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GVP(nn.Module):
    """(scalar, vector) -> (scalar, vector)."""

    def __init__(self, in_dims: Tuple[int, int], out_dims: Tuple[int, int], activation=F.relu,
                 vector_gate: bool = True):
        super().__init__()
        (in_s, in_v), (out_s, out_v) = in_dims, out_dims
        self.in_v, self.out_v = in_v, out_v
        self.hidden_v = max(in_v, out_v, 1)
        self.wh = nn.Linear(in_v, self.hidden_v, bias=False) if in_v > 0 else None
        self.ws = nn.Linear(in_s + (self.hidden_v if in_v > 0 else 0), out_s)
        self.activation = activation
        self.wv = None
        if out_v > 0:
            self.wv = nn.Linear(self.hidden_v, out_v, bias=False)
            self.vector_gate = vector_gate
            if vector_gate:
                self.wg = nn.Linear(out_s, out_v)

    def forward(self, s: torch.Tensor, v: Optional[torch.Tensor]):
        if self.in_v > 0:
            vh = self.wh(v.transpose(-1, -2)).transpose(-1, -2)      # [N, hidden_v, 3]
            s_in = torch.cat([s, vh.norm(dim=-1)], dim=-1)
        else:
            vh, s_in = None, s
        s_out = self.ws(s_in)
        if self.activation is not None:
            s_out = self.activation(s_out)
        if self.out_v == 0:
            return s_out, None
        v_out = self.wv(vh.transpose(-1, -2)).transpose(-1, -2)
        if self.vector_gate:
            v_out = v_out * torch.sigmoid(self.wg(s_out)).unsqueeze(-1)
        return s_out, v_out


class GVPConv(nn.Module):
    """One round of message passing: two-layer GVP messages, mean over neighbours, GVP update, residual."""

    def __init__(self, node_dims, edge_dims, hidden_dims):
        super().__init__()
        (node_s, node_v), (edge_s, edge_v) = node_dims, edge_dims
        self.message1 = GVP((2 * node_s + edge_s, 2 * node_v + edge_v), hidden_dims)
        self.message2 = GVP(hidden_dims, node_dims, activation=None)
        self.update = GVP((2 * node_s, 2 * node_v), node_dims)

    def forward(self, s, v, edge_index, edge_s, edge_v):
        src, dst = edge_index[0], edge_index[1]
        n = s.shape[0]
        m_s, m_v = self.message1(torch.cat([s[src], s[dst], edge_s], dim=-1), torch.cat([v[src], v[dst], edge_v], dim=1))
        m_s, m_v = self.message2(m_s, m_v)
        agg_s = torch.zeros(n, m_s.shape[-1], device=s.device, dtype=s.dtype).index_add_(0, dst, m_s)
        agg_v = torch.zeros(n, m_v.shape[1], 3, device=v.device, dtype=v.dtype).index_add_(0, dst, m_v)
        deg = torch.zeros(n, device=s.device, dtype=s.dtype).index_add_(
            0, dst, torch.ones(dst.shape[0], device=s.device, dtype=s.dtype)).clamp(min=1.0)
        s_new, v_new = self.update(torch.cat([s, agg_s / deg.unsqueeze(-1)], dim=-1),
                                   torch.cat([v, agg_v / deg.view(-1, 1, 1)], dim=1))
        return s + s_new, v + v_new


class GVPEncoder(nn.Module):
    """Graph (graph.build_knn_graph) -> [R, out_dim]."""

    def __init__(self, node_scalar_dim=6, node_vector_dim=3, edge_scalar_dim=16, edge_vector_dim=1,
                 hidden_scalar=256, hidden_vector=64, num_layers=5, out_dim=256):
        super().__init__()
        hidden = (hidden_scalar, hidden_vector)
        self.embed = GVP((node_scalar_dim, node_vector_dim), hidden)
        self.layers = nn.ModuleList([GVPConv(hidden, (edge_scalar_dim, edge_vector_dim), hidden)
                                     for _ in range(num_layers)])
        self.output = GVP(hidden, (out_dim, 0), activation=None)

    def forward(self, data) -> torch.Tensor:
        s, v = self.embed(data.s, data.v)
        for layer in self.layers:
            s, v = layer(s, v, data.edge_index, data.edge_s, data.edge_v)
        return self.output(s, v)[0]


def encoder_from_config(enc_cfg: dict) -> GVPEncoder:
    return GVPEncoder(hidden_scalar=enc_cfg["hidden_scalar"], hidden_vector=enc_cfg["hidden_vector"],
                      num_layers=enc_cfg["num_layers"], out_dim=enc_cfg["out_dim"])


def load_pretrained(path: str, device) -> Tuple[GVPEncoder, dict]:
    """The self-supervised encoder of an SSL checkpoint, frozen."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    enc = encoder_from_config(ckpt["config"]["encoder"])
    enc.load_state_dict(ckpt["encoder_state_dict"])
    enc.to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, ckpt["config"]["encoder"]
