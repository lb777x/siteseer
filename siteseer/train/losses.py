"""Probe losses. Focal loss (Lin et al. 2017) with alpha_t = alpha for positives and 1 - alpha for negatives
(alpha = 0.8, gamma = 2 in all configs).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "none")
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(inputs, targets)
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - pt) ** self.gamma * bce
        return loss.mean() if self.reduction == "mean" else loss


def build_probe_loss(probe_cfg: dict, device: torch.device, reduction: str = "mean"):
    """probe.loss 'focal' (focal_alpha, focal_gamma) or 'bce' (pos_weight)."""
    loss_type = probe_cfg.get("loss", "bce")
    if loss_type == "focal":
        focal = FocalLoss(probe_cfg.get("focal_alpha", 0.8), probe_cfg.get("focal_gamma", 2.0), reduction).to(device)
        return lambda logits, labels: focal(logits, labels)
    if loss_type == "bce":
        pos_weight = torch.tensor(probe_cfg["pos_weight"], device=device)
        return lambda logits, labels: F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight,
                                                                         reduction=reduction)
    raise ValueError(f"unknown probe.loss {loss_type!r} (expected 'focal' or 'bce')")


def masked_type_loss(type_logits, targets, mask, type_loss_fn):
    """Sum over types of the mean loss over that type's unmasked residues (0 for an unsupervised type)."""
    per_elem = type_loss_fn(type_logits, targets) * mask
    return (per_elem.sum(dim=0) / mask.sum(dim=0).clamp(min=1.0)).sum()
