import torch

from siteseer.models.probe import ActiveSiteProbe


def test_state_dict_keys_match_trained_checkpoints():
    keys = set(ActiveSiteProbe(embed_dim=1280, n_site_types=2).state_dict())
    assert keys == {"head.0.weight", "head.0.bias", "head.3.weight", "head.3.bias", "type_head.weight", "type_head.bias"}


def test_parameter_count_of_main_model():
    assert sum(p.numel() for p in ActiveSiteProbe(embed_dim=1280).parameters()) == 164_097


def test_forward_shapes():
    probe = ActiveSiteProbe(embed_dim=16, hidden_dim=8, n_site_types=2).eval()
    x = torch.randn(30, 16)
    logits, hidden, types = probe(x, return_hidden=True, return_types=True)
    assert logits.shape == (30,) and hidden.shape == (30, 8) and types.shape == (30, 2)
