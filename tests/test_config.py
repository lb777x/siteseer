import argparse
import os

import pytest

from siteseer.train.train_probe import apply_overrides, embed_dim_for, load_config, merge, resolve_encoder


def test_overrides_nested_and_typed():
    cfg = {"seed": 42, "data": {"train_csv": "a.csv"}}
    apply_overrides(cfg, ["seed=43", "data.train_csv=b.csv", "logging.checkpoint_dir=/tmp/x", "probe.n_site_types=2"])
    assert cfg["seed"] == 43 and cfg["data"]["train_csv"] == "b.csv"
    assert cfg["logging"]["checkpoint_dir"] == "/tmp/x" and cfg["probe"]["n_site_types"] == 2


def test_override_needs_equals():
    with pytest.raises(ValueError):
        apply_overrides({}, ["seed"])


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_overlay_merges_nested_and_overrides():
    cfg = merge({"data": {"a": 1, "b": 2}, "seed": 42}, {"data": {"b": 3, "c": None}})
    assert cfg == {"data": {"a": 1, "b": 3, "c": None}, "seed": 42}


@pytest.mark.parametrize("plm,encoder,dim", [("esm2_l33", "esm_raw", 1280), ("esm2_3b", "esm_raw", 2560),
                                              ("esmc_6b", "esmc_raw", 2560)])
def test_plm_overlays(plm, encoder, dim):
    for stage in ("stage1", "stage15", "stage2_mcsa", "kfold"):
        cfg = load_config([f"{ROOT}/configs/main/{stage}.yaml", f"{ROOT}/configs/plm/{plm}.yaml"])
        args = argparse.Namespace(encoder=None)
        resolve_encoder(args, cfg)
        assert args.encoder == encoder and embed_dim_for(cfg["data"], encoder) == dim
        assert cfg["data"]["norm_stats"] is None and cfg["probe"]["hidden_dim"] == 128


def test_main_config_defaults_to_esm2():
    cfg = load_config(f"{ROOT}/configs/main/stage1.yaml")
    args = argparse.Namespace(encoder=None)
    resolve_encoder(args, cfg)
    assert args.encoder == "esm_raw" and cfg["data"]["esm_layer"] == 30 and cfg["data"]["norm_stats"]
