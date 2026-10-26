import pytest

from siteseer.train.train_probe import apply_overrides


def test_overrides_nested_and_typed():
    cfg = {"seed": 42, "data": {"train_csv": "a.csv"}}
    apply_overrides(cfg, ["seed=43", "data.train_csv=b.csv", "logging.checkpoint_dir=/tmp/x", "probe.n_site_types=2"])
    assert cfg["seed"] == 43 and cfg["data"]["train_csv"] == "b.csv"
    assert cfg["logging"]["checkpoint_dir"] == "/tmp/x" and cfg["probe"]["n_site_types"] == 2


def test_override_needs_equals():
    with pytest.raises(ValueError):
        apply_overrides({}, ["seed"])
