"""Golden tests: the release code reproduces the main model's saved predictions and trains end to end.

Slow (ESM-2 650M on CPU, ~10 min for the 95 M-CSA test proteins); opt in with COMPASS_SLOW=1 and COMPASS_DATA_ROOT.
"""
import os
import subprocess
import sys

import pandas as pd
import pytest
import torch

from conftest import need

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
slow = pytest.mark.skipif(os.environ.get("COMPASS_SLOW") != "1", reason="slow; set COMPASS_SLOW=1")


def run(args, cwd):
    env = dict(os.environ, PYTHONPATH=REPO)
    out = subprocess.run([sys.executable, "-m", *args], cwd=cwd, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout


def stage2_config(data_root):
    return need(data_root, "configs/main/stage2_mcsa.yaml")


@slow
def test_main_model_reproduces_saved_test_predictions(data_root, tmp_path):
    ckpt = need(data_root, "checkpoints/main_stage2_mcsa/best.pt")
    saved = torch.load(need(data_root, "results/predictions/main_stage2_mcsa_test.pt"), weights_only=False)
    out = tmp_path / "pred.pt"
    run(["compass.eval.evaluate", "--config", stage2_config(data_root), "--encoder", "esm_raw",
         "--probe-checkpoint", ckpt, "--split", "test", "--n-bootstrap", "10", "--device", "cpu",
         "--save-predictions", str(out)], data_root)
    new = torch.load(out, weights_only=False)
    assert new["protein_ids"] == [str(i).strip() for i in saved["protein_ids"]]
    assert all(torch.equal(a, b) for a, b in zip(new["labels"], saved["labels"]))
    assert max((a - b).abs().max().item() for a, b in zip(new["logits"], saved["logits"])) < 1e-3


@slow
def test_training_runs_and_checkpoint_evaluates(data_root, tmp_path):
    cfg = stage2_config(data_root)
    import yaml
    data = yaml.safe_load(open(cfg))["data"]
    for split, n in (("train", 6), ("valid", 3)):
        pd.read_csv(os.path.join(data_root, data[f"{split}_csv"])).head(n).to_csv(tmp_path / f"{split}.csv", index=False)
    sets = [f"data.train_csv={tmp_path / 'train.csv'}", f"data.valid_csv={tmp_path / 'valid.csv'}",
            f"data.test_csv={tmp_path / 'valid.csv'}", "optim.epochs=2", f"logging.checkpoint_dir={tmp_path}",
            "run_name=smoke"]
    run(["compass.train.train_probe", "--config", cfg, "--encoder", "esm_raw", "--device", "cpu", "--set", *sets],
        data_root)
    assert (tmp_path / "smoke" / "best.pt").exists() and (tmp_path / "smoke" / "final.pt").exists()
    run(["compass.eval.evaluate", "--config", cfg, "--encoder", "esm_raw", "--device", "cpu", "--split", "test",
         "--probe-checkpoint", str(tmp_path / "smoke" / "best.pt"), "--n-bootstrap", "5", "--set", *sets], data_root)
