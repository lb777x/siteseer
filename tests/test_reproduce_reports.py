"""The release statistics code reproduces reported comparisons exactly from the saved predictions."""
import json
import os
import subprocess
import sys

import pytest

from conftest import need


def run_paired(a, b, label_a, label_b, n_bootstrap, out, cwd):
    cmd = [sys.executable, "-m", "siteseer.eval.paired_bootstrap", "--a", *a, "--b", *b, "--label-a", label_a,
           "--label-b", label_b, "--n-bootstrap", str(n_bootstrap), "--out-json", out]
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    subprocess.run(cmd, check=True, cwd=cwd, env=env, capture_output=True)
    return json.load(open(out))


def test_layer30_vs_layer33_kfold(data_root, tmp_path):
    ref = json.load(open(need(data_root, "results/reports/layer30_vs_layer33_kfold.json")))
    a = [need(data_root, f"results/predictions/esm2_l33_kfold{i}_test.pt") for i in range(5)]
    b = [need(data_root, f"results/predictions/main_kfold{i}_test.pt") for i in range(5)]
    got = run_paired(a, b, ref["label_a"], ref["label_b"], ref["n_bootstrap"], str(tmp_path / "out.json"), data_root)
    assert got["n_shared_proteins"] == ref["n_shared_proteins"] == 962
    for key, r in ref["metrics"].items():
        g = got["metrics"][key]
        assert g["a"] == pytest.approx(r["a"], abs=1e-12) and g["b"] == pytest.approx(r["b"], abs=1e-12), key
        assert g["ci95"] == pytest.approx(r["ci95"], abs=1e-12), key
