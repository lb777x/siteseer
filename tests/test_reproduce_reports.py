"""The release statistics code reproduces reported comparisons exactly from the saved predictions."""
import json
import os
import subprocess
import sys

import pytest

from conftest import need

slow = pytest.mark.skipif(os.environ.get("SITESEER_SLOW") != "1", reason="slow; set SITESEER_SLOW=1")


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


@pytest.mark.parametrize("variant", ["esm2_l33", "esm2_3b", "esmc_6b"])
def test_plm_series_kfold(data_root, tmp_path, variant):
    ref = json.load(open(need(data_root, f"results/reports/plm_kfold_main_vs_{variant}.json")))
    a = [need(data_root, f"results/predictions/main_kfold{i}_test.pt") for i in range(5)]
    b = [need(data_root, f"results/predictions/{variant}_kfold{i}_test.pt") for i in range(5)]
    got = run_paired(a, b, ref["label_a"], ref["label_b"], ref["n_bootstrap"], str(tmp_path / "out.json"), data_root)
    assert got["n_shared_proteins"] == ref["n_shared_proteins"] == 962
    for key, r in ref["metrics"].items():
        assert got["metrics"][key]["ci95"] == pytest.approx(r["ci95"], abs=1e-12), key


def test_seed_summary(data_root, tmp_path):
    runs = ["main"] + [f"main_seed{s}" for s in (43, 44, 45, 46)]
    for r in runs:
        for i in range(5):
            need(data_root, f"results/predictions/{r}_kfold{i}_test.pt")
    out = tmp_path / "seeds.json"
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    subprocess.run([sys.executable, "-m", "siteseer.eval.seed_summary", "--runs", *runs,
                    "--sets", "kfold_test", "stage2_easifa82_test", "--out-json", str(out)],
                   check=True, cwd=data_root, env=env, capture_output=True)
    s = json.load(open(out))
    kf, e82 = s["kfold_test"]["summary"], s["stage2_easifa82_test"]["summary"]
    assert round(kf["macro_auprc"]["mean"], 3) == 0.776 and round(kf["macro_auprc"]["sd"], 3) == 0.001
    assert round(kf["macro_f1"]["mean"], 3) == 0.670
    assert round(e82["macro_f1"]["mean"], 3) == 0.721 and round(e82["macro_auprc"]["mean"], 3) == 0.813


@pytest.mark.parametrize("variant", ["esm_bin", "saprot_bin"])
def test_easifa82(data_root, tmp_path, variant):
    ref = json.load(open(need(data_root, f"results/reports/easifa82_{variant}_vs_main.json")))
    a = [need(data_root, f"data/comparisons/easifa/easifa82_{variant}_preds.pt")]
    b = [need(data_root, "results/predictions/main_stage2_easifa82_test.pt")]
    got = run_paired(a, b, ref["label_a"], ref["label_b"], ref["n_bootstrap"], str(tmp_path / "out.json"), data_root)
    assert got["n_shared_proteins"] == ref["n_shared_proteins"] == 82
    for key, r in ref["metrics"].items():
        assert got["metrics"][key]["ci95"] == pytest.approx(r["ci95"], abs=1e-12), key


def test_easifa_ablation(data_root, tmp_path):
    results = need(data_root, "data/external/easifa_results/results")
    out = tmp_path / "ablation.json"
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    subprocess.run([sys.executable, "-m", "siteseer.comparisons.easifa_ablation", "--results-dir", results,
                    "--out-json", str(out)], check=True, env=env, capture_output=True)
    r = json.load(open(out))
    # reported values (figure 3.9): gain in macro F1 with 95% CI
    assert round(r["structure"]["0~40%"]["gain"], 4) == 0.0160
    assert [round(x, 4) for x in r["structure"]["0~40%"]["ci95"]] == [-0.0005, 0.0326]
    assert [round(x, 4) for x in r["reaction"]["0~40%"]["ci95"]] == [0.0068, 0.0386]
    assert [round(x, 4) for x in r["reaction"]["all"]["ci95"]] == [0.0277, 0.0480]


@slow
def test_easifa_swissprot(data_root, tmp_path):
    ours = [need(data_root, f"results/predictions/{n}_swissprot_test.pt") for n in ("easifa_own", "main_stage2_mcsa")]
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    subprocess.run([sys.executable, "-m", "siteseer.comparisons.easifa_swissprot", "--ours", *ours,
                    "--subsets", "lt80_noleak", "--out-dir", str(tmp_path)],
                   check=True, cwd=data_root, env=env, capture_output=True)
    for theirs in ("esm_bin", "ng_bin"):
        for name in ("easifa_own", "main_stage2_mcsa"):
            ref = json.load(open(need(data_root, f"results/reports/easifa_swissprot/{theirs}_vs_{name}_lt80_noleak.json")))
            got = json.load(open(tmp_path / f"{theirs}_vs_{name}_lt80_noleak.json"))
            assert got["n_shared_proteins"] == ref["n_shared_proteins"] == 648
            for key, r in ref["metrics"].items():
                assert got["metrics"][key]["ci95"] == pytest.approx(r["ci95"], abs=1e-12), (theirs, name, key)
    # reported: trained on EasIFA's data, macro AUPRC 0.816 vs 0.865 (ESM-bin) and 0.834 (NG-bin)
    r = json.load(open(tmp_path / "esm_bin_vs_easifa_own_lt80_noleak.json"))["metrics"]["macro_auprc"]
    assert (round(r["b"], 3), round(r["a"], 3)) == (0.816, 0.865)
