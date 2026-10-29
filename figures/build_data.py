"""Figure data (figures/data/*.json) from the analysis reports in results/, so a full rerun ends in the same plots.

  python figures/build_data.py [--results results] [--only easifa_comparison ...]

Numbers are rounded to three decimals (four for EasIFA's ablation), as printed in the figures.
"""
import argparse
import json
import os

import torch

R3 = lambda x: round(float(x), 3)
CI3 = lambda c: [R3(c[0]), R3(c[1])]
# EasIFA's trainable parameters, counted from its two released M-CSA checkpoints (all state-dict entries minus the
# 12,300 BatchNorm buffers; EasIFA fine-tunes everything, the language model at a 0.1 learning-rate ratio)
EASIFA_PARAMS = {"EasIFA ESM-bin": 675273846, "EasIFA SaProt-bin": 678016693}


def load(results, *path):
    with open(os.path.join(results, *path)) as f:
        return json.load(f)


def short(n):
    return f"{n / 1e6:.0f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


def param_lightweight(results, ckpt="checkpoints/main_stage2_mcsa/best.pt"):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["probe_state_dict"]
    ours = int(sum(v.numel() for v in sd.values()))
    bars = [{"name": "Flat (this work)", "value": ours, "label": short(ours), "color_key": "flat"}]
    bars += [{"name": k, "value": v, "label": short(v), "color_key": "baseline"} for k, v in EASIFA_PARAMS.items()]
    return {"xlabel": "Trainable parameters (log scale)", "bars": bars}


def easifa_comparison(results):
    rep = {v: load(results, "reports", f"easifa82_{v}_vs_main.json")["metrics"] for v in ("esm_bin", "saprot_bin")}
    metrics = []
    for key, name in (("macro_f1", "Macro F1"), ("macro_mcc", "Macro MCC"), ("macro_auprc", "Macro AUPRC")):
        e, s = rep["esm_bin"][key], rep["saprot_bin"][key]
        metrics.append({"name": name, "values": [R3(e["a"]), R3(s["a"]), R3(e["b"])],
                        "deltas": [{"vs": vs, "delta": R3(m["delta"]), "ci": CI3(m["ci95"]), "p_gt0": m["frac_delta_gt_0"]}
                                   for vs, m in (("ESM-bin", e), ("SaProt-bin", s))]})
    return {"xmax": 0.95, "series": [{"name": "EasIFA ESM-bin", "color_key": "baseline"},
                                     {"name": "EasIFA SaProt-bin", "color_key": "baseline_alt"},
                                     {"name": "Flat (this work)", "color_key": "flat"}],
            "metrics": metrics, "delta_xlim": [-0.02, 0.26]}


def easifa_ablation(results):
    r = load(results, "reports", "easifa_ablation.json")
    bands = ["0~40%", "40~60%", "60~80%"]
    R4 = lambda x: round(float(x), 4)
    series = []
    for comp, name, color in (("structure", "3D structure (GearNet)", "baseline"), ("reaction", "Reaction / ligand", "baseline_alt")):
        series.append({"name": name, "color_key": color, "values": [R4(r[comp][b]["gain"]) for b in bands],
                       "ci": [[R4(x) for x in r[comp][b]["ci95"]] for b in bands],
                       "_all": R4(r[comp]["all"]["gain"]), "_all_ci": [R4(x) for x in r[comp]["all"]["ci95"]]})
    return {"ylabel": "Macro F1 gain from component", "groups": ["<40%", "40-60%", "60-80%"],
            "n": [r["structure"][b]["n"] for b in bands], "ymax": 0.09, "series": series}


def plm_series(results):
    rows = [{"key": "esm2l30", "label": "ESM-2 650M · layer 30", "main": True}]
    for v, key, label in (("esm2_l33", "esm2_fix2", "ESM-2 650M · layer 33"), ("esm2_3b", "esm23b_fix2", "ESM-2 3B · layer 36"),
                          ("esmc_6b", "flat_fix2", "ESMC 6B · layer 80")):
        m = load(results, "reports", f"plm_kfold_test_main_vs_{v}.json")["metrics"]
        rows[0].update(macro_auprc=R3(m["macro_auprc"]["a"]), macro_f1=R3(m["macro_f1"]["a"]))
        rows.append({"key": key, "label": label, "main": False, "macro_auprc": R3(m["macro_auprc"]["b"]),
                     "macro_f1": R3(m["macro_f1"]["b"]), "delta_auprc": R3(m["macro_auprc"]["delta"]),
                     "ci_auprc": CI3(m["macro_auprc"]["ci95"]), "delta_f1": R3(m["macro_f1"]["delta"]),
                     "ci_f1": CI3(m["macro_f1"]["ci95"])})
    return {"rows": rows}


def structural_delta(results):
    rows = []
    for x, label in (("stage1", "63.8k proteins"), ("stage15", "33.5k proteins"), ("kfold", "770 (M-CSA, 5-fold)")):
        c = load(results, "reports", "structural", f"esmc_fix1_vs_structural_{x}.json")
        m = load(results, "reports", "structural", f"main_vs_structural_{x}.json")
        pc, pm = c["metrics"]["pooled_auprc"], m["metrics"]["pooled_auprc"]
        rows.append({"label": label, "delta": R3(pc["delta"]), "ci": CI3(pc["ci95"]), "n": c["n_shared_proteins"],
                     "flat": R3(pc["a"]), "structural": R3(pc["b"]), "paired": True,
                     "vs_main": {"delta": R3(pm["delta"]), "ci": CI3(pm["ci95"]), "n": m["n_shared_proteins"],
                                 "main": R3(pm["a"]), "structural": R3(pm["b"])}})
    return {"xlabel": "Δ pooled AUPRC (Structural − Flat), paired, 95% CI", "rows": rows}


def squidly_same_data(results):
    r = load(results, "reports", "squidly_catalodb_vs_ours.json")
    rows = [{"set": "CataloDB test", "n": r["n_shared_proteins"], "metric": name, "squidly": R3(r["metrics"][k]["a"]),
             "flat": R3(r["metrics"][k]["b"]), "delta": R3(r["metrics"][k]["delta"]), "ci": CI3(r["metrics"][k]["ci95"])}
            for k, name in (("macro_auprc", "Macro AUPRC"), ("macro_f1", "Macro F1"))]
    return {"rows": rows, "xmax": 1.0, "delta_xlim": [-0.06, 0.17]}


def p2rank_comparison(results):
    r = load(results, "reports", "p2rank", "results.json")["residue"]
    rows = []
    for key, label, ours in (("V/site", "Non-enzymes, all ligands", "ours_any"), ("V/metal", "Non-enzymes, metal ions", "ours_any"),
                             ("V/organic", "Non-enzymes, organic ligands", "ours_any"),
                             ("goliath/site", "Enzymes (Goliath), all sites", "ours_any"),
                             ("mcsa/site", "M-CSA, catalytic residues", "ours_final")):
        m = r[key]["pairs"][f"{ours} vs p2rank"]["macro_auprc"]
        rows.append({"key": key, "label": label, "n": r[key]["n"], "p2rank": R3(m["a"]), "ours": R3(m["b"]),
                     "delta": R3(m["delta"]), "ci": CI3(m["ci95"])})
    return {"rows": rows, "xmax": 1.12, "delta_xlim": [-0.2, 0.66]}


def negatives_control(results):
    return load(results, "reports", "negative_control.json")


def embedding_map(results):
    return load(results, "reports", "embedding_map.json")


def docking_paired(results):
    return load(results, "docking", "paired.json")


BUILDERS = {f.__name__: f for f in (param_lightweight, easifa_comparison, easifa_ablation, plm_series, structural_delta,
                                    squidly_same_data, p2rank_comparison, negatives_control, embedding_map, docking_paired)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out-dir", default="figures/data")
    ap.add_argument("--only", nargs="*", default=list(BUILDERS), choices=list(BUILDERS))
    args = ap.parse_args()
    for name in args.only:
        data = BUILDERS[name](args.results)
        with open(os.path.join(args.out_dir, f"{name}.json"), "w") as f:
            json.dump(data, f, indent=None if name == "embedding_map" else 1, ensure_ascii=False)
        print(f"[+] {args.out_dir}/{name}.json")


if __name__ == "__main__":
    main()
