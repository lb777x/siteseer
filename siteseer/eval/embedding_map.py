"""Embedding maps: what do the frozen ESM-2 embeddings already separate? (figures 3.21 and 3.22)

residues  Goliath stage-1 test proteins with both catalytic and binding annotations: catalytic-only, binding-only and
          non-site residues (all site residues, 20 random non-site residues per protein), in four representations:
          ESM-2 650M layer 33 (raw), layer 30 (standardised), and the 128-d hidden layer of the site-type model
          (stage 1) and of the main model (M-CSA stage) on layer 30.
proteins  mean-pooled layer-30 embedding (standardised) of the stage-1 test proteins with one EC class (first digit).
kNN purity (k = 10, Euclidean, original space): share of neighbours with the same class, per class and
class-balanced, next to chance. The 2-D UMAP (n_neighbors 15, min_dist 0.1, seed 42) only illustrates.

  python -m siteseer.eval.embedding_map extract      # ESM-2 650M layers 30 and 33 (GPU if available)
  python -m siteseer.eval.embedding_map project      # purity and maps -> results/reports/embedding_map.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from siteseer.data.labels import row_label_source
from siteseer.data.site_types import site_type_sets
from siteseer.embeddings.embedders import MAIN_NORM_STATS, load_esm2
from siteseer.models.probe import load_probe

ID = "alphafolddb-id"
CSV = "dataset/label_diffusion/cached_dataframe_fully_excl_fix2.csv"
CACHE = "data/embedding_map_cache.pt"
MAX_LEN = 1022


def extract(args):
    d = pd.read_csv(CSV, low_memory=False)
    t = d[d.dataset_flag == "test"].drop_duplicates(ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, batch_converter = load_esm2(device)
    res, prot = {}, {}
    for n, r in enumerate(t.to_dict("records")):
        seq = str(r["aa_sequence"])[:MAX_LEN]
        src = row_label_source(r)
        sets = site_type_sets(r["site_labels"], r.get("site_types"), src) if src == "goliath_original" else None
        with torch.no_grad():
            _, _, tok = batch_converter([("p", seq)])
            rep = model(tok.to(device), repr_layers=[30, 33])["representations"]
        l30, l33 = rep[30][0, 1:len(seq) + 1].half().cpu(), rep[33][0, 1:len(seq) + 1].half().cpu()
        prot[r[ID]] = {"ec": str(r.get("ec", "")), "mean30": l30.float().mean(0)}
        if sets is not None and sets["catalytic"] and sets["binding"]:
            res[r[ID]] = {"l30": l30, "l33": l33, "cat": sorted(sets["catalytic"]), "bind": sorted(sets["binding"]),
                          "ignore": sorted(sets["ignore"])}
        if (n + 1) % 100 == 0:
            print(f"[*] {n + 1}/{len(t)} proteins ({len(res)} with both types)", flush=True)
    torch.save({"residues": res, "proteins": prot}, args.cache)
    print(f"[+] {len(prot)} proteins, {len(res)} with catalytic and binding sites -> {args.cache}")


def knn_purity(X, y, k):
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    same = (y[idx] == y[:, None]).mean(1)
    per = {str(c): float(same[y == c].mean()) for c in np.unique(y)}
    chance = {str(c): float((y == c).mean()) for c in np.unique(y)}
    return {"per_class": per, "balanced": float(np.mean(list(per.values()))), "chance": chance,
            "chance_balanced": float(np.mean(list(chance.values())))}


def hidden(ckpt_path, X):
    probe, _ = load_probe(ckpt_path, torch.device("cpu"))
    with torch.no_grad():
        return probe.head[:-1](torch.from_numpy(X).float()).numpy()   # Linear -> ReLU -> Dropout (off): 128-d


def embed2d(X, args):
    import umap
    return umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, random_state=args.seed).fit_transform(X)


def project(args):
    cache = torch.load(args.cache, weights_only=False)
    st = torch.load(args.norm_stats, map_location="cpu", weights_only=False)
    mu, sd = st["mean"].float().numpy(), st["std"].float().clamp_min(1e-6).numpy()
    rng = np.random.default_rng(args.seed)
    X30, X33, y = [], [], []
    for r in cache["residues"].values():
        n = r["l30"].shape[0]
        cat, bind = set(r["cat"]), set(r["bind"])
        site = cat | bind | set(r["ignore"])
        lab = np.full(n, -1)
        for i in range(1, n + 1):
            if i in cat and i not in bind:
                lab[i - 1] = 1
            elif i in bind and i not in cat:
                lab[i - 1] = 2
            elif i not in site:
                lab[i - 1] = 0
        other = np.flatnonzero(lab == 0)
        keep = np.r_[np.flatnonzero(lab > 0), rng.choice(other, min(len(other), args.other_per_protein), replace=False)]
        X30.append((r["l30"].float().numpy()[keep] - mu) / sd)
        X33.append(r["l33"].float().numpy()[keep])
        y.append(lab[keep])
    X30, X33, y = np.concatenate(X30), np.concatenate(X33), np.concatenate(y)
    reps = {"layer33": X33, "layer30": X30, "hidden_sitetypes": hidden(args.sitetypes_ckpt, X30),
            "hidden_final": hidden(args.final_ckpt, X30)}
    classes = {"0": "non-site", "1": "catalytic", "2": "binding"}
    out = {"residues": {"n_proteins": len(cache["residues"]), "classes": classes,
                        "n_per_class": {str(c): int((y == c).sum()) for c in (0, 1, 2)}, "k": args.k, "maps": {}}}
    for name, X in reps.items():
        pur = knn_purity(X, y, args.k)
        out["residues"]["maps"][name] = {"purity": pur, "xy": np.round(embed2d(X, args), 3).tolist(), "y": y.tolist()}
        print(f"[residues] {name:17s} kNN purity {pur['balanced']:.3f} (chance {pur['chance_balanced']:.3f}) "
              + " ".join(f"{classes[c]} {v:.3f}" for c, v in pur["per_class"].items()), flush=True)
    accs, ecs, M = [], [], []
    for acc, p in cache["proteins"].items():
        first = {e.strip().split(".")[0] for e in p["ec"].replace(",", ";").split(";") if e.strip()}
        if len(first) == 1 and next(iter(first)).isdigit():
            accs.append(acc), ecs.append(int(next(iter(first)))), M.append((p["mean30"].numpy() - mu) / sd)
    M, ecs = np.stack(M), np.array(ecs)
    pur = knn_purity(M, ecs, args.k)
    out["proteins"] = {"n": len(accs), "n_per_ec": {str(c): int((ecs == c).sum()) for c in np.unique(ecs)},
                       "purity": pur, "xy": np.round(embed2d(M, args), 3).tolist(), "ec": ecs.tolist()}
    print(f"[proteins] {len(accs)} with one EC class; kNN purity {pur['balanced']:.3f} (chance {pur['chance_balanced']:.3f})")
    out["provenance"] = {"csv": CSV, "seed": args.seed, "other_per_protein": args.other_per_protein,
                         "n_neighbors": args.n_neighbors, "min_dist": args.min_dist}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"[+] {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["extract", "project"])
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--norm-stats", default=MAIN_NORM_STATS)
    ap.add_argument("--sitetypes-ckpt", default="checkpoints/sitetypes_stage1/best.pt")
    ap.add_argument("--final-ckpt", default="checkpoints/main_stage2_mcsa/best.pt")
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--other-per-protein", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/reports/embedding_map.json")
    args = ap.parse_args()
    {"extract": extract, "project": project}[args.step](args)


if __name__ == "__main__":
    main()
