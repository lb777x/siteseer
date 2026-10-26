"""Per-dimension mean / std of an embedding cache, from training proteins only.

Written to <cache>/norm_stats.pt ({"mean", "std", "n_proteins", "n_residues", "csv", "seed"}) and applied on loading
by compass.embeddings.embedders.apply_norm (data.norm_stats in the configs; also the esm2_650m_l30 embedder used for
inference): the same standardisation the linear layer screens used (StandardScaler), with statistics fixed once so
that training, evaluation and inference see identical inputs.

  python -m compass.embeddings.norm_stats --cache-dir data/embeddings_goliath_esm2_l30
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--csv", default="dataset/label_diffusion/cached_dataframe_fully_excl_fix2.csv")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = os.path.join(args.cache_dir, "norm_stats.pt")
    if os.path.exists(out):
        print(f"[*] {out} exists -- not overwritten (statistics must stay fixed once training used them)")
        return
    d = pd.read_csv(args.csv, usecols=lambda c: c in ("alphafolddb-id", "dataset_flag", "_original_row_index"))
    d = d[d.dataset_flag == "train"].drop_duplicates("alphafolddb-id")
    keys = d["_original_row_index"] if "_original_row_index" in d.columns else pd.Series(d.index, index=d.index)
    keys = np.random.default_rng(args.seed).permutation(keys.to_numpy())
    s = s2 = None
    n_prot = n_res = 0
    for k in keys:
        p = os.path.join(args.cache_dir, f"row_{k}.pt")
        if not os.path.exists(p):
            continue
        x = torch.load(p, map_location="cpu", weights_only=False)["embedding"].double()
        s = x.sum(0) if s is None else s + x.sum(0)
        s2 = (x * x).sum(0) if s2 is None else s2 + (x * x).sum(0)
        n_prot += 1
        n_res += x.shape[0]
        if n_prot == args.n:
            break
    mean = s / n_res
    std = (s2 / n_res - mean ** 2).clamp_min(0).sqrt()
    torch.save({"mean": mean.float(), "std": std.float(), "n_proteins": n_prot, "n_residues": n_res,
                "csv": args.csv, "seed": args.seed}, out)
    print(f"[*] {out}: {n_prot} train proteins, {n_res} residues; std median {std.median():.3f}, "
          f"max {std.max():.3f}; |mean| max {mean.abs().max():.1f}")


if __name__ == "__main__":
    main()
