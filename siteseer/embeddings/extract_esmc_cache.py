"""ESMC-6B embedding cache for a dataset CSV, in the layout CachedEmbeddingDataset reads:
<out-dir>/row_<_original_row_index>.pt with {"embedding": [L, 2560] fp16}, last hidden state without BOS/EOS,
truncated to 1,022 residues (siteseer.embeddings.embedders.ESMCEmbedder). Resumable: existing files are skipped.

  python -m siteseer.embeddings.extract_esmc_cache --csv dataset/label_diffusion/cached_dataframe_fully_excl_fix2.csv \\
      --out-dir data/embeddings_goliath_esmc
"""
import argparse
import os
import time

import pandas as pd
import torch

from siteseer.embeddings.embedders import get_embedder


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    d = pd.read_csv(args.csv)
    os.makedirs(args.out_dir, exist_ok=True)
    todo = [(int(i), s) for i, s in zip(d["_original_row_index"], d["aa_sequence"])
            if not os.path.exists(os.path.join(args.out_dir, f"row_{int(i)}.pt"))]
    print(f"[*] {len(d)} rows, {len(todo)} to embed -> {args.out_dir}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = get_embedder("esmc_6b")
    emb.load(device)
    t0 = time.time()
    for k, (i, seq) in enumerate(todo, 1):
        x = emb.embed(seq[:emb.max_len]).half()
        path = os.path.join(args.out_dir, f"row_{i}.pt")
        torch.save({"embedding": x}, path + ".tmp")
        os.replace(path + ".tmp", path)
        if k % 500 == 0 or k == len(todo):
            print(f"  {k}/{len(todo)} · {(time.time() - t0) / k:.2f} s/protein", flush=True)


if __name__ == "__main__":
    main()
