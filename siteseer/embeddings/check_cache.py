"""Check that an embedding cache covers every row a training CSV reads (missing rows are skipped silently by
CachedEmbeddingDataset). --quarantine-recent N renames unreadable files among the N newest to *.corrupt (-1: all).

  python -m siteseer.embeddings.check_cache --embedding-dir data/embeddings_goliath_esm2_l30 \\
      --csv dataset/label_diffusion/cached_dataframe_fully_excl_fix2.csv
"""
import argparse
import os
import sys

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedding-dir", required=True)
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--quarantine-recent", type=int, default=0)
    args = ap.parse_args()
    if args.quarantine_recent:
        import torch
        files = [os.path.join(args.embedding_dir, f) for f in os.listdir(args.embedding_dir) if f.endswith(".pt")]
        n = args.quarantine_recent
        for path in (files if n < 0 else sorted(files, key=os.path.getmtime)[-n:]):
            try:
                torch.load(path, map_location="cpu", weights_only=False)["embedding"]
            except Exception as e:
                os.rename(path, path + ".corrupt")
                print(f"[!] unreadable, renamed to {path}.corrupt ({str(e)[:80]})")
    have = {int(f[4:-3]) for f in os.listdir(args.embedding_dir) if f.startswith("row_") and f.endswith(".pt")}
    print(f"[*] {args.embedding_dir}: {len(have)} cached rows")
    n_gaps = 0
    for path in args.csv:
        df = pd.read_csv(path, usecols=lambda c: c in ("dataset_flag", "_original_row_index"))
        keys = df["_original_row_index"] if "_original_row_index" in df.columns else pd.Series(df.index, index=df.index)
        for split in args.splits:
            k = keys[df.dataset_flag == split]
            missing = sorted(set(k) - have)
            n_gaps += len(missing)
            print(f"  {os.path.basename(path)} {split}: {len(k)} rows, {len(missing)} missing"
                  + (f" (e.g. {missing[:5]})" if missing else ""))
    if n_gaps:
        print(f"[!] {n_gaps} rows without an embedding -- fill the cache first (the extraction scripts are resumable)")
        sys.exit(1)
    print("[*] cache complete for every requested split")


if __name__ == "__main__":
    main()
