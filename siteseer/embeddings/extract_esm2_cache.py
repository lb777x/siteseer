"""ESM-2 embedding cache: <out-dir>/row_<key>.pt with {"embedding": [L, d] fp16}, not standardised.

Covers the train/valid/test rows of the given CSVs; the key is the row in cached_dataframe_final_resplit90.csv (or
`_original_row_index`). Identical sequences are hard-linked; existing files are skipped.

  python -m siteseer.embeddings.extract_esm2_cache --layer 30 --out-dir data/embeddings_goliath_esm2_l30
  python -m siteseer.embeddings.extract_esm2_cache --model esm2_t36_3B_UR50D --layer 36 --out-dir data/embeddings_goliath_esm2_3b
"""
import argparse
import os

import pandas as pd
import torch

SRC = "dataset/label_diffusion/cached_dataframe_final_resplit90.csv"
NEED = ["dataset/label_diffusion/cached_dataframe_fully_excl_fix2.csv",
        "dataset/label_diffusion/cached_dataframe_real_labels_only_fix2.csv"]
MAX_LEN = 1022


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="esm2_t33_650M_UR50D", help="fair-esm checkpoint, e.g. esm2_t36_3B_UR50D")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--need-csv", nargs="+", default=NEED)
    ap.add_argument("--own-sequences", action="store_true",
                    help="read sequences from the --need-csv rows (CSV with its own keys, e.g. CataloDB)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    need, own = set(), {}
    for p in args.need_csv:
        d = pd.read_csv(p, usecols=lambda c: c in ("dataset_flag", "_original_row_index", "aa_sequence"))
        keys = d["_original_row_index"] if "_original_row_index" in d.columns else pd.Series(d.index, index=d.index)
        sel = d.dataset_flag.isin(["train", "valid", "test"])
        need |= set(keys[sel].astype(int))
        if args.own_sequences:
            own.update(zip(keys[sel].astype(int), d.aa_sequence[sel]))
    seqs = pd.Series(own) if args.own_sequences else pd.read_csv(SRC, usecols=["aa_sequence"]).aa_sequence
    todo = [k for k in sorted(need) if not os.path.exists(os.path.join(args.out_dir, f"row_{k}.pt"))]
    print(f"[*] {len(need)} rows needed, {len(todo)} to compute ({args.model}, layer {args.layer})", flush=True)

    import esm
    model, alphabet = getattr(esm.pretrained, args.model)()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)
    bc = alphabet.get_batch_converter()
    if not 1 <= args.layer <= model.num_layers:
        raise SystemExit(f"--layer {args.layer} outside 1..{model.num_layers}")

    first_path = {}     # sequence -> file already written for it (hard-link duplicates)
    done = linked = failed = 0
    with torch.no_grad():
        for i, k in enumerate(todo):
            seq = seqs.loc[k] if args.own_sequences else seqs.iloc[k]
            out_path = os.path.join(args.out_dir, f"row_{k}.pt")
            if not isinstance(seq, str) or not seq:
                failed += 1
                continue
            seq = seq[:MAX_LEN]
            if seq in first_path:
                os.link(first_path[seq], out_path)
                linked += 1
                continue
            try:
                _, _, tokens = bc([("p", seq)])
                rep = model(tokens.to(device), repr_layers=[args.layer])["representations"][args.layer]
                emb = rep[0, 1:len(seq) + 1].half().cpu()
                if not torch.isfinite(emb).all():
                    raise RuntimeError("non-finite values in fp16")
                tmp = out_path + ".tmp"
                torch.save({"embedding": emb}, tmp)
                os.replace(tmp, out_path)          # never leaves a half-written row_*.pt behind
                first_path[seq] = out_path
                done += 1
            except Exception as e:
                failed += 1
                print(f"[!] row {k}: {str(e)[:120]}", flush=True)
            if (i + 1) % 5000 == 0:
                print(f"[*] {i + 1}/{len(todo)} (computed {done}, linked {linked}, failed {failed})", flush=True)
    print(f"[*] done: computed {done}, hard-linked {linked}, failed {failed}", flush=True)


if __name__ == "__main__":
    main()
