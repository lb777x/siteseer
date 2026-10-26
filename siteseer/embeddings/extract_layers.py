"""Hidden layers of ESM-2 650M or ESMC-6B for a random subset of proteins, one cache directory per layer.

Layer n is the output of block n (1-based); BOS/EOS are stripped, sequences capped at 1022 residues. ESM-2 comes from
transformers (facebook/esm2_t33_650M_UR50D, output_hidden_states); ESMC-6B through forward hooks on its block list.
Proteins: --n-train from the train split and --n-test from --test-splits, drawn with --seed. Writes
<out-root>/layer_<n>/row_<_original_row_index>.pt with {"embedding": [L, d] fp16}; existing rows are skipped.

  python -m siteseer.embeddings.extract_layers --model esm2_650m --layers 18 24 28 30 31 32 33 \\
      --csv dataset/label_diffusion/cached_dataframe_real_labels_only_fix1.csv --out-root data/esm2_layers \\
      --n-train 2500 --n-test 500
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from siteseer.embeddings.embedders import ESMC_MODEL_ID, ESMC_REVISION, MAX_SEQ_LEN, load_esmc_tokenizer

ID = "alphafolddb-id"
ESM2_HF_ID = "facebook/esm2_t33_650M_UR50D"


def pick_proteins(df, n_train, n_test, test_splits, seed):
    rng = np.random.default_rng(seed)
    picked = []
    for splits, n in ((["train"], n_train), (test_splits, n_test)):
        rows = df[df["dataset_flag"].isin(splits)].drop_duplicates(ID)
        picked.append(rows.iloc[rng.permutation(len(rows))].head(n))
        print(f"[*] {'+'.join(splits)}: {len(picked[-1])} proteins")
    return pd.concat(picked)


def esm2_backend(device, layers):
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(ESM2_HF_ID, dtype=torch.float16).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(ESM2_HF_ID)
    n = model.config.num_hidden_layers
    if not all(1 <= k <= n for k in layers):
        raise SystemExit(f"layers must lie in 1..{n}")

    def run(seq):
        out = model(**{k: v.to(device) for k, v in tokenizer(seq, return_tensors="pt").items()},
                    output_hidden_states=True)
        return {k: out.hidden_states[k][0] for k in layers}
    return run, lambda: None


def esmc_backend(device, layers):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(ESMC_MODEL_ID, revision=ESMC_REVISION, dtype=torch.float16,
                                      trust_remote_code=True).eval().to(device)
    tokenizer = load_esmc_tokenizer()
    # the block stack is the longest ModuleList in the remote model
    blocks = max((m for _, m in model.named_modules() if isinstance(m, torch.nn.ModuleList)), key=len)
    if not all(1 <= k <= len(blocks) for k in layers):
        raise SystemExit(f"layers must lie in 1..{len(blocks)}")
    captured = {}

    def hook_for(k):
        def hook(_m, _i, out):
            captured[k] = out[0] if isinstance(out, tuple) else out
        return hook
    handles = [blocks[k - 1].register_forward_hook(hook_for(k)) for k in layers]

    def run(seq):
        captured.clear()
        model(**{k: v.to(device) for k, v in tokenizer(seq, return_tensors="pt").items()})
        return {k: (h[0] if h.dim() == 3 else h) for k, h in captured.items()}
    return run, lambda: [h.remove() for h in handles]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["esm2_650m", "esmc_6b"])
    ap.add_argument("--layers", nargs="+", type=int, required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--test-splits", nargs="+", default=["test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    df = pd.read_csv(args.csv, low_memory=False)
    sub = pick_proteins(df, args.n_train, args.n_test, args.test_splits, args.seed)
    dirs = {k: os.path.join(args.out_root, f"layer_{k}") for k in args.layers}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    run, close = (esm2_backend if args.model == "esm2_650m" else esmc_backend)(device, args.layers)
    done = skipped = failed = 0
    try:
        with torch.inference_mode():
            for i, (_, row) in enumerate(sub.iterrows()):
                idx = int(row["_original_row_index"])
                paths = {k: os.path.join(d, f"row_{idx}.pt") for k, d in dirs.items()}
                if all(os.path.exists(p) for p in paths.values()):
                    skipped += 1
                    continue
                seq = row.get("aa_sequence")
                if not isinstance(seq, str) or not seq:
                    failed += 1
                    continue
                seq = seq[:MAX_SEQ_LEN]
                try:
                    hidden = run(seq)
                    for k, p in paths.items():
                        emb = hidden[k][1:len(seq) + 1].half().cpu()
                        if emb.shape[0] != len(seq):
                            raise RuntimeError(f"layer {k}: {emb.shape[0]} residues != {len(seq)}")
                        torch.save({"embedding": emb}, p)
                    done += 1
                except Exception as e:
                    failed += 1
                    print(f"[!] row {idx}: {type(e).__name__}: {e}")
                if (i + 1) % 250 == 0:
                    print(f"    {i + 1}/{len(sub)} ({done} written, {skipped} present, {failed} failed)", flush=True)
    finally:
        close()
    print(f"[+] {done} written, {skipped} present, {failed} failed")


if __name__ == "__main__":
    main()
