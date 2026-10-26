"""Our models against EasIFA on its Swiss-Prot benchmark: paired bootstrap (2,000 draws over proteins) per EasIFA
variant and protein subset, EasIFA's labels on both sides, threshold 0.5. The reported subset is lt80_noleak (clean
for both training sets); see pipeline/13_easifa_swissprot.py for the subsets.

  python -m siteseer.comparisons.easifa_swissprot --ours results/predictions/easifa_own_swissprot_test.pt \\
      results/predictions/main_stage2_mcsa_swissprot_test.pt --out-dir results/reports/easifa_swissprot
"""
import argparse
import glob
import json
import os
import subprocess
import sys

THEIRS = ("esm_bin", "saprot_bin", "ng_bin")
KEYS = ("pooled_auprc", "macro_auprc", "macro_f1", "macro_mcc")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", nargs="+", required=True, help="our --save-predictions files on the benchmark")
    ap.add_argument("--easifa-dir", default="data/comparisons/easifa")
    ap.add_argument("--ids-dir", default="dataset/easifa")
    ap.add_argument("--subsets", nargs="*", default=None, help="default: every swissprot_ids_*.txt")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--out-dir", default="results/reports/easifa_swissprot")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    subsets = args.subsets or sorted(os.path.basename(p)[len("swissprot_ids_"):-4]
                                     for p in glob.glob(f"{args.ids_dir}/swissprot_ids_*.txt"))
    lines = ["| ours | EasIFA | subset | n | " + " | ".join(KEYS) + " |", "|---|---|---|---:|" + "---:|" * len(KEYS)]
    for ours in args.ours:
        name = os.path.basename(ours).removesuffix("_swissprot_test.pt").removesuffix(".pt")
        for theirs in THEIRS:
            for s in subsets:
                out = f"{args.out_dir}/{theirs}_vs_{name}_{s}.json"
                if not os.path.exists(out):
                    subprocess.run([sys.executable, "-m", "siteseer.eval.paired_bootstrap",
                                    "--a", f"{args.easifa_dir}/swissprot_{theirs}_preds.pt", "--label-a", f"easifa_{theirs}",
                                    "--b", ours, "--label-b", name, "--restrict-ids", f"{args.ids_dir}/swissprot_ids_{s}.txt",
                                    "--n-bootstrap", str(args.n_bootstrap), "--out-json", out],
                                   check=True, stdout=subprocess.DEVNULL)
                r = json.load(open(out))
                m = r["metrics"]
                cells = [f"{m[k]['b']:.3f} vs {m[k]['a']:.3f}, {m[k]['delta']:+.3f} [{m[k]['ci95'][0]:+.3f}, "
                         f"{m[k]['ci95'][1]:+.3f}]" for k in KEYS]
                lines.append(f"| {name} | {theirs} | {s} | {r['n_shared_proteins']} | " + " | ".join(cells) + " |")
    table = "\n".join(lines) + "\n"
    with open(f"{args.out_dir}/summary.md", "w") as f:
        f.write(table)
    print(table)


if __name__ == "__main__":
    main()
