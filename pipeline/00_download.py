"""Step 00: download the raw sources and verify their SHA-256 checksums.

EasIFA's dataset.zip (Goliath pool, BLAST database, Swiss-Prot test set, M-CSA splits; MIT licence), EasIFA's
per-residue test predictions (EasIFA_results.zip, Zenodo 12819674, CC BY 4.0), CataloDB from Squidly's dataset bundle
(datasets.zip, Zenodo 15541320, CC BY 4.0, 4.8 GB; four files are kept), the protein list Squidly's released models
were trained on (WRiegs/Squidly, MIT, commit 00f1456) and the M3Site tables (commit 42b730a) come from their original
sources. BioLiP_nr.txt, the UniProt sequences of M3Site proteins and the UniProt queries behind the negative-control
sets change over time and come as frozen copies from the SiteSeer data release.

  python pipeline/00_download.py [--only easifa easifa_results catalodb squidly_train m3site release]
"""
import argparse
import hashlib
import os
import shutil
import urllib.request
import zipfile

EASIFA_URL = ("https://drive.usercontent.google.com/download?id=15c-KoZ47TpF9_qyQfJiY67gcgVZ8N5WR"
              "&export=download&confirm=t")
EASIFA_SHA256 = "40a38c68567a99cbabf02d6f58e683cc8fefd0379741f950506a015bf142f6d4"
EASIFA_RESULTS_URL = "https://zenodo.org/records/12819674/files/EasIFA_results.zip?download=1"
EASIFA_RESULTS_SHA256 = "96ab46a638e5a9d1d699633d946141512aba11ea8dd07c41e6098832053d7d39"
SQUIDLY_DATASETS_URL = "https://zenodo.org/records/15541320/files/datasets.zip?download=1"
SQUIDLY_DATASETS_SHA256 = "9499405fb05b86b918c558f374e621ecb7a2181b5f0c8398a3a0fa2ffee89c6c"
CATALODB_FILES = ["All_metadata.tsv", "final_test_set_post_structural_filtering.txt", "fastas/train_and_test.fasta",
                  "fastas/test.fasta"]
SQUIDLY_COMMIT = "00f14561cab08044c3d836ca5282b11a558544ac"
SQUIDLY_TRAIN = ("manuscript/data/Low30_mmseq_ID_exp_subset_train.csv",
                 "e5a3ac031d08348d6d8fc8b59a25f3107978b814dea482aa63f45402438dce10")
M3SITE_COMMIT = "42b730abb0ed649b7ebae80fcb4771e68b82c9db"
M3SITE_FILES = {
    "data/label.json": ("label.json", "ed0e2cce0ad80f2ce380c99e9da022b04b484b8216d727ecf5ec61d9328a046a"),
    "data/splits/train/train_0.5.tsv": ("train_0.5.tsv",
                                        "eb108630989c76fa917ae27ae507112cdb891b3c60b328c0da5a6906ef8cc64c"),
    "data/splits/test/test_0.5.tsv": ("test_0.5.tsv", "aa74ea6cc822653f2bba1afe907b8ebb43a47ecbc7eaac0d875e67914e62ed3c"),
}
# Base URL of the SiteSeer data release; set once the record is published.
DATA_RELEASE_URL = None
RELEASE_FILES = {
    "data/external/biolip/BioLiP_nr.txt": "4688b8c3c3acf68a6e3816780cc0ddbba8d2ba6aaa40a41daba741b099d33099",
    # UniProt queries of 2026-10-05 behind the negative-control sets (steps 17-19)
    "data/external/uniprot/uniprot_A.tsv.gz": "1398cf606c33678c76b70610aa2f551a249b85afec5e292c7f6d86e7b0e05509",
    "data/external/uniprot/uniprot_B.tsv.gz": "baeebcea01c3bc859593f36e1074b7544b07c4ab1fc4968f81edb3bc7e236689",
    "data/external/uniprot/uniprot_gold.tsv.gz": "77040a34da19b3779ade6a967c120ea8cd6a06c7d525ab0d8195c99b1c9193e5",
    "data/external/uniprot/swissprot_pfam_ec.tsv.gz": "09da1242152d29dfa1c9e63243e982a5719817777ad2092163254756a86c04dd",
    "data/external/m3site/m3site_sequences_fetched.json":
        "981ce26b0547764becb0a5a6aecce166b12724b45e09de32268c99250aa4cb33",
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, path: str, expected: str):
    if os.path.exists(path) and sha256(path) == expected:
        print(f"[*] {path} present, checksum ok")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"[*] downloading {url}")
    with urllib.request.urlopen(url) as r, open(path + ".part", "wb") as f:
        shutil.copyfileobj(r, f)
    got = sha256(path + ".part")
    if got != expected:
        raise SystemExit(f"checksum mismatch for {path}: {got} (expected {expected})")
    os.replace(path + ".part", path)
    print(f"[+] {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sources = ["easifa", "easifa_results", "catalodb", "squidly_train", "m3site", "release"]
    ap.add_argument("--only", nargs="*", choices=sources, default=sources)
    args = ap.parse_args()

    if "easifa" in args.only:
        zip_path = "data/external/easifa_dataset.zip"
        fetch(EASIFA_URL, zip_path, EASIFA_SHA256)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(".")
        print("[+] EasIFA dataset extracted to dataset/")
    if "easifa_results" in args.only:
        zip_path = "data/external/easifa_results.zip"
        fetch(EASIFA_RESULTS_URL, zip_path, EASIFA_RESULTS_SHA256)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall("data/external/easifa_results")
        print("[+] EasIFA results extracted to data/external/easifa_results/results/")
    if "catalodb" in args.only:
        zip_path = "data/external/squidly_datasets.zip"
        fetch(SQUIDLY_DATASETS_URL, zip_path, SQUIDLY_DATASETS_SHA256)
        with zipfile.ZipFile(zip_path) as z:
            for name in CATALODB_FILES:
                out = f"data/external/catalodb/{name}"
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with z.open(f"datasets/CataloDB/{name}") as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        print("[+] CataloDB files extracted to data/external/catalodb/ (the 4.8 GB zip can be deleted)")
    if "squidly_train" in args.only:
        remote, digest = SQUIDLY_TRAIN
        fetch(f"https://raw.githubusercontent.com/WRiegs/Squidly/{SQUIDLY_COMMIT}/{remote}",
              f"data/external/squidly/{os.path.basename(remote)}", digest)
    if "m3site" in args.only:
        for remote, (name, digest) in M3SITE_FILES.items():
            fetch(f"https://raw.githubusercontent.com/Gift-OYS/M3Site/{M3SITE_COMMIT}/{remote}",
                  f"data/external/m3site/{name}", digest)
    if "release" in args.only:
        if DATA_RELEASE_URL is None:
            raise SystemExit("DATA_RELEASE_URL is not set yet: place the frozen copies listed in RELEASE_FILES "
                             "manually, or rebuild them (pipeline/fetch_m3site_sequences.py; BioLiP_nr.txt from "
                             "zhanggroup.org/BioLiP, which will differ from the frozen copy)")
        for path, digest in RELEASE_FILES.items():
            fetch(f"{DATA_RELEASE_URL.rstrip('/')}/{path}", path, digest)


if __name__ == "__main__":
    main()
