"""Fetch UniProt sequences of M3Site proteins missing from M3Site's split tables (batches of 300). Documents how
the frozen copy in the data release was made; a new fetch will differ as UniProt changes.

  python pipeline/fetch_m3site_sequences.py
"""
import argparse
import json
import os
import time

import pandas as pd
import requests

from common import EXTERNAL

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/accessions"


def fetch_batch(ids: list) -> dict:
    resp = requests.get(UNIPROT_URL, params={"accessions": ",".join(ids), "format": "fasta"}, timeout=60)
    resp.raise_for_status()
    seqs, current, chunks = {}, None, []
    for line in resp.text.splitlines():
        if line.startswith(">"):
            if current:
                seqs[current] = "".join(chunks)
            current = line.split("|")[1] if "|" in line else line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line.strip())
    if current:
        seqs[current] = "".join(chunks)
    return seqs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m3site-dir", default=f"{EXTERNAL}/m3site")
    ap.add_argument("--out", default=f"{EXTERNAL}/m3site/m3site_sequences_fetched.json")
    ap.add_argument("--batch-size", type=int, default=300)
    args = ap.parse_args()
    ids = set(json.load(open(f"{args.m3site_dir}/label.json")))
    known = set()
    for split in ("train_0.5.tsv", "test_0.5.tsv"):
        known |= set(pd.read_csv(f"{args.m3site_dir}/{split}", sep="\t")["Entry"])
    fetched = json.load(open(args.out)) if os.path.exists(args.out) else {}
    missing = sorted(ids - known - set(fetched))
    print(f"[*] {len(ids)} M3Site proteins, {len(known)} in the split tables, {len(missing)} to fetch")
    for i in range(0, len(missing), args.batch_size):
        try:
            fetched.update(fetch_batch(missing[i:i + args.batch_size]))
        except requests.RequestException as e:
            print(f"[!] batch {i // args.batch_size + 1} failed ({e}); rerun to retry")
        if (i // args.batch_size) % 10 == 9:
            json.dump(fetched, open(args.out, "w"))
        time.sleep(0.3)
    json.dump(fetched, open(args.out, "w"))
    print(f"[+] {args.out}: {len(fetched)} sequences")


if __name__ == "__main__":
    main()
