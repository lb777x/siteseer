# SiteSeer: Enzyme Active Sites from Sequence with a Frozen Language Model

SiteSeer predicts which residues of an enzyme form its active site, from the amino-acid sequence alone. It reads
layer 30 of the frozen protein language model ESM-2 650M and classifies every residue with a two-layer MLP of 164K
trainable parameters, trained in three stages: a large pool of weakly labelled enzymes, curated labels, and the
Mechanism and Catalytic Site Atlas (M-CSA). No structure, reaction or multiple sequence alignment is needed.

Code by Luca Bohr. The paper and the data release are linked here on publication.

<details><summary><b>Citation</b></summary>

The citation follows on publication.
</details>

<details open><summary><b>Table of contents</b></summary>

- [Results](#results)
- [Usage](#usage)
  - [Quick start](#quickstart)
  - [Setup environment](#environment)
- [Datasets](#datasets)
- [Replicate results](#replicate)
- [Repository layout](#layout)
- [License](#license)
- [Acknowledgements](#acknowledgements)
</details>

## Results <a name="results"></a>

| Benchmark | SiteSeer | Compared with |
|---|---|---|
| M-CSA, 5-fold over 962 proteins, macro AUPRC / F1 | 0.776 / 0.670 | ESM-2 layer 33: 0.736 / 0.620; ESMC-6B: 0.709 / 0.591 |
| EasIFA's 82 M-CSA test proteins, macro F1 / AUPRC | 0.725 / 0.811 | EasIFA ESM-bin 0.586 / 0.673; SaProt-bin 0.613 / 0.705 |
| Squidly's CataloDB test set, same training data, macro AUPRC | 0.770 | Squidly 3B 0.735 |
| Confident non-enzymes predicted to have an active site | 10.8% | enzymes: 96.5-98.6% |
| DiffDock rank-1 pose at the annotated site, poses re-ranked by SiteSeer | 76.9% | DiffDock alone 65.3% |

Every comparison is paired over the same proteins (bootstrap, 95% CI). Over five training seeds the M-CSA macro
AUPRC is 0.776 ± 0.001.

## Usage <a name="usage"></a>

### Quick start <a name="quickstart"></a>

With the environment below and the trained model from the data release (`checkpoints/main_stage2_mcsa/best.pt`,
`data/embeddings_goliath_esm2_l30/norm_stats.pt`):

    python -m siteseer.predict proteins.fasta --out predictions.tsv

This prints the predicted active-site residues per protein and writes the probability of every residue to
`predictions.tsv`. For T4 lysozyme it returns Glu11 and Asp20, its two catalytic residues. ESM-2 downloads its weights
on first use (2.5 GB); a CPU is enough for single proteins. Sequences longer than 1,022 residues are truncated.

### Setup environment <a name="environment"></a>

    conda env create -f envs/siteseer.yml
    conda activate siteseer
    pip install -e .

The environment pins the versions used for the reported results (PyTorch 2.5.1 with CUDA 12.1 wheels, which also run
on CPU). The comparisons and the data pipeline need further environments:

    conda env create -f envs/siteseer-tools.yml    # BLAST+, MMseqs2, Java (data pipeline, P2Rank)
    bash envs/install_p2rank.sh                     # P2Rank 2.5.1
    bash envs/install_esmc.sh                       # ESMC-6B: language-model comparison, structural ablation
    bash envs/install_squidly.sh                    # Squidly 0.1.0 and its 3B weights
    bash envs/install_diffdock.sh                   # DiffDock at commit 85c49b6, for the docking application

## Datasets <a name="datasets"></a>

All data live under the repository root (`dataset/`, `data/`, `checkpoints/`, `results/`, ignored by git). Either
download the processed data from the data release, or rebuild them from the original sources:

    python pipeline/00_download.py
    bash pipeline/run_pipeline.sh

`00_download.py` fetches, with checksums:

- **EasIFA:** the dataset (Goliath pool, M-CSA splits, Swiss-Prot benchmark) and its published test predictions
  ([Zenodo 12819674](https://zenodo.org/records/12819674)).
- **M3Site:** residue labels (GitHub, commit 42b730a).
- **CataloDB:** Squidly's benchmark ([Zenodo 15541320](https://zenodo.org/records/15541320)) and the protein list its
  models were trained on (GitHub WRiegs/Squidly).
- **Frozen copies from the data release:** sources that change over time, namely BioLiP and the UniProt queries
  behind the negative control (2026-10-05).

`run_pipeline.sh` runs steps 01-19 (label transfer from M-CSA, M3Site and BioLiP, 90%-identity splits with MMseqs2,
exclusion of M-CSA and EasIFA test homologues, cross-validation folds, comparison and negative-control sets) and
checks every output against `pipeline/expected_sha256.txt`. Structures for the structural ablation and the docking
come from the AlphaFold DB:

    python -m siteseer.structural.download --csv <CSV files> --out-dir data/structures

## Replicate results <a name="replicate"></a>

Each experiment has one script in `jobs/`. They carry SLURM headers but also run with `bash`, from the repository
root, with the environment active. Each resumes where it stopped.

**Main model** (ESM-2 650M layer 30): embedding cache, three training stages, the M-CSA split and the 5-fold:

    sbatch jobs/chain.sh                       # or: MODEL=main bash jobs/chain.sh

**Language models** (figure 3.23), then the paired comparisons:

    for M in esm2_l33 esm2_3b esmc_6b; do sbatch --export=ALL,MODEL=$M jobs/chain.sh; done
    bash jobs/compare_plms.sh

**Further experiments**, in this order:

| Script | Reproduces | Needs |
|---|---|---|
| `jobs/seeds.sh` | seed spread | main model |
| `jobs/layer_screens.sh` | linear screens of ESM-2 and ESMC layers | ESMC and layer-33 chains |
| `jobs/same_data.sh`, `jobs/squidly.sh`, `jobs/compare_external.sh` | EasIFA and Squidly comparisons (figures 3.8, 3.9, 3.24) | main model |
| `jobs/sitetypes.sh` | catalytic vs binding outputs | main model |
| `jobs/negatives.sh` | negative control (figures 3.18-3.20) | site-type model |
| `jobs/p2rank.sh` | comparison with P2Rank (figure 3.25) | negative control |
| `jobs/embedding_map.sh` | embedding maps (figures 3.21, 3.22) | site-type model |
| `jobs/structural.sh` | structural ablation (figure 3.15) | ESMC chain |
| `jobs/docking.sh`, then `bash jobs/docking.sh post` | docking (figures 3.16, 3.17) | main model, structures |

**Figures:** `figures/data/` holds the numbers of every figure; rebuild them from your runs, then draw:

    python figures/build_data.py
    python figures/plot_plm_series.py            # one script per figure, written to figures/output/

**Single runs:** training and evaluation take a base config plus optional overlays and `--set` overrides:

    python -m siteseer.train.train_probe --config configs/main/kfold.yaml --set seed=43 run_name=kfold0_seed43
    python -m siteseer.eval.evaluate --config configs/main/kfold.yaml \
        --probe-checkpoint checkpoints/kfold0_seed43/best.pt --split test --save-predictions kfold0_seed43.pt
    python -m siteseer.eval.paired_bootstrap --a model_a.pt --b model_b.pt --n-bootstrap 2000

**Tests:** `pytest tests` runs the unit tests. With `SITESEER_DATA_ROOT` set to the data release, the tests also
recompute the reported numbers from the saved predictions; `SITESEER_SLOW=1` adds CPU inference on M-CSA and the
structural checks.

## Repository layout <a name="layout"></a>

    pipeline/    data acquisition and preprocessing (steps 00-20)
    siteseer/    data, embeddings, models, train, eval, comparisons, negatives, structural, docking, predict
    configs/     main model, language-model overlays, same-data runs, structural ablation
    jobs/        one script per experiment
    figures/     figure data and plot scripts
    envs/        pinned environments and install scripts
    tests/       unit and reproduction tests

## License <a name="license"></a>

The code is released under the MIT license, see [LICENSE](LICENSE). Data from other sources keep their terms:

| Source | Terms |
|---|---|
| EasIFA dataset and code | MIT |
| EasIFA's published predictions (Zenodo 12819674) | CC BY 4.0 |
| M-CSA | CC BY 4.0 |
| UniProtKB, AlphaFold DB | CC BY 4.0 |
| Squidly datasets (Zenodo 15541320) and code | CC BY 4.0, MIT |
| BioLiP2 | freely available, as stated by its providers |
| M3Site | no licence stated; its files are downloaded from the source, not redistributed |

## Acknowledgements <a name="acknowledgements"></a>

SiteSeer builds on ESM-2 (Meta AI), ESMC (EvolutionaryScale), M-CSA, UniProt, BioLiP, M3Site and the AlphaFold
Database. The comparisons use the data and released models of EasIFA, Squidly, P2Rank and DiffDock; we thank their
authors for making them available.
