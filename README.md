# SDE-Rank — Reproducibility Package (KSII TIIS submission)

This repository provides code and data to reproduce the experiments reported in our **KSII Transactions on Internet and Information Systems (TIIS)** submission on **SDE-Rank** (Structure–Dynamics–Emotion influence ranking).

## What can be reproduced
- **Baselines**: PageRank, HITS, Katz, LeaderRank, Degree, Betweenness (approx), k-core
- **SDE-Rank** (S+B+E) and **ablations**: S-only, S+B, S+E
- **Parameter sensitivity scans**:
  - gamma (emotion amplification)
  - lambda (time decay)
- **Influence ground-truth** estimation via Independent Cascade (IC) simulation

## Repository structure
- `run_experiments.py` — main entry to run experiments, baselines, ablations, and sensitivity scans
- `sde_rank.py` — SDE-Rank implementation
- `baselines.py` — baseline ranking methods
- `ic_simulation.py` — IC diffusion simulation for influence estimation
- `generators.py` — synthetic generators for activity/sentiment signals
- `data/` — datasets (edge lists)
- `scripts/` — helper scripts (download / merge / plotting)
- `results/`
  - `results/tables/` — CSV tables
  - `results/figures/` — PNG figures

## Quick Start + Outputs (copy & run)

> `--dataset` supports: `email-eu-core`, `wiki-vote`, or `path/to/edgelist.txt`.

~~~bash
# 1) Environment (Windows + Anaconda)
conda create -n sderank python=3.11 -y
conda activate sderank
pip install -r requirements.txt

# 2) Run (built-in datasets)
python run_experiments.py --dataset email-eu-core --out results
python run_experiments.py --dataset wiki-vote --out results

# 3) Outputs (paper-ready)
# After running the commands above, the following files will be generated (or updated) under results/.

# Figures
# - results/figures/FigX_a_gamma.png — RMSE vs γ (email-eu-core vs wiki-vote)
# - results/figures/FigX_b_lambda.png — RMSE vs λ (email-eu-core vs wiki-vote)
# - results/figures/email-eu-core_sens_gamma.png
# - results/figures/email-eu-core_sens_lambda.png
# - results/figures/wiki-vote_sens_gamma.png
# - results/figures/wiki-vote_sens_lambda.png

# Tables (CSV)
# - results/tables/TableX_metrics_compare.csv — overall comparison table used in the paper
# - results/tables/email-eu-core_metrics.csv
# - results/tables/email-eu-core_sens_gamma.csv
# - results/tables/email-eu-core_sens_lambda.csv
# - results/tables/wiki-vote_metrics.csv
# - results/tables/wiki-vote_sens_gamma.csv
# - results/tables/wiki-vote_sens_lambda.csv

# 4) Optional: download datasets
# If you prefer downloading datasets via script (instead of using included edge lists):
# python scripts/download_datasets.py --out data --dataset all

# 5) Notes / Assumptions
# Some public graph topologies do not provide message content, temporal activity logs, or sentiment labels.
# In such cases, this repository uses documented synthetic generators for activity/sentiment signals.
# Replace them with real logs when available.
~~~

## License
MIT License (see `LICENSE`).
