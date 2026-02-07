# SDE-Rank Reproducibility Package (IJCS submission)

This package reproduces:
- Baselines: PageRank, HITS, Katz, LeaderRank, Degree, Betweenness, k-core
- SDE-Rank (S+B+E) and ablations: S-only, S+B, S+E
- Parameter sensitivity scans for gamma (emotion amplification) and lambda (time decay)
- IC diffusion simulation with emotion-aware edge activation probability

## Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Download public datasets (optional)
```bash
python scripts/download_datasets.py --out data --dataset all
```

## Run
```bash
python run_experiments.py --dataset email-eu-core --data_dir data --out results
```

Outputs:
- results/*_metrics.csv  (RMSE, Top-k overlap)
- results/*_sens_gamma.csv, *_sens_lambda.csv
- results/*_sens_gamma.png, *_sens_lambda.png

## Note
Public topologies lack message content/activity/sentiment labels; the package uses documented synthetic generators for these signals.
Replace them with real logs when available.
