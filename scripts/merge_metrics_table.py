
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"

def load(ds):
    df = pd.read_csv(T / f"{ds}_metrics.csv")
    df.insert(0, "Dataset", ds)
    return df

out = pd.concat([load("email-eu-core"), load("wiki-vote")], ignore_index=True)
out_path = T / "TableX_metrics_compare.csv"
out.to_csv(out_path, index=False)
print("Wrote:", out_path)
