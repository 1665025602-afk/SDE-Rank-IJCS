from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
OUTDIR = ROOT / "results" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

def load_csv(ds: str, kind: str):
    # kind in {"gamma","lambda"}
    p = TABLES / f"{ds}_sens_{kind}.csv"
    df = pd.read_csv(p)
    # 自动识别列名：通常是 gamma/rmse 或 lam/rmse 或 lambda/rmse
    cols = [c.lower() for c in df.columns]
    if "rmse" not in cols:
        raise ValueError(f"{p} missing RMSE column: {df.columns}")
    y = df[df.columns[cols.index("rmse")]]
    # x 列：优先 gamma，其次 lambda/lam
    for key in (kind, "lambda", "lam"):
        if key in cols:
            x = df[df.columns[cols.index(key)]]
            return x, y
    # 如果没找到，就默认第一列当 x
    return df.iloc[:, 0], y

def plot_compare(kind: str, out_name: str, title: str):
    plt.figure()
    for ds in ["email-eu-core", "wiki-vote"]:
        x, y = load_csv(ds, kind)
        plt.plot(x, y, marker="o", label=ds)
    plt.xlabel(kind)
    plt.ylabel("RMSE")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / out_name, dpi=300)
    print("Saved:", OUTDIR / out_name)

if __name__ == "__main__":
    plot_compare("gamma", "TIIS_Fig_sens_gamma_compare.png", "Sensitivity (RMSE vs gamma)")
    plot_compare("lambda", "TIIS_Fig_sens_lambda_compare.png", "Sensitivity (RMSE vs lambda)")
