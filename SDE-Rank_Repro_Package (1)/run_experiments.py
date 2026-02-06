import argparse, os, time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from generators import generate_activity_series, generate_sentiment
from baselines import (
    degree_centrality, betweenness, k_core, pagerank, hits_authority,
    katz_numpy_safe, leaderrank
)
from sde_rank import sde_rank
from ic_simulation import estimate_influence


def load_edgelist(path: str, remove_selfloops: bool = True):
    G = nx.DiGraph()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            a, b = line.split()[:2]
            a = int(a); b = int(b)
            if remove_selfloops and a == b:
                continue
            G.add_edge(a, b)
    return G


def topk_overlap(rank_a, rank_b, k: int):
    a = [u for u, _ in rank_a[:k]]
    b = [u for u, _ in rank_b[:k]]
    return len(set(a).intersection(b)) / float(k)


def rmse(yhat, y):
    yhat = np.asarray(yhat, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.sqrt(np.mean((yhat - y) ** 2)))


def _timer(msg: str):
    print(msg, flush=True)
    return time.time()


def _done(t0: float, msg: str = "Done"):
    dt = time.time() - t0
    print(f"{msg}  (elapsed {dt:.2f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="email-eu-core | wiki-vote | path/to/edgelist.txt")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="results")

    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--p0", type=float, default=0.02)
    ap.add_argument("--eta", type=float, default=0.5)

    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)

    # Speed/robustness knobs
    ap.add_argument("--bw_k", type=int, default=200, help="Approx betweenness pivots k (100-300 recommended)")
    ap.add_argument("--skip_bw", action="store_true", help="Skip betweenness baseline")
    ap.add_argument("--skip_hits", action="store_true", help="Skip HITS baseline")
    ap.add_argument("--skip_katz", action="store_true", help="Skip Katz baseline")
    ap.add_argument("--skip_lr", action="store_true", help="Skip LeaderRank baseline")

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if os.path.exists(args.dataset):
        path = args.dataset
        ds_name = os.path.basename(path).split(".")[0]
    else:
        ds_name = args.dataset
        path = os.path.join(args.data_dir, f"{ds_name}.txt")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Edgelist not found: {path}")

    t0 = _timer(f"[1/6] Loading graph from: {path}")
    G = load_edgelist(path, remove_selfloops=True)

    # Extra safety: core_number does not permit self-loops. :contentReference[oaicite:8]{index=8}
    G.remove_edges_from(nx.selfloop_edges(G))

    nodes = list(G.nodes())
    n = len(nodes)
    _done(t0, f"Loaded {ds_name}: |V|={n}, |E|={G.number_of_edges()}")

    # Synthetic signals
    t0 = _timer("[2/6] Generating synthetic activity & sentiment")
    activity = generate_activity_series(n, T=args.T, seed=args.seed)
    sentiment = generate_sentiment(n, seed=args.seed)
    node_to_idx = {u: i for i, u in enumerate(nodes)}
    _done(t0, "Signals generated")

    # Structural score S from PageRank
    t0 = _timer("[3/6] Computing PageRank (structure S)")
    pr = pagerank(G)
    S = np.array([pr[u] for u in nodes], dtype=float)
    _done(t0, "PageRank computed")

    # Full SDE-Rank + components
    t0 = _timer("[4/6] Computing SDE-Rank (S+B+E)")
    R_full, comps = sde_rank(
        S, activity, sentiment,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, lam=args.lam
    )
    E_norm = {u: float(comps["E_norm"][node_to_idx[u]]) for u in nodes}
    _done(t0, "SDE-Rank computed")

    # Ground truth via IC simulation
    t0 = _timer("[5/6] Estimating influence ground-truth via IC simulation")
    gt = estimate_influence(
        G, n_trials=args.trials, p0=args.p0, eta=args.eta,
        E_norm=E_norm, seed=args.seed
    )
    y = np.array([gt[u] for u in nodes], dtype=float)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
    _done(t0, "Ground truth estimated")

    # Baselines
    baseline_scores = {}
    print("[6/6] Computing baselines (with progress)...", flush=True)

    t0 = _timer("  - Degree")
    deg = degree_centrality(G)
    baseline_scores["Degree"] = np.array([deg.get(u, 0) for u in nodes], dtype=float)
    _done(t0)

    if not args.skip_bw:
        t0 = _timer(f"  - Betweenness (approx k={args.bw_k})")
        bw = betweenness(G, k=args.bw_k, seed=args.seed, normalized=True)
        baseline_scores["Betweenness"] = np.array([bw.get(u, 0.0) for u in nodes], dtype=float)
        _done(t0)
    else:
        print("  - Betweenness skipped", flush=True)

    t0 = _timer("  - K-core")
    kc = k_core(G)  # internally removes self-loops in undirected view
    baseline_scores["K-core"] = np.array([kc.get(u, 0) for u in nodes], dtype=float)
    _done(t0)

    # PageRank already computed
    baseline_scores["PageRank"] = np.array([pr[u] for u in nodes], dtype=float)

    if not args.skip_hits:
        t0 = _timer("  - HITS-Authority")
        try:
            auth = hits_authority(G, max_iter=500, tol=1e-6)
            baseline_scores["HITS-Authority"] = np.array([auth.get(u, 0.0) for u in nodes], dtype=float)
            _done(t0)
        except Exception as e:
            print(f"    HITS failed: {e}  -> zeros", flush=True)
            baseline_scores["HITS-Authority"] = np.zeros(n, dtype=float)
    else:
        print("  - HITS skipped", flush=True)

    if not args.skip_katz:
        t0 = _timer("  - Katz (numpy safe alpha)")
        try:
            kz = katz_numpy_safe(G, beta=1.0, safety=0.9)
            baseline_scores["Katz"] = np.array([kz.get(u, 0.0) for u in nodes], dtype=float)
            _done(t0)
        except Exception as e:
            print(f"    Katz failed: {e}  -> zeros", flush=True)
            baseline_scores["Katz"] = np.zeros(n, dtype=float)
    else:
        print("  - Katz skipped", flush=True)

    if not args.skip_lr:
        t0 = _timer("  - LeaderRank")
        try:
            lr = leaderrank(G, tol=1e-10, max_iter=1000)
            baseline_scores["LeaderRank"] = np.array([lr.get(u, 0.0) for u in nodes], dtype=float)
            _done(t0)
        except Exception as e:
            print(f"    LeaderRank failed: {e}  -> zeros", flush=True)
            baseline_scores["LeaderRank"] = np.zeros(n, dtype=float)
    else:
        print("  - LeaderRank skipped", flush=True)

    # Ablations
    print("Computing ablations...", flush=True)
    R_S = (S - S.min()) / (S.max() - S.min() + 1e-12)  # S-only
    R_SB, _ = sde_rank(S, activity, sentiment, alpha=args.alpha, beta=args.beta, gamma=0.0, lam=args.lam)  # S+B
    B_const = np.ones_like(activity)
    R_SE, _ = sde_rank(S, B_const, sentiment, alpha=args.alpha, beta=args.beta, gamma=args.gamma, lam=1.0)  # S+E

    methods = {
        "SDE-Rank (S+B+E)": R_full,
        "S-only": R_S,
        "S+B": R_SB,
        "S+E": R_SE,
        **baseline_scores,
    }

    gt_rank = sorted(zip(nodes, y_norm), key=lambda x: x[1], reverse=True)

    rows = []
    for name, sc in methods.items():
        sc = np.asarray(sc, dtype=float)
        scn = (sc - sc.min()) / (sc.max() - sc.min() + 1e-12)
        sc_rank = sorted(zip(nodes, scn), key=lambda x: x[1], reverse=True)
        rows.append({
            "Method": name,
            "RMSE": rmse(scn, y_norm),
            "Top10Overlap": topk_overlap(sc_rank, gt_rank, min(10, n)),
            "Top50Overlap": topk_overlap(sc_rank, gt_rank, min(50, n)),
        })

    df = pd.DataFrame(rows).sort_values("RMSE")
    out_metrics = os.path.join(args.out, f"{ds_name}_metrics.csv")
    df.to_csv(out_metrics, index=False)
    print(f"\nWrote metrics: {out_metrics}", flush=True)

    # Sensitivity scans
    gammas = [0, 0.1, 0.2, 0.5, 1.0, 2.0]
    out_gamma = []
    for g in gammas:
        Rg, _ = sde_rank(S, activity, sentiment, alpha=args.alpha, beta=args.beta, gamma=g, lam=args.lam)
        Rgn = (Rg - Rg.min()) / (Rg.max() - Rg.min() + 1e-12)
        out_gamma.append({"gamma": g, "RMSE": rmse(Rgn, y_norm)})
    out_gamma_csv = os.path.join(args.out, f"{ds_name}_sens_gamma.csv")
    pd.DataFrame(out_gamma).to_csv(out_gamma_csv, index=False)
    print(f"Wrote sensitivity gamma: {out_gamma_csv}", flush=True)

    lams = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    out_lam = []
    for lam in lams:
        Rl, _ = sde_rank(S, activity, sentiment, alpha=args.alpha, beta=args.beta, gamma=args.gamma, lam=lam)
        Rln = (Rl - Rl.min()) / (Rl.max() - Rl.min() + 1e-12)
        out_lam.append({"lambda": lam, "RMSE": rmse(Rln, y_norm)})
    out_lam_csv = os.path.join(args.out, f"{ds_name}_sens_lambda.csv")
    pd.DataFrame(out_lam).to_csv(out_lam_csv, index=False)
    print(f"Wrote sensitivity lambda: {out_lam_csv}", flush=True)

    # Plots
    plt.figure()
    plt.plot([d["gamma"] for d in out_gamma], [d["RMSE"] for d in out_gamma], marker="o")
    plt.xlabel("gamma")
    plt.ylabel("RMSE")
    plt.title(f"Sensitivity to gamma ({ds_name})")
    plt.tight_layout()
    out_gamma_png = os.path.join(args.out, f"{ds_name}_sens_gamma.png")
    plt.savefig(out_gamma_png, dpi=200)

    plt.figure()
    plt.plot([d["lambda"] for d in out_lam], [d["RMSE"] for d in out_lam], marker="o")
    plt.xlabel("lambda")
    plt.ylabel("RMSE")
    plt.title(f"Sensitivity to lambda ({ds_name})")
    plt.tight_layout()
    out_lam_png = os.path.join(args.out, f"{ds_name}_sens_lambda.png")
    plt.savefig(out_lam_png, dpi=200)

    print(f"Wrote plots: {out_gamma_png}  and  {out_lam_png}", flush=True)

    print("\nTop methods by RMSE:")
    print(df.head(15).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
