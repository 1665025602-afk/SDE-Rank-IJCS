import numpy as np
import networkx as nx

def ic_spread(G: nx.DiGraph, seeds, p0: float, w_uv: dict, E_norm: dict, eta: float, rng: np.random.Generator):
    active = set(seeds)
    frontier = set(seeds)
    while frontier:
        new_frontier = set()
        for u in frontier:
            for v in G.successors(u):
                if v in active:
                    continue
                p = p0 * w_uv.get((u, v), 1.0) * (1.0 + eta * E_norm.get(u, 0.0))
                p = min(max(p, 0.0), 1.0)
                if rng.random() < p:
                    active.add(v)
                    new_frontier.add(v)
        frontier = new_frontier
    return len(active)

def estimate_influence(G: nx.DiGraph, n_trials: int, p0: float, eta: float, E_norm: dict, seed: int = 42):
    rng = np.random.default_rng(seed)
    w_uv = {}
    for u in G.nodes():
        out = list(G.successors(u))
        if not out:
            continue
        w = 1.0 / len(out)
        for v in out:
            w_uv[(u, v)] = w

    influence = {}
    for u in G.nodes():
        spreads = [ic_spread(G, [u], p0, w_uv, E_norm, eta, rng) for _ in range(n_trials)]
        influence[u] = float(np.mean(spreads))
    return influence
