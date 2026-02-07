import numpy as np
import networkx as nx


def degree_centrality(G: nx.DiGraph):
    # Out-degree for directed graph
    return dict(G.out_degree())


def betweenness(G: nx.DiGraph, k: int = 200, seed: int = 42, normalized: bool = True):
    """
    Approximate betweenness by sampling k sources (NetworkX supports this via k).  :contentReference[oaicite:4]{index=4}
    """
    n = G.number_of_nodes()
    if k is None or k <= 0:
        return nx.betweenness_centrality(G, normalized=normalized)
    k_eff = min(k, max(1, n - 1))
    return nx.betweenness_centrality(G, k=k_eff, normalized=normalized, seed=seed)


def k_core(G: nx.DiGraph):
    """
    core_number raises NetworkXNotImplemented if the graph contains self loops. :contentReference[oaicite:5]{index=5}
    """
    Gu = G.to_undirected()
    Gu.remove_edges_from(nx.selfloop_edges(Gu))
    return nx.core_number(Gu)


def pagerank(G: nx.DiGraph, alpha: float = 0.85):
    return nx.pagerank(G, alpha=alpha)


def hits_authority(G: nx.DiGraph, max_iter: int = 500, tol: float = 1e-6):
    """
    HITS uses power iteration and has no guarantee of convergence; it stops at max_iter
    or when number_of_nodes(G)*tol is reached. :contentReference[oaicite:6]{index=6}
    """
    hubs, auth = nx.hits(G, max_iter=max_iter, tol=tol, normalized=True)
    return auth


def katz_numpy_safe(G: nx.DiGraph, beta: float = 1.0, safety: float = 0.9):
    """
    NetworkX: alpha must be strictly less than 1/lambda_max for a solution. :contentReference[oaicite:7]{index=7}
    We set alpha = safety/lambda_max with safety<1 for robustness.
    """
    eigvals = nx.adjacency_spectrum(G)
    lam_max = float(np.max(np.abs(eigvals)))
    if lam_max <= 0:
        alpha = 0.1
    else:
        alpha = safety / lam_max
    return nx.katz_centrality_numpy(G, alpha=float(alpha), beta=beta)


def leaderrank(G: nx.DiGraph, tol: float = 1e-10, max_iter: int = 1000):
    """
    LeaderRank via adding a ground node connected bidirectionally to all nodes.
    Dense transition matrix implementation (OK for ~1k nodes).
    """
    nodes = list(G.nodes())
    n = len(nodes)

    ground = "__GROUND__"
    H = G.copy()
    H.add_node(ground)
    for u in nodes:
        H.add_edge(ground, u)
        H.add_edge(u, ground)

    idx = {u: i for i, u in enumerate(H.nodes())}
    m = len(idx)

    P = np.zeros((m, m), dtype=float)
    for u in H.nodes():
        out = list(H.successors(u))
        if not out:
            continue
        j = idx[u]
        inv = 1.0 / len(out)
        for v in out:
            i = idx[v]
            P[i, j] += inv

    s = np.ones(m, dtype=float) / m
    for _ in range(max_iter):
        s_new = P @ s
        if np.linalg.norm(s_new - s, 1) < tol:
            s = s_new
            break
        s = s_new

    g_idx = idx[ground]
    g_score = s[g_idx]
    return {u: float(s[idx[u]] + g_score / n) for u in nodes}
