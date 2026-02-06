import numpy as np

def generate_activity_series(n_nodes: int, T: int = 30, seed: int = 42,
                             activity_power: float = 1.5, scale: float = 5.0):
    # Heavy-tailed per-node base rate + lognormal day-to-day noise
    rng = np.random.default_rng(seed)
    base = (rng.pareto(activity_power, size=n_nodes) + 1.0) * scale
    daily = rng.lognormal(mean=0.0, sigma=0.5, size=(n_nodes, T))
    activity = np.round(base[:, None] * daily).astype(int)
    activity[activity < 0] = 0
    return activity

def generate_sentiment(n_nodes: int, seed: int = 42, mode: str = "uniform"):
    # Sentiment in [-1, 1]
    rng = np.random.default_rng(seed + 1)
    if mode == "beta":
        x = rng.beta(2.0, 2.0, size=n_nodes)
        return 2.0 * x - 1.0
    return rng.uniform(-1.0, 1.0, size=n_nodes)
