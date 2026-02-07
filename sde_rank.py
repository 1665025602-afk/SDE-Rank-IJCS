import numpy as np

def minmax_norm(x: np.ndarray, eps: float = 1e-12):
    xmin, xmax = float(np.min(x)), float(np.max(x))
    return (x - xmin) / (xmax - xmin + eps)

def compute_B(activity: np.ndarray, lam: float = 0.9):
    n, T = activity.shape
    weights = np.array([lam ** (T - 1 - t) for t in range(T)], dtype=float)
    return activity @ weights

def sde_rank(S: np.ndarray, activity: np.ndarray, sentiment: np.ndarray,
             alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2, lam: float = 0.9):
    # R(u) = alpha*S'(u) + beta*B'(u)*(1 + gamma*E'(u)), E'(u)=|sentiment(u)|
    S1 = minmax_norm(S)
    B = compute_B(activity, lam=lam)
    B1 = minmax_norm(B)
    E = np.abs(sentiment)
    E1 = minmax_norm(E)
    R = alpha * S1 + beta * B1 * (1.0 + gamma * E1)
    return R, {"S_norm": S1, "B_norm": B1, "E_norm": E1}
