"""Pairwise distance helpers for similarity --search-type values."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.config import SEARCH_TYPES


SOFTDTW_GAMMA = 0.1

# Metrics that ignore --price and use full OHLC structure.
OHLC_STRUCT_METRICS = frozenset({"ohlc-joint", "candle-shape", "dtw-multiv"})

# Features already scale-free / pre-normalized — compare with plain Euclidean.
_PLAIN_EUCLIDEAN_METRICS = frozenset({"ohlc-joint", "candle-shape"})


def z_normalize(window: np.ndarray) -> np.ndarray:
    std = window.std()
    if std < 1e-12:
        return np.zeros_like(window, dtype=float)
    return (window - window.mean()) / std


def _z_normalize_rows(candidates: np.ndarray) -> np.ndarray:
    mean = candidates.mean(axis=1, keepdims=True)
    std = candidates.std(axis=1, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (candidates - mean) / std


def extract_windows(
    values: np.ndarray, starts: np.ndarray, window: int
) -> np.ndarray:
    """Stack windows of length `window` at arbitrary start indices."""
    if len(starts) == 0:
        return np.empty((0, window))
    return np.stack([values[int(s) : int(s) + window] for s in starts])


def sliding_windows(
    values: np.ndarray, window: int, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (starts, windows) where windows has shape (n, window)."""
    starts = np.arange(0, len(values) - window + 1, stride, dtype=int)
    return starts, extract_windows(values, starts, window)


def pairwise_euclidean(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Plain L2 distance (no z-normalization)."""
    q = np.asarray(query, dtype=float)
    return np.linalg.norm(candidates - q, axis=1)


def pairwise_z_euclidean(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    q = z_normalize(query)
    normalized = _z_normalize_rows(candidates)
    return np.linalg.norm(normalized - q, axis=1)


def pairwise_z_manhattan(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    q = z_normalize(query)
    normalized = _z_normalize_rows(candidates)
    return np.abs(normalized - q).sum(axis=1)


def pairwise_pearson(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Pearson distance = 1 - corr."""
    q = np.asarray(query, dtype=float)
    q = q - q.mean()
    q_std = q.std()
    if q_std < 1e-12:
        c_std = candidates.std(axis=1)
        return np.where(c_std < 1e-12, 0.0, 1.0)

    q = q / q_std
    mean = candidates.mean(axis=1, keepdims=True)
    std = candidates.std(axis=1, keepdims=True)
    safe_std = np.where(std < 1e-12, 1.0, std)
    c = (candidates - mean) / safe_std
    n = q.shape[0]
    corr = (c * q).sum(axis=1) / n
    corr = np.where(std.ravel() < 1e-12, 0.0, corr)
    corr = np.clip(corr, -1.0, 1.0)
    return 1.0 - corr


def pairwise_cosine(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Cosine distance = 1 - cosine similarity (no mean-centering)."""
    q = np.asarray(query, dtype=float)
    qn = np.linalg.norm(q)
    if qn < 1e-12:
        cn = np.linalg.norm(candidates, axis=1)
        return np.where(cn < 1e-12, 0.0, 1.0)
    q = q / qn
    cn = np.linalg.norm(candidates, axis=1, keepdims=True)
    cn = np.where(cn < 1e-12, 1.0, cn)
    c = candidates / cn
    sim = (c * q).sum(axis=1)
    sim = np.where(cn.ravel() < 1e-12, 0.0, sim)
    return 1.0 - np.clip(sim, -1.0, 1.0)


def _rankdata(x: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(x, dtype=float)).rank(method="average").to_numpy()


def pairwise_spearman(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Spearman distance = Pearson distance on ranks."""
    q_rank = _rankdata(query)
    c_ranks = np.vstack([_rankdata(c) for c in candidates])
    return pairwise_pearson(q_rank, c_ranks)


def _complexity(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.sqrt(np.sum(np.diff(x) ** 2)))


def pairwise_cid(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    Complexity-Invariant Distance on z-normalized series:
    ED(x,y) * max(CE(x),CE(y)) / min(CE(x),CE(y)).
    """
    q = z_normalize(query)
    ce_q = _complexity(q)
    normalized = _z_normalize_rows(candidates)
    ed = np.linalg.norm(normalized - q, axis=1)
    out = np.empty(len(candidates), dtype=float)
    for i in range(len(candidates)):
        ce_c = _complexity(normalized[i])
        denom = min(ce_q, ce_c)
        if denom < 1e-12:
            out[i] = ed[i]
        else:
            out[i] = ed[i] * (max(ce_q, ce_c) / denom)
    return out


def to_logret_windows(price_windows: np.ndarray) -> np.ndarray:
    """Convert (n, w) price windows to (n, w-1) log-return windows."""
    pw = np.asarray(price_windows, dtype=float)
    if np.any(pw <= 0):
        raise ValueError("logret requires strictly positive prices")
    return np.diff(np.log(pw), axis=1)


def _dtw(
    a: np.ndarray,
    b: np.ndarray,
    cost_fn: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """Classic DTW DP with pluggable local cost."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        raise ValueError("DTW requires non-empty sequences")

    prev = np.full(m + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr = np.full(m + 1, np.inf)
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = cost_fn(ai, b[j - 1])
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return float(prev[m])


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Classic DTW with absolute local cost."""
    return _dtw(a, b, lambda x, y: float(abs(float(x) - float(y))))


def pairwise_z_dtw(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    q = z_normalize(query)
    out = np.empty(len(candidates), dtype=float)
    for i in range(len(candidates)):
        out[i] = dtw_distance(q, z_normalize(candidates[i]))
    return out


def _softmin(values: np.ndarray, gamma: float) -> float:
    values = np.asarray(values, dtype=float)
    return float(-gamma * np.logaddexp.reduce(-values / gamma))


def softdtw_distance(
    a: np.ndarray, b: np.ndarray, gamma: float = SOFTDTW_GAMMA
) -> float:
    """Soft-DTW with squared local cost (Cuturi), after z-normalization."""
    a = z_normalize(np.asarray(a, dtype=float))
    b = z_normalize(np.asarray(b, dtype=float))
    n, m = len(a), len(b)
    R = np.full((n + 1, m + 1), np.inf)
    R[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            R[i, j] = cost + _softmin(
                np.array([R[i - 1, j], R[i, j - 1], R[i - 1, j - 1]]), gamma
            )
    return float(R[n, m])


def pairwise_z_softdtw(
    query: np.ndarray, candidates: np.ndarray, gamma: float = SOFTDTW_GAMMA
) -> np.ndarray:
    out = np.empty(len(candidates), dtype=float)
    for i in range(len(candidates)):
        out[i] = softdtw_distance(query, candidates[i], gamma=gamma)
    return out


def ohlc_joint_vector(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    start: int,
    window: int,
) -> np.ndarray:
    """Concat(z(close), z(high-low)) for one window."""
    sl = slice(start, start + window)
    cl = close[sl]
    rg = high[sl] - low[sl]
    return np.concatenate([z_normalize(cl), z_normalize(rg)])


def candle_shape_vector(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    start: int,
    window: int,
) -> np.ndarray:
    """Per-bar (body, upper_wick, lower_wick) / range, flattened."""
    feats: list[float] = []
    for i in range(start, start + window):
        rng = float(high[i] - low[i])
        if rng < 1e-12:
            body = upper = lower = 0.0
        else:
            o, h, l, c = float(open_[i]), float(high[i]), float(low[i]), float(close[i])
            body = (c - o) / rng
            upper = (h - max(o, c)) / rng
            lower = (min(o, c) - l) / rng
        feats.extend([body, upper, lower])
    return np.asarray(feats, dtype=float)


def multiv_window(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    start: int,
    window: int,
) -> np.ndarray:
    """(T, 4) window with each OHLC channel z-normalized."""
    sl = slice(start, start + window)
    return np.column_stack(
        [
            z_normalize(open_[sl]),
            z_normalize(high[sl]),
            z_normalize(low[sl]),
            z_normalize(close[sl]),
        ]
    )


def dtw_multiv_distance(a: np.ndarray, b: np.ndarray) -> float:
    """DTW on multivariate series; local cost = Euclidean across channels."""
    return _dtw(
        a,
        b,
        lambda x, y: float(np.linalg.norm(np.asarray(x) - np.asarray(y))),
    )


def pairwise_dtw_multiv(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """query (T, D), candidates (n, T, D)."""
    out = np.empty(len(candidates), dtype=float)
    for i in range(len(candidates)):
        out[i] = dtw_multiv_distance(query, candidates[i])
    return out


def build_feature_windows(
    search_type: str,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    price_values: np.ndarray,
    starts: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Build feature matrix for vector search types: shape (n_starts, feat_dim).

    For dtw-multiv returns shape (n_starts, window, 4).
    """
    if search_type == "logret":
        return to_logret_windows(extract_windows(price_values, starts, window))

    if search_type == "ohlc-joint":
        return np.stack(
            [
                ohlc_joint_vector(open_, high, low, close, int(s), window)
                for s in starts
            ]
        )

    if search_type == "candle-shape":
        return np.stack(
            [
                candle_shape_vector(open_, high, low, close, int(s), window)
                for s in starts
            ]
        )

    if search_type == "dtw-multiv":
        return np.stack(
            [multiv_window(open_, high, low, close, int(s), window) for s in starts]
        )

    return extract_windows(price_values, starts, window)


def pairwise_distances(
    search_type: str, query: np.ndarray, candidates: np.ndarray
) -> np.ndarray:
    """Dispatch distance for a query feature row vs candidate feature matrix."""
    if search_type in _PLAIN_EUCLIDEAN_METRICS:
        # Features already z-normalized / scale-free — do not z-norm again.
        return pairwise_euclidean(query, candidates)
    if search_type in ("z-euclidean", "logret"):
        # logret: z-Euclidean on the return series (intentional single z-norm).
        return pairwise_z_euclidean(query, candidates)
    if search_type == "z-manhattan":
        return pairwise_z_manhattan(query, candidates)
    if search_type == "pearson":
        return pairwise_pearson(query, candidates)
    if search_type == "spearman":
        return pairwise_spearman(query, candidates)
    if search_type == "cosine":
        return pairwise_cosine(query, candidates)
    if search_type == "cid":
        return pairwise_cid(query, candidates)
    if search_type == "dtw":
        return pairwise_z_dtw(query, candidates)
    if search_type == "softdtw":
        return pairwise_z_softdtw(query, candidates)
    if search_type == "dtw-multiv":
        return pairwise_dtw_multiv(query, candidates)
    raise ValueError(
        f"Unsupported search type: {search_type}. "
        f"Choose from: {', '.join(sorted(SEARCH_TYPES))}"
    )
