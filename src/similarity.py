"""Time-series similarity search (window matching + horizon stats)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from src.config import DEFAULT_WORKERS, SEARCH_TYPES, TIMEFRAMES, resolve_workers
from src.metrics import (
    OHLC_STRUCT_METRICS,
    build_feature_windows,
    pairwise_distances,
)

# ProcessPool worker state (set via initializer; Windows-spawn friendly).
_WORKER_OHLC: pd.DataFrame | None = None
_WORKER_KWARGS: dict[str, Any] | None = None


def _init_find_similar_worker(ohlc: pd.DataFrame, kwargs: dict[str, Any]) -> None:
    global _WORKER_OHLC, _WORKER_KWARGS
    _WORKER_OHLC = ohlc
    _WORKER_KWARGS = kwargs


def _find_similar_worker(search_type: str) -> tuple[str, pd.DataFrame]:
    if _WORKER_OHLC is None or _WORKER_KWARGS is None:
        raise RuntimeError("similarity worker not initialized")
    results = find_similar(_WORKER_OHLC, search_type=search_type, **_WORKER_KWARGS)
    return search_type, results


def resolve_start_index(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp,
    timeframe: str,
    window: int,
) -> tuple[int, pd.Timestamp, pd.Timestamp]:
    """
    Quantize start_date to timeframe, map to bar index.

    Returns (index, quantized_ts, actual_bar_ts).
    If the exact quantized bar is missing (e.g. weekend gap), uses the first
    available bar at or after the quantized timestamp.
    """
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    start = pd.Timestamp(start_date)
    quantized = start.floor(TIMEFRAMES[timeframe])
    times = pd.to_datetime(df["datetime"])
    pos = int(times.searchsorted(quantized, side="left"))

    if pos >= len(df):
        raise ValueError(
            f"start-date {start} (quantized {quantized}) is after the last bar "
            f"({times.iloc[-1]})."
        )
    if pos + window > len(df):
        raise ValueError(
            f"Not enough bars after {times.iloc[pos]} for window={window} "
            f"(need {window}, have {len(df) - pos})."
        )

    actual = pd.Timestamp(times.iloc[pos])
    return pos, quantized, actual


def consecutive_pair_crosses_weekend(
    t0: pd.Timestamp, t1: pd.Timestamp
) -> bool:
    """True if Saturday or Sunday falls on a calendar day after t0 up to t1."""
    t0 = pd.Timestamp(t0)
    t1 = pd.Timestamp(t1)
    if t1 <= t0:
        return False
    day = t0.normalize() + pd.Timedelta(days=1)
    end = t1.normalize()
    while day <= end:
        if day.weekday() >= 5:  # Saturday=5, Sunday=6
            return True
        day += pd.Timedelta(days=1)
    return False


def weekend_gap_flags(dates) -> np.ndarray:
    """
    For each consecutive pair dates[i] -> dates[i+1], True if that step
    crosses a weekend. Length = len(dates) - 1.
    """
    times = pd.to_datetime(dates)
    n = len(times)
    if n < 2:
        return np.zeros(0, dtype=bool)
    flags = np.empty(n - 1, dtype=bool)
    for i in range(n - 1):
        flags[i] = consecutive_pair_crosses_weekend(times[i], times[i + 1])
    return flags


def segment_has_gap(gap_flags: np.ndarray, start: int, length: int) -> bool:
    """True if any flag inside [start, start+length) consecutive pairs is True."""
    if length < 2:
        return False
    end = start + length - 1
    if end > len(gap_flags):
        end = len(gap_flags)
    if start >= end:
        return False
    return bool(gap_flags[start:end].any())


def segment_has_weekend_gap(gap_flags: np.ndarray, start: int, length: int) -> bool:
    return segment_has_gap(gap_flags, start, length)


def price_gap_flags(
    open_: np.ndarray, close: np.ndarray, max_price_gap: float
) -> np.ndarray:
    """
    Classic OHLC candle gap: |open[i+1] - close[i]| > max_price_gap.
    Length = len(open_) - 1.
    """
    open_ = np.asarray(open_, dtype=float)
    close = np.asarray(close, dtype=float)
    if len(open_) != len(close):
        raise ValueError("open and close must have the same length")
    if len(open_) < 2:
        return np.zeros(0, dtype=bool)
    return np.abs(open_[1:] - close[:-1]) > float(max_price_gap)


def _forward_stats(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    dates: np.ndarray,
    match_end: int,
    horizon: int,
) -> dict:
    """
    Horizon stats after match_end using OHLC.

    Anchor = match close. max_dev_high from highs, max_dev_low from lows.
    """
    start = match_end + 1
    end = match_end + horizon  # inclusive index of last after-bar
    empty = {
        "after_end_time": pd.NaT,
        "max_dev_high": np.nan,
        "max_dev_high_time": pd.NaT,
        "max_dev_low": np.nan,
        "max_dev_low_time": pd.NaT,
    }
    if start >= len(close) or end >= len(close):
        return empty

    anchor = close[match_end]
    high_rets = (high[start : end + 1] / anchor - 1.0) * 100.0
    low_rets = (low[start : end + 1] / anchor - 1.0) * 100.0
    high_i = int(np.argmax(high_rets))
    low_i = int(np.argmin(low_rets))

    return {
        "after_end_time": dates[end],
        "max_dev_high": float(high_rets[high_i]),
        "max_dev_high_time": dates[start + high_i],
        "max_dev_low": float(low_rets[low_i]),
        "max_dev_low_time": dates[start + low_i],
    }


def find_similar(
    ohlc: pd.DataFrame,
    window: int = 20,
    stride: int = 1,
    top_k: int = 10,
    query_start: int | None = None,
    horizon: int = 20,
    price: str = "close",
    search_type: str = "z-euclidean",
    skip_weekends: bool = False,
    max_price_gap: float = 0.0,
) -> pd.DataFrame:
    """
    Find top-K windows most similar to the query window.

    Univariate search types use --price. Struct types (ohlc-joint, candle-shape,
    dtw-multiv) use full OHLC. Horizon extremes use high/low vs match close.
    """
    if price not in ("open", "high", "low", "close"):
        raise ValueError(f"Unsupported price column: {price}")
    if search_type not in SEARCH_TYPES:
        raise ValueError(
            f"Unsupported search type: {search_type}. "
            f"Choose from: {', '.join(sorted(SEARCH_TYPES))}"
        )
    if max_price_gap < 0:
        raise ValueError("max_price_gap must be >= 0 (0 disables the filter)")

    open_ = ohlc["open"].to_numpy(dtype=float)
    close = ohlc["close"].to_numpy(dtype=float)
    high = ohlc["high"].to_numpy(dtype=float)
    low = ohlc["low"].to_numpy(dtype=float)
    dates = ohlc["datetime"].to_numpy()
    price_values = ohlc[price].to_numpy(dtype=float)

    max_start = len(close) - window - horizon
    if max_start < 0:
        raise ValueError("Not enough data for the given window/horizon.")

    starts_all = np.arange(0, len(close) - window + 1, stride, dtype=int)
    if len(starts_all) < 2:
        raise ValueError("Not enough data for the given window/stride.")

    if query_start is None:
        query_idx = len(starts_all) - 1
    else:
        matches = np.where(starts_all == query_start)[0]
        if len(matches) == 0:
            raise ValueError(
                f"query_start={query_start} is not a valid window start "
                f"for window={window}, stride={stride}."
            )
        query_idx = int(matches[0])

    q_start = int(starts_all[query_idx])

    weekend_flags = weekend_gap_flags(dates) if skip_weekends else None
    gap_flags = (
        price_gap_flags(open_, close, max_price_gap) if max_price_gap > 0 else None
    )

    if skip_weekends and segment_has_gap(weekend_flags, q_start, window):
        raise ValueError(
            f"Query window starting at index {q_start} "
            f"({dates[q_start]}) crosses a weekend; "
            "pick another --start-date/--start-index or disable --skip-weekends."
        )
    if max_price_gap > 0 and segment_has_gap(gap_flags, q_start, window):
        raise ValueError(
            f"Query window starting at index {q_start} "
            f"({dates[q_start]}) has a price gap > {max_price_gap}; "
            "pick another --start-date/--start-index or raise --max-price-gap."
        )

    cand_mask = (starts_all <= max_start) & (np.abs(starts_all - q_start) >= window)
    span = window + horizon
    if skip_weekends:
        no_weekend = np.array(
            [not segment_has_gap(weekend_flags, int(s), span) for s in starts_all],
            dtype=bool,
        )
        cand_mask = cand_mask & no_weekend
    if max_price_gap > 0:
        no_price_gap = np.array(
            [not segment_has_gap(gap_flags, int(s), span) for s in starts_all],
            dtype=bool,
        )
        cand_mask = cand_mask & no_price_gap

    cand_starts = starts_all[cand_mask]
    if len(cand_starts) == 0:
        extras = []
        if skip_weekends:
            extras.append("without weekend gaps")
        if max_price_gap > 0:
            extras.append(f"without price gaps > {max_price_gap}")
        suffix = f" ({' and '.join(extras)})" if extras else ""
        raise ValueError(
            f"No eligible candidates with full forward horizon{suffix}."
        )

    # Build features only for the query and eligible candidates.
    query_feat = build_feature_windows(
        search_type,
        open_,
        high,
        low,
        close,
        price_values,
        np.array([q_start], dtype=int),
        window,
    )[0]
    cand_feats = build_feature_windows(
        search_type,
        open_,
        high,
        low,
        close,
        price_values,
        cand_starts,
        window,
    )
    distances = pairwise_distances(search_type, query_feat, cand_feats)
    order = np.argsort(distances)[:top_k]

    price_label = price if search_type not in OHLC_STRUCT_METRICS else "ohlc"

    rows = []
    for rank, idx in enumerate(order, start=1):
        s = int(cand_starts[idx])
        match_end = s + window - 1
        dist = float(distances[idx])
        fwd = _forward_stats(close, high, low, dates, match_end, horizon)
        rows.append(
            {
                "rank": rank,
                "distance": dist,
                "match_start": s,
                "match_end": match_end,
                "match_start_time": dates[s],
                "match_end_time": dates[match_end],
                "query_start": q_start,
                "query_end": q_start + window - 1,
                "query_start_time": dates[q_start],
                "query_end_time": dates[q_start + window - 1],
                "horizon": horizon,
                "price": price_label,
                "search_type": search_type,
                "skip_weekends": skip_weekends,
                "max_price_gap": max_price_gap,
                **fwd,
            }
        )

    return pd.DataFrame(rows)


def find_similar_many(
    ohlc: pd.DataFrame,
    search_types: list[str],
    *,
    workers: int | str = DEFAULT_WORKERS,
    window: int = 20,
    stride: int = 1,
    top_k: int = 10,
    query_start: int | None = None,
    horizon: int = 20,
    price: str = "close",
    skip_weekends: bool = False,
    max_price_gap: float = 0.0,
) -> list[tuple[str, pd.DataFrame]]:
    """
    Run find_similar for each search type.

    When workers > 1 and there are multiple types, uses ProcessPoolExecutor
    (one process per type, capped by workers). Results keep search_types order.
    """
    if not search_types:
        raise ValueError("search_types must be non-empty")

    common: dict[str, Any] = {
        "window": window,
        "stride": stride,
        "top_k": top_k,
        "query_start": query_start,
        "horizon": horizon,
        "price": price,
        "skip_weekends": skip_weekends,
        "max_price_gap": max_price_gap,
    }

    workers_n = min(resolve_workers(workers), len(search_types))
    if workers_n <= 1 or len(search_types) == 1:
        return [
            (st, find_similar(ohlc, search_type=st, **common)) for st in search_types
        ]

    by_type: dict[str, pd.DataFrame] = {}
    with ProcessPoolExecutor(
        max_workers=workers_n,
        initializer=_init_find_similar_worker,
        initargs=(ohlc, common),
    ) as pool:
        futures = {
            pool.submit(_find_similar_worker, st): st for st in search_types
        }
        for fut in as_completed(futures):
            st, results = fut.result()
            by_type[st] = results

    return [(st, by_type[st]) for st in search_types]
