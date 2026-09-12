"""Horizon max_dev_high / max_dev_low checks on fixture OHLC (not data/ohlc)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.prepare import load_ohlc
from src.similarity import _forward_stats, find_similar, resolve_start_index


def _expected_horizon(
    ohlc: pd.DataFrame, match_end: int, horizon: int
) -> dict:
    close = ohlc["close"].to_numpy(float)
    high = ohlc["high"].to_numpy(float)
    low = ohlc["low"].to_numpy(float)
    dates = pd.to_datetime(ohlc["datetime"])
    anchor = close[match_end]
    h = high[match_end + 1 : match_end + 1 + horizon]
    l = low[match_end + 1 : match_end + 1 + horizon]
    path_dates = dates.iloc[match_end + 1 : match_end + 1 + horizon]
    high_rets = (h / anchor - 1.0) * 100.0
    low_rets = (l / anchor - 1.0) * 100.0
    return {
        "max_dev_high": float(high_rets.max()),
        "max_dev_low": float(low_rets.min()),
        "max_dev_high_time": pd.Timestamp(
            path_dates.iloc[int(np.argmax(high_rets))]
        ),
        "max_dev_low_time": pd.Timestamp(
            path_dates.iloc[int(np.argmin(low_rets))]
        ),
    }


def test_forward_stats_known_spike():
    """Unit check: planted high/low extremes map to correct % and times."""
    dates = pd.date_range("2024-01-02 00:00", periods=10, freq="h")
    # match_end=2, horizon=4 -> bars 3..6
    close = np.array(
        [1.10, 1.10, 1.10, 1.11, 1.12, 1.09, 1.105, 1.10, 1.10, 1.10],
        dtype=float,
    )
    high = close + 0.001
    low = close - 0.001
    high[4] = 1.21  # +10% vs anchor 1.10
    low[5] = 0.99  # -10% vs anchor
    fwd = _forward_stats(close, high, low, dates.to_numpy(), match_end=2, horizon=4)
    assert abs(fwd["max_dev_high"] - 10.0) < 1e-9
    assert abs(fwd["max_dev_low"] - (-10.0)) < 1e-9
    assert pd.Timestamp(fwd["max_dev_high_time"]) == dates[4]
    assert pd.Timestamp(fwd["max_dev_low_time"]) == dates[5]


def test_max_dev_from_fixture_ohlc(sample_ohlc_1h: Path):
    ohlc = load_ohlc(path=sample_ohlc_1h)
    assert len(ohlc) >= 60
    # Must not depend on data/ohlc/
    assert "data/ohlc" not in str(sample_ohlc_1h).replace("\\", "/")
    assert "fixtures" in str(sample_ohlc_1h).replace("\\", "/")

    window, horizon, top_k = 5, 5, 3
    q_start, _, _ = resolve_start_index(
        ohlc, "2024-01-02 10:00", "1h", window
    )
    assert q_start == 10

    results = find_similar(
        ohlc,
        window=window,
        stride=1,
        top_k=top_k,
        query_start=q_start,
        horizon=horizon,
        price="close",
    )
    assert len(results) == top_k

    for _, row in results.iterrows():
        exp = _expected_horizon(ohlc, int(row["match_end"]), horizon)
        assert abs(row["max_dev_high"] - exp["max_dev_high"]) < 1e-9
        assert abs(row["max_dev_low"] - exp["max_dev_low"]) < 1e-9
        assert pd.Timestamp(row["max_dev_high_time"]) == exp["max_dev_high_time"]
        assert pd.Timestamp(row["max_dev_low_time"]) == exp["max_dev_low_time"]


def test_max_dev_planted_match_in_fixture(sample_ohlc_1h: Path):
    """Match window at idx 40 has known high@47 / low@48 in horizon."""
    ohlc = load_ohlc(path=sample_ohlc_1h)
    window, horizon = 5, 5
    match_start = 40
    match_end = match_start + window - 1  # 44
    close = ohlc["close"].to_numpy(float)
    high = ohlc["high"].to_numpy(float)
    low = ohlc["low"].to_numpy(float)
    dates = ohlc["datetime"].to_numpy()

    fwd = _forward_stats(close, high, low, dates, match_end, horizon)
    exp = _expected_horizon(ohlc, match_end, horizon)
    assert abs(fwd["max_dev_high"] - exp["max_dev_high"]) < 1e-9
    assert abs(fwd["max_dev_low"] - exp["max_dev_low"]) < 1e-9
    # Planted extremes should dominate
    assert pd.Timestamp(fwd["max_dev_high_time"]) == pd.Timestamp(
        ohlc["datetime"].iloc[47]
    )
    assert pd.Timestamp(fwd["max_dev_low_time"]) == pd.Timestamp(
        ohlc["datetime"].iloc[48]
    )
    anchor = close[44]
    assert abs(fwd["max_dev_high"] - (high[47] / anchor - 1.0) * 100.0) < 1e-9
    assert abs(fwd["max_dev_low"] - (low[48] / anchor - 1.0) * 100.0) < 1e-9


def test_load_ohlc_fixture_not_ohlc_dir(sample_ohlc_1h: Path, tmp_path: Path):
    """Sanity: fixture loads via path override, independent of data/ohlc."""
    missing = tmp_path / "EURUSD_1h.csv"
    with pytest.raises(FileNotFoundError, match="OHLC data not found"):
        load_ohlc(path=missing)
    df = load_ohlc(path=sample_ohlc_1h)
    assert list(df.columns) == ["datetime", "open", "high", "low", "close"]
    assert (df["high"] >= df["low"]).all()
