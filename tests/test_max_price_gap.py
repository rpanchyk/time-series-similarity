"""--max-price-gap filtering tests (OHLC close→open gaps)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from main import parse_args
from src.similarity import find_similar, price_gap_flags, segment_has_gap


def test_price_gap_flags_uses_prev_close_to_next_open():
    open_ = np.array([1.100, 1.101, 1.110, 1.111])
    close = np.array([1.101, 1.102, 1.111, 1.112])
    # gaps: |1.101-1.101|=0, |1.110-1.102|=0.008, |1.111-1.111|=0
    flags = price_gap_flags(open_, close, max_price_gap=0.005)
    assert list(flags) == [False, True, False]


def test_cli_max_price_gap_default_zero():
    args = parse_args(
        [
            "--similarity",
            "--symbol",
            "EURUSD",
            "--timeframe",
            "1h",
            "--search-type",
            "dtw",
        ]
    )
    assert args.max_price_gap == 0.0


def test_cli_max_price_gap_value():
    args = parse_args(
        [
            "--similarity",
            "--symbol",
            "EURUSD",
            "--timeframe",
            "1h",
            "--search-type",
            "dtw",
            "--max-price-gap",
            "0.00133",
        ]
    )
    assert abs(args.max_price_gap - 0.00133) < 1e-12


def test_cli_prepare_rejects_max_price_gap():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--prepare",
                "--symbol",
                "EURUSD",
                "--timeframe",
                "1h",
                "--max-price-gap",
                "0.001",
            ]
        )


def test_cli_max_price_gap_negative_rejected():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--similarity",
                "--symbol",
                "EURUSD",
                "--timeframe",
                "1h",
                "--search-type",
                "dtw",
                "--max-price-gap",
                "-0.1",
            ]
        )


def _ohlc_with_candle_gap() -> pd.DataFrame:
    """Continuous OHLC except a close→open gap between bars 10 and 11."""
    dates = pd.date_range("2024-01-02 00:00", periods=40, freq="h")
    close = np.full(40, 1.100, dtype=float)
    open_ = np.full(40, 1.100, dtype=float)
    # normal small body
    for i in range(40):
        open_[i] = 1.100 + i * 0.00001
        close[i] = open_[i] + 0.00005
    # plant matching patterns away from the gap
    pattern_o = np.array([1.1000, 1.1001, 1.1002, 1.1001, 1.1000])
    pattern_c = pattern_o + 0.00005
    open_[20:25] = pattern_o
    close[20:25] = pattern_c
    open_[30:35] = pattern_o
    close[30:35] = pattern_c
    # candle gap: prev close 1.100, next open 1.106 (|Δ|=0.006)
    close[10] = 1.100
    open_[11] = 1.106
    close[11] = 1.1065
    high = np.maximum(open_, close) + 0.0001
    low = np.minimum(open_, close) - 0.0001
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )


def test_max_price_gap_excludes_candidates():
    ohlc = _ohlc_with_candle_gap()
    flags = price_gap_flags(
        ohlc["open"].to_numpy(), ohlc["close"].to_numpy(), 0.004
    )
    assert flags[10]  # close[10] -> open[11]

    results = find_similar(
        ohlc,
        window=5,
        horizon=5,
        top_k=10,
        query_start=30,
        search_type="z-euclidean",
        max_price_gap=0.004,
    )
    for _, row in results.iterrows():
        s = int(row["match_start"])
        assert not segment_has_gap(flags, s, 5 + 5)


def test_max_price_gap_rejects_query():
    ohlc = _ohlc_with_candle_gap()
    with pytest.raises(ValueError, match="price gap"):
        find_similar(
            ohlc,
            window=5,
            horizon=5,
            query_start=8,  # includes gap between 10 and 11
            search_type="z-euclidean",
            max_price_gap=0.004,
        )


def test_max_price_gap_zero_disables_filter():
    ohlc = _ohlc_with_candle_gap()
    results = find_similar(
        ohlc,
        window=5,
        horizon=5,
        top_k=3,
        query_start=8,
        search_type="z-euclidean",
        max_price_gap=0.0,
    )
    assert len(results) == 3
