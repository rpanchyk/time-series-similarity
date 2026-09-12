"""Weekend-gap filtering for similarity search."""

from __future__ import annotations

import pandas as pd
import pytest

from main import parse_args
from src.similarity import (
    consecutive_pair_crosses_weekend,
    find_similar,
    segment_has_weekend_gap,
    weekend_gap_flags,
)


def test_consecutive_pair_crosses_weekend_fri_to_mon():
    fri = pd.Timestamp("2024-01-05 20:00")  # Friday
    mon = pd.Timestamp("2024-01-08 00:00")  # Monday
    assert consecutive_pair_crosses_weekend(fri, mon) is True


def test_consecutive_pair_same_week_no_weekend():
    thu = pd.Timestamp("2024-01-04 10:00")
    fri = pd.Timestamp("2024-01-05 10:00")
    assert consecutive_pair_crosses_weekend(thu, fri) is False


def test_consecutive_pair_same_day_no_weekend():
    a = pd.Timestamp("2024-01-05 10:00")
    b = pd.Timestamp("2024-01-05 11:00")
    assert consecutive_pair_crosses_weekend(a, b) is False


def test_weekend_gap_flags_and_segment():
    dates = pd.to_datetime(
        [
            "2024-01-04 22:00",  # Thu
            "2024-01-05 22:00",  # Fri
            "2024-01-08 00:00",  # Mon  <-- weekend gap
            "2024-01-08 01:00",
            "2024-01-08 02:00",
        ]
    )
    flags = weekend_gap_flags(dates)
    assert list(flags) == [False, True, False, False]
    assert segment_has_weekend_gap(flags, 0, 3) is True  # includes Fri→Mon
    assert segment_has_weekend_gap(flags, 2, 3) is False  # Mon only


def _ohlc_with_weekend() -> pd.DataFrame:
    """Thu–Fri bars, weekend gap, then Monday+ bars."""
    week = pd.date_range("2024-01-04 08:00", periods=37, freq="h")  # ends Fri 20:00
    mon = pd.date_range("2024-01-08 00:00", periods=30, freq="h")
    dates = pd.DatetimeIndex(list(week) + list(mon))
    n = len(dates)
    close = 1.10 + 0.001 * pd.Series(range(n), dtype=float)
    pattern = [1.100, 1.101, 1.102, 1.101, 1.100]
    close.iloc[0:5] = pattern
    close.iloc[42:47] = pattern  # well after weekend (Mon starts at 37)
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": close,
            "high": close + 0.0005,
            "low": close - 0.0005,
            "close": close,
        }
    )


def test_skip_weekends_excludes_cross_weekend_candidates():
    ohlc = _ohlc_with_weekend()
    with_skip = find_similar(
        ohlc,
        window=5,
        horizon=5,
        top_k=5,
        query_start=42,
        search_type="z-euclidean",
        skip_weekends=True,
    )
    without = find_similar(
        ohlc,
        window=5,
        horizon=5,
        top_k=20,
        query_start=42,
        search_type="z-euclidean",
        skip_weekends=False,
    )
    flags = weekend_gap_flags(ohlc["datetime"])
    for _, row in with_skip.iterrows():
        s = int(row["match_start"])
        assert not segment_has_weekend_gap(flags, s, 5 + 5)
    assert len(without) >= len(with_skip)


def test_skip_weekends_rejects_query_crossing_weekend():
    ohlc = _ohlc_with_weekend()
    # start=34: bars include Fri evening and Monday across the gap
    with pytest.raises(ValueError, match="crosses a weekend"):
        find_similar(
            ohlc,
            window=5,
            horizon=5,
            query_start=34,
            search_type="z-euclidean",
            skip_weekends=True,
        )


def test_cli_skip_weekends_default_false():
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
    assert args.skip_weekends is False


def test_cli_skip_weekends_flag():
    args = parse_args(
        [
            "--similarity",
            "--symbol",
            "EURUSD",
            "--timeframe",
            "1h",
            "--search-type",
            "dtw",
            "--skip-weekends",
        ]
    )
    assert args.skip_weekends is True


def test_cli_prepare_rejects_skip_weekends():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--prepare",
                "--symbol",
                "EURUSD",
                "--timeframe",
                "1h",
                "--skip-weekends",
            ]
        )
