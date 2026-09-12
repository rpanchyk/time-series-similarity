"""OHLC shard-merge rules and boundary-split tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.prepare import (
    _OHLC_COLS,
    _byte_ranges,
    _merge_bar_parts,
    _process_range,
)


def _shard(
    bar: str,
    first: str,
    last: str,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "bar": pd.Timestamp(bar),
                "first_time": pd.Timestamp(first),
                "last_time": pd.Timestamp(last),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
            }
        ],
        columns=_OHLC_COLS,
    )


def test_merge_empty_parts():
    out = _merge_bar_parts([])
    assert list(out.columns) == _OHLC_COLS
    assert out.empty

    out2 = _merge_bar_parts([pd.DataFrame(columns=_OHLC_COLS)])
    assert out2.empty


def test_merge_disjoint_bars_keeps_both():
    a = _shard(
        "2024-01-02 00:00",
        "2024-01-02 00:01",
        "2024-01-02 00:50",
        1.10,
        1.12,
        1.09,
        1.11,
    )
    b = _shard(
        "2024-01-02 01:00",
        "2024-01-02 01:01",
        "2024-01-02 01:50",
        1.11,
        1.13,
        1.10,
        1.12,
    )
    out = _merge_bar_parts([a, b])
    assert len(out) == 2
    assert list(out["bar"]) == [
        pd.Timestamp("2024-01-02 00:00"),
        pd.Timestamp("2024-01-02 01:00"),
    ]


def test_merge_same_bar_open_from_earliest_first_time():
    # Later shard listed first, but earlier first_time must win for open.
    late = _shard(
        "2024-01-02 04:00",
        "2024-01-02 05:00",
        "2024-01-02 07:50",
        open_=1.20,
        high=1.25,
        low=1.18,
        close=1.22,
    )
    early = _shard(
        "2024-01-02 04:00",
        "2024-01-02 04:05",
        "2024-01-02 04:50",
        open_=1.10,
        high=1.15,
        low=1.08,
        close=1.12,
    )
    out = _merge_bar_parts([late, early])
    assert len(out) == 1
    row = out.iloc[0]
    assert row["open"] == 1.10
    assert row["first_time"] == pd.Timestamp("2024-01-02 04:05")


def test_merge_same_bar_close_from_latest_last_time():
    early = _shard(
        "2024-01-02 04:00",
        "2024-01-02 04:05",
        "2024-01-02 04:50",
        open_=1.10,
        high=1.15,
        low=1.08,
        close=1.12,
    )
    late = _shard(
        "2024-01-02 04:00",
        "2024-01-02 05:00",
        "2024-01-02 07:50",
        open_=1.20,
        high=1.25,
        low=1.18,
        close=1.22,
    )
    out = _merge_bar_parts([early, late])
    row = out.iloc[0]
    assert row["close"] == 1.22
    assert row["last_time"] == pd.Timestamp("2024-01-02 07:50")


def test_merge_same_bar_high_max_low_min():
    a = _shard(
        "2024-01-02 04:00",
        "2024-01-02 04:05",
        "2024-01-02 04:50",
        open_=1.10,
        high=1.15,
        low=1.08,
        close=1.12,
    )
    b = _shard(
        "2024-01-02 04:00",
        "2024-01-02 05:00",
        "2024-01-02 07:50",
        open_=1.20,
        high=1.30,  # global high
        low=1.05,  # global low
        close=1.22,
    )
    out = _merge_bar_parts([a, b])
    row = out.iloc[0]
    assert row["high"] == 1.30
    assert row["low"] == 1.05
    assert row["open"] == 1.10
    assert row["close"] == 1.22


def test_merge_three_shards_same_bar():
    parts = [
        _shard(
            "2024-06-16 08:00",
            "2024-06-16 08:01",
            "2024-06-16 08:20",
            1.08,
            1.09,
            1.07,
            1.085,
        ),
        _shard(
            "2024-06-16 08:00",
            "2024-06-16 08:21",
            "2024-06-16 09:00",
            1.085,
            1.12,
            1.06,
            1.10,
        ),
        _shard(
            "2024-06-16 08:00",
            "2024-06-16 09:01",
            "2024-06-16 11:55",
            1.10,
            1.11,
            1.05,
            1.09,
        ),
    ]
    out = _merge_bar_parts(parts)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["open"] == 1.08
    assert row["close"] == 1.09
    assert row["high"] == 1.12
    assert row["low"] == 1.05
    assert row["first_time"] == pd.Timestamp("2024-06-16 08:01")
    assert row["last_time"] == pd.Timestamp("2024-06-16 11:55")


def test_merge_preserves_ohlc_invariants():
    parts = [
        _shard(
            "2024-01-02 00:00",
            "2024-01-02 00:01",
            "2024-01-02 00:30",
            1.10,
            1.11,
            1.09,
            1.105,
        ),
        _shard(
            "2024-01-02 00:00",
            "2024-01-02 00:31",
            "2024-01-02 00:59",
            1.105,
            1.20,
            1.00,
            1.15,
        ),
    ]
    out = _merge_bar_parts(parts)
    row = out.iloc[0]
    assert row["high"] >= max(row["open"], row["close"], row["low"])
    assert row["low"] <= min(row["open"], row["close"], row["high"])


def _write_ticks(path: Path, rows: list[tuple[str, float, float]]) -> None:
    lines = [f"{ts},{bid:.5f},{ask:.5f}" for ts, bid, ask in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_boundary_split_bar_matches_single_pass(tmp_path: Path):
    """Bar straddling a byte-range cut must merge to full-file OHLC."""
    ticks = [
        ("2024.01.02 04:00:01.000", 1.10000, 1.10010),
        ("2024.01.02 04:10:00.000", 1.10100, 1.10110),  # mid ~1.10105
        ("2024.01.02 05:00:00.000", 1.09900, 1.09910),  # low mid
        ("2024.01.02 06:00:00.000", 1.10500, 1.10510),  # high mid
        ("2024.01.02 07:50:00.000", 1.10300, 1.10310),  # close mid
    ]
    ticks_file = tmp_path / "ticks.csv"
    _write_ticks(ticks_file, ticks)

    full = _process_range(str(ticks_file), 0, ticks_file.stat().st_size, "4h")
    assert len(full) == 1

    # Force a mid-file cut after the 2nd line so shards share the 04:00 bar.
    size = ticks_file.stat().st_size
    mid = size // 2
    with ticks_file.open("rb") as f:
        f.seek(mid)
        f.readline()
        cut = f.tell()
    assert 0 < cut < size

    left = _process_range(str(ticks_file), 0, cut, "4h")
    right = _process_range(str(ticks_file), cut, size, "4h")
    assert not left.empty and not right.empty
    # Both shards should see the same 4h bucket.
    assert left.iloc[0]["bar"] == right.iloc[0]["bar"] == full.iloc[0]["bar"]

    merged = _merge_bar_parts([left, right])
    assert len(merged) == 1
    for col in ("open", "high", "low", "close"):
        assert abs(merged.iloc[0][col] - full.iloc[0][col]) < 1e-12, col
    assert merged.iloc[0]["first_time"] == full.iloc[0]["first_time"]
    assert merged.iloc[0]["last_time"] == full.iloc[0]["last_time"]


def test_byte_ranges_parallel_equals_full_file(sample_ticks_5dp: Path):
    ranges = _byte_ranges(sample_ticks_5dp, 5)
    assert len(ranges) >= 2
    # Ranges cover [0, size) without gaps/overlaps.
    assert ranges[0][0] == 0
    assert ranges[-1][1] == sample_ticks_5dp.stat().st_size
    for (a, b), (c, d) in zip(ranges, ranges[1:]):
        assert b == c
        assert a < b

    full = _process_range(
        str(sample_ticks_5dp), 0, sample_ticks_5dp.stat().st_size, "1h"
    )
    parts = [
        _process_range(str(sample_ticks_5dp), a, b, "1h") for a, b in ranges
    ]
    merged = _merge_bar_parts(parts).sort_values("bar").reset_index(drop=True)
    full = full.sort_values("bar").reset_index(drop=True)
    assert len(merged) == len(full)
    for col in ("open", "high", "low", "close"):
        assert (merged[col].round(10) == full[col].round(10)).all(), col
