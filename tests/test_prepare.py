"""Prepare pipeline tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import ohlc_path
from src.prepare import (
    _byte_ranges,
    _merge_bar_parts,
    _process_range,
    detect_price_decimals,
    load_ohlc,
    prepare_ohlc,
)


def test_detect_price_decimals_5dp(sample_ticks_5dp: Path):
    assert detect_price_decimals(sample_ticks_5dp) == 5


def test_detect_price_decimals_3dp(sample_ticks_3dp: Path):
    assert detect_price_decimals(sample_ticks_3dp) == 3


def test_detect_price_decimals_fails_without_fraction(tmp_path: Path):
    ticks_file = tmp_path / "ticks.csv"
    ticks_file.write_text(
        "2003.05.05 03:00:05.100,112161,112177\n"
        "2003.05.05 03:00:20.101,112162,112178\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Could not detect price decimals"):
        detect_price_decimals(ticks_file)


def test_prepare_ticks_decimals_fails_when_undetectable(tmp_path: Path):
    ticks_file = tmp_path / "ticks.csv"
    ticks_file.write_text(
        "2003.05.05 03:00:05.100,112161,112177\n"
        "2003.05.05 03:00:20.101,112162,112178\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Could not detect price decimals"):
        prepare_ohlc(
            timeframe="1h",
            ticks_file=ticks_file,
            ohlc_path=tmp_path / "out.csv",
            workers=1,
            decimals="ticks",
        )


def test_ohlc_path_naming():
    path = ohlc_path(timeframe="1h", symbol="EURUSD")
    assert path.name == "EURUSD_1h.csv"
    assert path.parent.name == "ohlc"


def test_prepare_requires_symbol_when_no_ticks_file(tmp_path: Path):
    with pytest.raises(ValueError, match="symbol is required"):
        prepare_ohlc(
            timeframe="1h",
            ohlc_path=tmp_path / "out.csv",
            workers=1,
            decimals=5,
        )


def test_load_ohlc_requires_symbol_and_timeframe_without_path():
    with pytest.raises(ValueError, match="symbol and timeframe are required"):
        load_ohlc()
    with pytest.raises(ValueError, match="symbol and timeframe are required"):
        load_ohlc(symbol="EURUSD")
    with pytest.raises(ValueError, match="symbol and timeframe are required"):
        load_ohlc(timeframe="1h")


def test_prepare_writes_named_file(sample_ticks_5dp: Path, tmp_path: Path):
    out = tmp_path / "EURUSD_1h.csv"
    df = prepare_ohlc(
        timeframe="1h",
        symbol="EURUSD",
        ticks_file=sample_ticks_5dp,
        ohlc_path=out,
        workers=2,
        decimals="ticks",
    )
    assert out.exists()
    assert len(df) > 0
    loaded = load_ohlc(path=out)
    assert list(loaded.columns) == ["datetime", "open", "high", "low", "close"]
    assert len(loaded) == len(df)


def test_prepare_decimals_ticks_vs_fixed(sample_ticks_5dp: Path, tmp_path: Path):
    out_ticks = tmp_path / "from_ticks.csv"
    out_2 = tmp_path / "dp2.csv"
    prepare_ohlc(
        timeframe="1h",
        ticks_file=sample_ticks_5dp,
        ohlc_path=out_ticks,
        workers=1,
        decimals="ticks",
    )
    prepare_ohlc(
        timeframe="1h",
        ticks_file=sample_ticks_5dp,
        ohlc_path=out_2,
        workers=1,
        decimals=2,
    )
    line_ticks = out_ticks.read_text(encoding="utf-8").splitlines()[0]
    line_2 = out_2.read_text(encoding="utf-8").splitlines()[0]
    # datetime,open,high,low,close
    open_ticks = line_ticks.split(",")[1]
    open_2 = line_2.split(",")[1]
    assert len(open_ticks.split(".")[1]) == 5
    assert len(open_2.split(".")[1]) == 2


def test_parallel_merge_matches_single_worker(sample_ticks_5dp: Path):
    def run(workers: int) -> pd.DataFrame:
        ranges = _byte_ranges(sample_ticks_5dp, workers)
        parts = [_process_range(str(sample_ticks_5dp), a, b, "1h") for a, b in ranges]
        return _merge_bar_parts(parts).sort_values("bar").reset_index(drop=True)

    one = run(1)
    many = run(4)
    assert len(one) == len(many)
    assert (one["bar"] == many["bar"]).all()
    for col in ("open", "high", "low", "close"):
        assert (one[col].round(8) == many[col].round(8)).all(), col


def test_ohlc_invariants(sample_ticks_5dp: Path, tmp_path: Path):
    out = tmp_path / "EURUSD_4h.csv"
    prepare_ohlc(
        timeframe="4h",
        ticks_file=sample_ticks_5dp,
        ohlc_path=out,
        workers=2,
        decimals=5,
    )
    df = load_ohlc(path=out)
    assert (df["high"] + 1e-12 >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] - 1e-12 <= df[["open", "close"]].min(axis=1)).all()
    assert (df["high"] >= df["low"]).all()


def test_prepare_unsupported_timeframe(sample_ticks_5dp: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        prepare_ohlc(
            timeframe="2h",  # type: ignore[arg-type]
            ticks_file=sample_ticks_5dp,
            ohlc_path=tmp_path / "x.csv",
            workers=1,
        )
