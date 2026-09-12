"""Output naming and CSV column tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.output import (
    TABLE_COLUMNS,
    print_results_table,
    save_results_table,
    similarity_chart_path,
    similarity_output_path,
)


def _sample_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rank": 1,
                "distance": 0.1234,
                "match_start_time": pd.Timestamp("2024-01-02 00:00"),
                "match_end_time": pd.Timestamp("2024-01-02 04:00"),
                "max_dev_high": 1.5,
                "max_dev_high_time": pd.Timestamp("2024-01-02 05:00"),
                "max_dev_low": -2.0,
                "max_dev_low_time": pd.Timestamp("2024-01-02 06:00"),
                "query_start_time": pd.Timestamp("2025-06-16 08:00"),
                "query_end_time": pd.Timestamp("2025-06-16 12:00"),
                "query_start": 10,
                "query_end": 14,
                "horizon": 5,
            }
        ]
    )


def test_similarity_output_path_includes_symbol_tf_search_type_ts(tmp_path: Path):
    path = similarity_output_path(
        timeframe="1h",
        search_type="dtw",
        query_start_time="2025-06-16 08:00",
        suffix="csv",
        price="close",
        symbol="EURUSD",
        out_dir=tmp_path,
    )
    assert path.name == "EURUSD_1h_20250616-0800_dtw.csv"
    assert path.parent == tmp_path


def test_similarity_chart_path_png_naming_without_price(tmp_path: Path):
    path = similarity_chart_path(
        timeframe="1h",
        search_type="dtw",
        query_start_time=pd.Timestamp("2025-06-16 08:00"),
        price="close",
        symbol="EURUSD",
        out_dir=tmp_path,
    )
    assert path.name == "EURUSD_1h_20250616-0800_dtw.png"


def test_print_results_table_json(capsys: pytest.CaptureFixture[str]):
    print_results_table(_sample_results(), symbol="EURUSD", table_format="json")
    out = capsys.readouterr().out
    assert '"symbol": "EURUSD"' in out or '"symbol":"EURUSD"' in out
    assert '"rank": 1' in out or '"rank":1' in out


def test_save_results_table_name_and_columns(tmp_path: Path):
    results = _sample_results()
    out = save_results_table(
        results,
        timeframe="1h",
        search_type="z-euclidean",
        price="open",
        symbol="EURUSD",
        out_dir=tmp_path,
    )
    assert out.name == "EURUSD_1h_20250616-0800_z-euclidean.csv"
    assert out.exists()

    loaded = pd.read_csv(out)
    assert list(loaded.columns) == TABLE_COLUMNS
    assert "symbol" in loaded.columns
    assert loaded.iloc[0]["symbol"] == "EURUSD"
    assert loaded.iloc[0]["rank"] == 1


def test_save_results_table_json(tmp_path: Path):
    results = _sample_results()
    out = save_results_table(
        results,
        timeframe="1h",
        search_type="dtw",
        symbol="EURUSD",
        out_dir=tmp_path,
        table_format="json",
    )
    assert out.name == "EURUSD_1h_20250616-0800_dtw.json"
    loaded = pd.read_json(out)
    assert list(loaded.columns) == TABLE_COLUMNS
    assert loaded.iloc[0]["symbol"] == "EURUSD"
    assert int(loaded.iloc[0]["rank"]) == 1


def test_save_results_table_rejects_unknown_format(tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported table format"):
        save_results_table(
            _sample_results(),
            timeframe="1h",
            search_type="dtw",
            symbol="EURUSD",
            out_dir=tmp_path,
            table_format="xml",
        )