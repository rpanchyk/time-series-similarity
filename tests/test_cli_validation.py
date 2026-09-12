"""CLI argument validation tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from main import cmd_prepare, cmd_similarity, parse_args
from src.config import resolve_workers


def test_prepare_defaults():
    args = parse_args(
        ["--prepare", "--symbol", "EURUSD", "--timeframe", "4h"]
    )
    assert args.prepare is True
    assert args.timeframe == "4h"
    assert args.symbol == "EURUSD"
    assert args.decimals == "ticks"
    assert args.workers == resolve_workers("max")


def test_prepare_requires_symbol(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--prepare", "--timeframe", "1h"])
    assert exc.value.code == 2
    assert "symbol" in capsys.readouterr().err.lower()


def test_prepare_requires_timeframe(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--prepare", "--symbol", "EURUSD"])
    assert exc.value.code == 2
    assert "timeframe" in capsys.readouterr().err.lower()


def test_prepare_requires_symbol_and_timeframe(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--prepare"])
    assert exc.value.code == 2
    err = capsys.readouterr().err.lower()
    assert "symbol" in err and "timeframe" in err


def test_similarity_requires_search_type():
    with pytest.raises(SystemExit):
        parse_args(
            ["--similarity", "--symbol", "EURUSD", "--timeframe", "1h"]
        )


def test_similarity_requires_symbol(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc:
        parse_args(
            ["--similarity", "--search-type", "z-euclidean", "--timeframe", "1h"]
        )
    assert exc.value.code == 2
    assert "symbol" in capsys.readouterr().err.lower()


def test_similarity_requires_timeframe(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc:
        parse_args(
            ["--similarity", "--symbol", "EURUSD", "--search-type", "z-euclidean"]
        )
    assert exc.value.code == 2
    assert "timeframe" in capsys.readouterr().err.lower()


def test_similarity_requires_symbol_and_timeframe(
    capsys: pytest.CaptureFixture[str],
):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--similarity", "--search-type", "z-euclidean"])
    assert exc.value.code == 2
    err = capsys.readouterr().err.lower()
    assert "symbol" in err and "timeframe" in err


def test_prepare_rejects_search_type():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--prepare",
                "--symbol",
                "EURUSD",
                "--timeframe",
                "1h",
                "--search-type",
                "z-euclidean",
            ]
        )


def test_prepare_rejects_start_date():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--prepare",
                "--symbol",
                "EURUSD",
                "--timeframe",
                "1h",
                "--start-date",
                "2024-01-01",
            ]
        )


def test_prepare_rejects_start_index():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--prepare",
                "--symbol",
                "EURUSD",
                "--timeframe",
                "1h",
                "--start-index",
                "10",
            ]
        )


def test_similarity_rejects_start_date_and_start_index():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--similarity",
                "--symbol",
                "EURUSD",
                "--timeframe",
                "1h",
                "--search-type",
                "z-euclidean",
                "--start-date",
                "2024-01-01",
                "--start-index",
                "10",
            ]
        )


def test_plots_must_be_positive():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--similarity",
                "--symbol",
                "EURUSD",
                "--timeframe",
                "1h",
                "--search-type",
                "z-euclidean",
                "--plots",
                "0",
            ]
        )


def test_output_flags_defaults_and_toggles():
    base = [
        "--similarity",
        "--symbol",
        "EURUSD",
        "--timeframe",
        "1h",
        "--search-type",
        "dtw",
    ]
    defaults = parse_args(base)
    assert defaults.output_image == "png"
    assert defaults.output_table == "file"
    assert defaults.output_table_format == "csv"

    off = parse_args(
        [*base, "--output-image", "skip", "--output-table", "skip"]
    )
    assert off.output_image == "skip"
    assert off.output_table == "skip"

    console = parse_args([*base, "--output-table", "console"])
    assert console.output_table == "console"

    json_fmt = parse_args([*base, "--output-table-format", "json"])
    assert json_fmt.output_table_format == "json"

    console_json = parse_args(
        [*base, "--output-table", "console", "--output-table-format", "json"]
    )
    assert console_json.output_table == "console"
    assert console_json.output_table_format == "json"


def test_output_table_format_rejected_with_skip():
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
                "--output-table",
                "skip",
                "--output-table-format",
                "json",
            ]
        )


def test_prepare_rejects_output_flags():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--prepare",
                "--symbol",
                "EURUSD",
                "--timeframe",
                "1h",
                "--output-image",
                "skip",
            ]
        )
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--prepare",
                "--symbol",
                "EURUSD",
                "--timeframe",
                "1h",
                "--output-table",
                "skip",
            ]
        )
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--prepare",
                "--symbol",
                "EURUSD",
                "--timeframe",
                "1h",
                "--output-table-format",
                "json",
            ]
        )


def test_decimals_ticks_and_int():
    base = ["--prepare", "--symbol", "EURUSD", "--timeframe", "1h"]
    assert parse_args([*base, "--decimals", "ticks"]).decimals == "ticks"
    assert parse_args([*base, "--decimals", "5"]).decimals == 5
    assert parse_args([*base, "--decimals", "0"]).decimals == 0


def test_decimals_invalid():
    base = ["--prepare", "--symbol", "EURUSD", "--timeframe", "1h"]
    with pytest.raises(SystemExit):
        parse_args([*base, "--decimals", "foo"])
    with pytest.raises(SystemExit):
        parse_args([*base, "--decimals", "-1"])


def test_workers_max_and_int():
    base = ["--prepare", "--symbol", "EURUSD", "--timeframe", "1h"]
    assert parse_args([*base, "--workers", "max"]).workers == max(
        1, os.cpu_count() or 1
    )
    assert parse_args([*base, "--workers", "4"]).workers == 4
    sim = parse_args(
        [
            "--similarity",
            "--symbol",
            "EURUSD",
            "--timeframe",
            "1h",
            "--search-type",
            "dtw,pearson",
            "--workers",
            "2",
        ]
    )
    assert sim.workers == 2
    assert sim.search_type == ["dtw", "pearson"]


def test_workers_invalid():
    base = ["--prepare", "--symbol", "EURUSD", "--timeframe", "1h"]
    with pytest.raises(SystemExit):
        parse_args([*base, "--workers", "0"])
    with pytest.raises(SystemExit):
        parse_args([*base, "--workers", "x"])


def test_unknown_timeframe():
    with pytest.raises(SystemExit):
        parse_args(["--prepare", "--symbol", "EURUSD", "--timeframe", "2h"])


def test_unknown_price():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--similarity",
                "--symbol",
                "EURUSD",
                "--timeframe",
                "1h",
                "--search-type",
                "z-euclidean",
                "--price",
                "mid",
            ]
        )


def test_unknown_search_type():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--similarity",
                "--symbol",
                "EURUSD",
                "--timeframe",
                "1h",
                "--search-type",
                "not-a-metric",
            ]
        )


def test_dtw_search_type_accepted():
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
    assert args.search_type == ["dtw"]
    assert args.symbol == "EURUSD"
    assert args.timeframe == "1h"


def test_multiple_search_types_accepted():
    args = parse_args(
        [
            "--similarity",
            "--symbol",
            "EURUSD",
            "--timeframe",
            "1h",
            "--search-type",
            "dtw, pearson,z-euclidean",
        ]
    )
    assert args.search_type == ["dtw", "pearson", "z-euclidean"]


def test_search_types_deduplicated():
    args = parse_args(
        [
            "--similarity",
            "--symbol",
            "EURUSD",
            "--timeframe",
            "1h",
            "--search-type",
            "dtw,dtw,pearson",
        ]
    )
    assert args.search_type == ["dtw", "pearson"]


def test_unknown_in_comma_list_search_type():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--similarity",
                "--symbol",
                "EURUSD",
                "--timeframe",
                "1h",
                "--search-type",
                "dtw,not-a-metric",
            ]
        )


def test_empty_search_type():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--similarity",
                "--symbol",
                "EURUSD",
                "--timeframe",
                "1h",
                "--search-type",
                ", ,",
            ]
        )


def test_symbol_uppercased():
    args = parse_args(
        ["--prepare", "--symbol", "eurusd", "--timeframe", "1h"]
    )
    assert args.symbol == "EURUSD"
    args = parse_args(
        [
            "--similarity",
            "--symbol",
            "eurusd",
            "--timeframe",
            "1h",
            "--search-type",
            "z-euclidean",
        ]
    )
    assert args.symbol == "EURUSD"


def test_missing_ticks_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "main.ticks_path", lambda symbol: tmp_path / f"{symbol}_ticks.csv"
    )
    args = parse_args(
        ["--prepare", "--symbol", "EURUSD", "--timeframe", "1h"]
    )
    with pytest.raises(SystemExit) as exc:
        cmd_prepare(args)
    assert "ticks file not found" in str(exc.value).lower()


def test_missing_ohlc_data_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    missing = tmp_path / "EURUSD_1h.csv"
    monkeypatch.setattr(
        "main.ohlc_path",
        lambda *, timeframe, symbol: missing,
    )
    args = parse_args(
        [
            "--similarity",
            "--search-type",
            "z-euclidean",
            "--symbol",
            "EURUSD",
            "--timeframe",
            "1h",
        ]
    )
    with pytest.raises(SystemExit) as exc:
        cmd_similarity(args)
    msg = str(exc.value).lower()
    assert "ohlc data not found" in msg
    assert "prepare" in msg
