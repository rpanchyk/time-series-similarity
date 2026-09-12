"""Table and chart output for similarity results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR
from src.metrics import OHLC_STRUCT_METRICS, z_normalize


TABLE_COLUMNS = [
    "symbol",
    "rank",
    "distance",
    "match_start_time",
    "match_end_time",
    "max_dev_high",
    "max_dev_high_time",
    "max_dev_low",
    "max_dev_low_time",
    "query_start_time",
    "query_end_time",
]


def _with_symbol(results: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = results.copy()
    out["symbol"] = symbol
    return out


def print_results_table(
    results: pd.DataFrame,
    *,
    symbol: str,
    table_format: str = "csv",
) -> None:
    if results.empty:
        print("No matches found.")
        return

    table = _with_symbol(results, symbol)[TABLE_COLUMNS]
    fmt = str(table_format).strip().lower()
    if fmt == "json":
        print(table.to_json(orient="records", date_format="iso", indent=2))
        return
    if fmt != "csv":
        raise ValueError(f"Unsupported table format: {table_format}")

    display = table.copy()
    display["distance"] = display["distance"].map(lambda x: f"{x:.4f}")
    for col in ("max_dev_high", "max_dev_low"):
        display[col] = display[col].map(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "n/a"
        )
    print(display.to_string(index=False))


def similarity_output_path(
    timeframe: str,
    search_type: str,
    query_start_time,
    suffix: str,
    *,
    symbol: str,
    price: str = "close",
    out_dir: Path = OUTPUT_DIR,
) -> Path:
    # `price` kept for call-site compatibility; not part of the filename.
    _ = price
    ts = pd.Timestamp(query_start_time).strftime("%Y%m%d-%H%M")
    name = f"{symbol}_{timeframe}_{ts}_{search_type}.{suffix.lstrip('.')}"
    return out_dir / name


def save_results_table(
    results: pd.DataFrame,
    timeframe: str,
    search_type: str,
    *,
    symbol: str,
    price: str = "close",
    out_dir: Path = OUTPUT_DIR,
    table_format: str = "csv",
) -> Path:
    if results.empty:
        raise ValueError("Nothing to save.")
    fmt = str(table_format).strip().lower()
    if fmt not in ("csv", "json"):
        raise ValueError(f"Unsupported table format: {table_format}")
    out_dir.mkdir(parents=True, exist_ok=True)
    q_from = results.iloc[0]["query_start_time"]
    out_path = similarity_output_path(
        timeframe,
        search_type,
        q_from,
        fmt,
        price=price,
        symbol=symbol,
        out_dir=out_dir,
    )
    table = _with_symbol(results, symbol)[TABLE_COLUMNS]
    if fmt == "csv":
        table.to_csv(out_path, index=False)
    else:
        table.to_json(out_path, orient="records", date_format="iso", indent=2)
    print(f"Saved table -> {out_path}")
    return out_path


def similarity_chart_path(
    timeframe: str,
    search_type: str,
    query_start_time,
    *,
    symbol: str,
    price: str = "close",
    out_dir: Path = OUTPUT_DIR,
) -> Path:
    return similarity_output_path(
        timeframe,
        search_type,
        query_start_time,
        "png",
        price=price,
        symbol=symbol,
        out_dir=out_dir,
    )


def plot_matches(
    ohlc: pd.DataFrame,
    results: pd.DataFrame,
    window: int,
    timeframe: str,
    search_type: str,
    *,
    symbol: str,
    price: str = "close",
    horizon: int | None = None,
    out_dir: Path = OUTPUT_DIR,
    plots: int = 5,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    if results.empty:
        raise ValueError("Nothing to plot.")

    q_from = results.iloc[0]["query_start_time"]
    out_path = similarity_chart_path(
        timeframe,
        search_type,
        q_from,
        price=price,
        symbol=symbol,
        out_dir=out_dir,
    )

    if horizon is None:
        horizon = int(results.iloc[0]["horizon"])

    # Struct metrics match on OHLC features; overlay uses close for readability.
    viz_price = "close" if search_type in OHLC_STRUCT_METRICS else price
    series = ohlc[viz_price].to_numpy(dtype=float)
    close = ohlc["close"].to_numpy(dtype=float)
    high = ohlc["high"].to_numpy(dtype=float)
    low = ohlc["low"].to_numpy(dtype=float)

    q_start = int(results.iloc[0]["query_start"])
    query = z_normalize(series[q_start : q_start + window])

    x_match = np.arange(window)
    x_fwd = np.arange(window - 1, window + horizon)

    n = min(plots, len(results))
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(11, 2.8 * n + 0.6),
        sharex=True,
        layout="constrained",
    )
    if n == 1:
        axes = [axes]

    q_to = results.iloc[0]["query_end_time"]
    series_label = (
        f"{viz_price} (viz; match={search_type})"
        if search_type in OHLC_STRUCT_METRICS
        else viz_price
    )

    for i, (ax, (_, row)) in enumerate(zip(axes, results.head(n).iterrows())):
        m_start = int(row["match_start"])
        m_end = int(row["match_end"])
        match = z_normalize(series[m_start : m_start + window])

        ax.plot(
            x_match,
            query,
            label=f"query ({series_label})",
            color="#1f77b4",
            linewidth=2,
        )
        ax.plot(
            x_match,
            match,
            label=f"match ({series_label})",
            color="#ff7f0e",
            linewidth=2,
            alpha=0.9,
        )

        ax_r = ax.twinx()
        anchor = close[m_end]
        close_path = close[m_end : m_end + horizon + 1]
        high_path = np.concatenate([[anchor], high[m_end + 1 : m_end + 1 + horizon]])
        low_path = np.concatenate([[anchor], low[m_end + 1 : m_end + 1 + horizon]])
        close_rets = (close_path / anchor - 1.0) * 100.0
        high_rets = (high_path / anchor - 1.0) * 100.0
        low_rets = (low_path / anchor - 1.0) * 100.0

        ax_r.plot(
            x_fwd,
            close_rets,
            label="after close (%)",
            color="#2ca02c",
            linewidth=1.5,
            linestyle="--",
        )
        ax_r.plot(
            x_fwd,
            high_rets,
            label="after high (%)",
            color="#d62728",
            linewidth=1.2,
            alpha=0.85,
        )
        ax_r.plot(
            x_fwd,
            low_rets,
            label="after low (%)",
            color="#9467bd",
            linewidth=1.2,
            alpha=0.85,
        )
        ax_r.axhline(0.0, color="#2ca02c", linewidth=0.8, alpha=0.4)

        path_x = x_fwd[1:]
        if len(path_x) > 0:
            high_i = int(np.argmax(high_rets[1:]))
            low_i = int(np.argmin(low_rets[1:]))
            ax_r.scatter(
                [path_x[high_i]],
                [high_rets[1:][high_i]],
                color="#d62728",
                s=36,
                zorder=5,
                label="max_dev_high",
            )
            ax_r.scatter(
                [path_x[low_i]],
                [low_rets[1:][low_i]],
                color="#9467bd",
                s=36,
                zorder=5,
                label="max_dev_low",
            )

        ax_r.set_ylabel("after %", color="#2ca02c")
        ax.axvline(window - 1, color="gray", linestyle=":", linewidth=1.2)

        high_v = row["max_dev_high"]
        low_v = row["max_dev_low"]
        high_s = f"{high_v:+.2f}%" if pd.notna(high_v) else "n/a"
        low_s = f"{low_v:+.2f}%" if pd.notna(low_v) else "n/a"
        ax.set_title(
            f"#{int(row['rank'])}  dist={row['distance']:.4f}  "
            f"match {row['match_start_time']} -> {row['match_end_time']}\n"
            f"max_dev_high={high_s} @ {row['max_dev_high_time']}  |  "
            f"max_dev_low={low_s} @ {row['max_dev_low_time']}",
            fontsize=9,
            pad=6,
            loc="left",
        )
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("z-score")

        if i == 0:
            lines_l, labels_l = ax.get_legend_handles_labels()
            lines_r, labels_r = ax_r.get_legend_handles_labels()
            ax.legend(
                lines_l + lines_r,
                labels_l + labels_r,
                loc="upper left",
                fontsize=8,
                framealpha=0.9,
            )

    if search_type in OHLC_STRUCT_METRICS:
        price_note = f"overlay={viz_price} z-norm (match used {search_type})"
    else:
        price_note = f"price={viz_price} (z-normalized)"

    fig.suptitle(
        f"{symbol} | Query: {q_from} -> {q_to} | search-type={search_type} | "
        f"{price_note} | "
        f"horizon from OHLC (high/low extremes, close return)",
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved chart -> {out_path}")
    return out_path
