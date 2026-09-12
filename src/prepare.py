"""Prepare OHLC series from tick data (multiprocess-safe by TF buckets)."""

from __future__ import annotations

import io
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from src.config import (
    DEFAULT_WORKERS,
    TIMEFRAMES,
    resolve_workers,
    ticks_path,
    ohlc_path as resolve_ohlc_path,
)

_READ_CHUNK = 8 * 1024 * 1024  # 8 MiB text chunks inside a worker
_OHLC_COLS = ["bar", "first_time", "last_time", "open", "high", "low", "close"]
_DETECT_SAMPLE_BYTES = 1 * 1024 * 1024  # 1 MiB sample to infer price decimals


def detect_price_decimals(
    ticks_file: Path,
    sample_bytes: int = _DETECT_SAMPLE_BYTES,
) -> int:
    """
    Infer decimal places from tick bid/ask text (same precision as source ticks).
    Uses the maximum fractional length found in the sample.
    Raises ValueError if no fractional prices are found.
    """
    with ticks_file.open("rb") as f:
        sample = f.read(sample_bytes)
    text = sample.decode("utf-8", errors="replace")
    last_nl = text.rfind("\n")
    if last_nl >= 0:
        text = text[: last_nl + 1]

    max_dp = 0
    found = False
    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue
        for price in parts[1:3]:
            price = price.strip()
            if "." not in price:
                continue
            frac = price.split(".", 1)[1]
            # Ignore trailing spaces / non-digits just in case.
            digits = "".join(ch for ch in frac if ch.isdigit())
            if digits:
                found = True
                max_dp = max(max_dp, len(digits))

    if not found:
        raise ValueError(
            f"Could not detect price decimals from {ticks_file}. "
            "Tick bid/ask sample has no fractional prices; "
            "pass --decimals N explicitly."
        )
    return max_dp


def _parse_tick_text(text: str) -> pd.DataFrame:
    if not text.strip():
        return pd.DataFrame(columns=["datetime", "mid"])
    chunk = pd.read_csv(
        io.StringIO(text),
        header=None,
        names=["datetime", "bid", "ask"],
        dtype={"datetime": str, "bid": float, "ask": float},
    )
    chunk["datetime"] = pd.to_datetime(
        chunk["datetime"], format="%Y.%m.%d %H:%M:%S.%f", errors="coerce"
    )
    chunk = chunk.dropna(subset=["datetime"])
    if chunk.empty:
        return pd.DataFrame(columns=["datetime", "mid"])
    chunk["mid"] = (chunk["bid"] + chunk["ask"]) / 2.0
    return chunk[["datetime", "mid"]]


def _bars_from_ticks(ticks: pd.DataFrame, pandas_tf: str) -> pd.DataFrame:
    """Map ticks to TF buckets; local OHLC + first/last tick times per bar."""
    if ticks.empty:
        return pd.DataFrame(columns=_OHLC_COLS)
    ticks = ticks.sort_values("datetime", kind="mergesort")
    bars = ticks["datetime"].dt.floor(pandas_tf)
    out = (
        ticks.assign(bar=bars)
        .groupby("bar", as_index=False)
        .agg(
            first_time=("datetime", "first"),
            last_time=("datetime", "last"),
            open=("mid", "first"),
            high=("mid", "max"),
            low=("mid", "min"),
            close=("mid", "last"),
        )
    )
    return out


def _byte_ranges(path: Path, workers: int) -> list[tuple[int, int]]:
    size = path.stat().st_size
    if size == 0:
        return []
    workers = max(1, min(workers, size))
    step = size // workers
    cuts = [0]
    with path.open("rb") as f:
        for i in range(1, workers):
            f.seek(step * i)
            f.readline()  # align to next full line
            pos = f.tell()
            if pos < size and pos > cuts[-1]:
                cuts.append(pos)
    cuts.append(size)
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


def _process_range(path_str: str, start: int, end: int, pandas_tf: str) -> pd.DataFrame:
    """
    Read [start, end) byte range and aggregate to TF OHLC bars.

    Shards may share a bar on the boundary; merge later by first/last tick times.
    """
    path = Path(path_str)
    parts: list[pd.DataFrame] = []
    buf = ""

    with path.open("rb") as f:
        f.seek(start)
        remaining = end - start
        while remaining > 0:
            n = min(_READ_CHUNK, remaining)
            chunk = f.read(n)
            if not chunk:
                break
            remaining -= len(chunk)
            buf += chunk.decode("utf-8", errors="replace")

            if remaining > 0:
                last_nl = buf.rfind("\n")
                if last_nl < 0:
                    continue
                text, buf = buf[: last_nl + 1], buf[last_nl + 1 :]
            else:
                text, buf = buf, ""

            ticks = _parse_tick_text(text)
            part = _bars_from_ticks(ticks, pandas_tf)
            if not part.empty:
                parts.append(part)

    if not parts:
        return pd.DataFrame(columns=_OHLC_COLS)
    return _merge_bar_parts(parts)


def _merge_bar_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge shard OHLC for the same bar:
    open from earliest first_time, close from latest last_time,
    high=max, low=min.
    """
    if not parts:
        return pd.DataFrame(columns=_OHLC_COLS)
    merged = pd.concat(parts, ignore_index=True)
    if merged.empty:
        return pd.DataFrame(columns=_OHLC_COLS)

    rows: list[dict] = []
    for bar, g in merged.groupby("bar", sort=True):
        first_i = g["first_time"].idxmin()
        last_i = g["last_time"].idxmax()
        rows.append(
            {
                "bar": bar,
                "first_time": g.loc[first_i, "first_time"],
                "last_time": g.loc[last_i, "last_time"],
                "open": float(g.loc[first_i, "open"]),
                "high": float(g["high"].max()),
                "low": float(g["low"].min()),
                "close": float(g.loc[last_i, "close"]),
            }
        )
    return pd.DataFrame(rows, columns=_OHLC_COLS)


def prepare_ohlc(
    timeframe: str,
    symbol: str | None = None,
    ticks_file: Path | None = None,
    ohlc_path: Path | None = None,
    workers: int | str = DEFAULT_WORKERS,
    decimals: str | int = "ticks",
) -> pd.DataFrame:
    """Aggregate tick mid prices into TF OHLC CSV (multiprocess)."""
    if symbol is None and ticks_file is None:
        raise ValueError("symbol is required when ticks_file is not provided")
    if timeframe not in TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}. "
            f"Choose from: {', '.join(TIMEFRAMES)}"
        )
    pandas_tf = TIMEFRAMES[timeframe]
    if ticks_file is None:
        ticks_file = ticks_path(symbol)
    if ohlc_path is None:
        if symbol is None:
            raise ValueError("symbol is required when ohlc_path is not provided")
        ohlc_path = resolve_ohlc_path(timeframe=timeframe, symbol=symbol)

    if not ticks_file.exists():
        raise FileNotFoundError(f"Ticks file not found: {ticks_file}")

    if isinstance(decimals, str) and decimals.lower() == "ticks":
        decimals_n = detect_price_decimals(ticks_file)
        decimals_src = "ticks"
    else:
        decimals_n = int(decimals)
        if decimals_n < 0:
            raise ValueError("--decimals must be >= 0 or 'ticks'")
        decimals_src = "fixed"

    ohlc_path.parent.mkdir(parents=True, exist_ok=True)
    workers_n = resolve_workers(workers)
    ranges = _byte_ranges(ticks_file, workers_n)
    print(
        f"Preparing {symbol or ticks_file.name} {timeframe} OHLC with {len(ranges)} workers "
        f"({ticks_file.stat().st_size / (1024**3):.2f} GiB), "
        f"decimals={decimals_n} ({decimals_src})..."
    )

    parts: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=len(ranges)) as pool:
        futures = {
            pool.submit(_process_range, str(ticks_file), start, end, pandas_tf): (
                start,
                end,
            )
            for start, end in ranges
        }
        done = 0
        for fut in as_completed(futures):
            part = fut.result()
            if not part.empty:
                parts.append(part)
            done += 1
            start, end = futures[fut]
            print(
                f"  shard {done}/{len(ranges)} done "
                f"[{start:,}:{end:,}) -> {len(part):,} local bars"
            )

    if not parts:
        raise RuntimeError("No bars produced from ticks.")

    merged = _merge_bar_parts(parts).sort_values("bar")
    ohlc = pd.DataFrame(
        {
            "datetime": merged["bar"],
            "open": merged["open"].round(decimals_n),
            "high": merged["high"].round(decimals_n),
            "low": merged["low"].round(decimals_n),
            "close": merged["close"].round(decimals_n),
        }
    )

    ohlc.to_csv(
        ohlc_path,
        index=False,
        header=False,
        float_format=f"%.{decimals_n}f",
    )
    print(f"Saved {len(ohlc):,} bars -> {ohlc_path}")
    return ohlc.reset_index(drop=True)


def load_ohlc(
    path: Path | None = None,
    timeframe: str | None = None,
    symbol: str | None = None,
) -> pd.DataFrame:
    if path is None:
        if symbol is None or timeframe is None:
            raise ValueError(
                "symbol and timeframe are required when path is not provided"
            )
        path = resolve_ohlc_path(timeframe=timeframe, symbol=symbol)
    if not path.exists():
        sym = symbol or "?"
        tf = timeframe or "?"
        raise FileNotFoundError(
            f"OHLC data not found: {path}. "
            f"Run with --prepare --symbol {sym} --timeframe {tf} first."
        )
    df = pd.read_csv(
        path,
        header=None,
        names=["datetime", "open", "high", "low", "close"],
        parse_dates=["datetime"],
    )
    return df.sort_values("datetime").reset_index(drop=True)
