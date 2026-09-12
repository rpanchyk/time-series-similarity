from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
TICKS_DIR = ROOT / "data" / "ticks"
OHLC_DIR = ROOT / "data" / "ohlc"
OUTPUT_DIR = ROOT / "output"

DEFAULT_WINDOW = 20
DEFAULT_STRIDE = 1
DEFAULT_TOP_K = 10
DEFAULT_HORIZON = 20
DEFAULT_PLOTS = 5
DEFAULT_PRICE = "close"
DEFAULT_DECIMALS = "ticks"
DEFAULT_WORKERS = "max"
DEFAULT_MAX_PRICE_GAP = 0.0
DEFAULT_OUTPUT_IMAGE = "png"
DEFAULT_OUTPUT_TABLE = "file"
DEFAULT_OUTPUT_TABLE_FORMAT = "csv"
OUTPUT_IMAGE_MODES = ("png", "skip")
OUTPUT_TABLE_MODES = ("file", "console", "skip")
OUTPUT_TABLE_FORMATS = ("csv", "json")


def resolve_workers(workers: str | int = DEFAULT_WORKERS) -> int:
    """Resolve CLI workers value: 'max' -> CPU count, or positive int."""
    if isinstance(workers, str) and workers.strip().lower() == "max":
        return max(1, os.cpu_count() or 1)
    n = int(workers)
    if n < 1:
        raise ValueError("--workers must be >= 1 or 'max'")
    return n

# CLI label -> pandas offset alias
TIMEFRAMES = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "D": "D",
}

PRICE_COLUMNS = ("open", "high", "low", "close")


def ticks_path(symbol: str) -> Path:
    return TICKS_DIR / f"{symbol}_ticks.csv"


def ohlc_path(*, timeframe: str, symbol: str) -> Path:
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return OHLC_DIR / f"{symbol}_{timeframe}.csv"


# Extensible registry: CLI --search-type -> short description
SEARCH_TYPES = {
    "z-euclidean": "z-normalized Euclidean distance",
    "dtw": "z-normalized Dynamic Time Warping distance",
    "softdtw": "z-normalized Soft-DTW distance",
    "pearson": "Pearson distance (1 - correlation)",
    "spearman": "Spearman rank distance (1 - rank correlation)",
    "cosine": "Cosine distance (1 - cosine similarity)",
    "z-manhattan": "z-normalized Manhattan (L1) distance",
    "cid": "Complexity-Invariant Distance (z-normalized)",
    "logret": "z-Euclidean on log-returns of --price",
    "ohlc-joint": "Euclidean on concat(z(close), z(high-low))",
    "candle-shape": "Euclidean on candle body/wick shape features",
    "dtw-multiv": "multivariate DTW on z-normalized OHLC channels",
}


def parse_search_types(raw: str) -> list[str]:
    """Parse comma-separated --search-type values; order preserved, duplicates dropped."""
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        raise ValueError(
            "--search-type must list at least one type "
            f"(one of: {', '.join(sorted(SEARCH_TYPES))})"
        )
    unknown = sorted({p for p in parts if p not in SEARCH_TYPES})
    if unknown:
        raise ValueError(
            f"Unknown search type(s): {', '.join(unknown)}. "
            f"Choose from: {', '.join(sorted(SEARCH_TYPES))}"
        )
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out
