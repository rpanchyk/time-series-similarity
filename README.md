# Time Series Similarity

Research CLI for Forex OHLC time-series similarity: build bars from ticks, find historical windows similar to a query, report forward (horizon) outcomes as a table and chart.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run from the repository root (`python main.py ...`).

## Data layout

| Path | Format |
|------|--------|
| `data/ticks/<SYMBOL>_ticks.csv` | `datetime,bid,ask` — datetime `%Y.%m.%d %H:%M:%S.%f` |
| `data/ohlc/<SYMBOL>_<TF>.csv` | `datetime,open,high,low,close` (no header) |
| `output/` | CSV + PNG results |

## Commands

```bash
# Build OHLC from ticks (multiprocess)
python main.py --prepare --symbol EURUSD --timeframe 1h --workers max

# Find similar windows
python main.py --similarity --symbol EURUSD --timeframe 1h \
  --search-type dtw --start-date "2025-06-16 08:00" \
  --window 20 --horizon 20 --top-k 10

# Several metrics in one run (comma-separated; parallel via --workers)
python main.py --similarity --symbol EURUSD --timeframe 1h \
  --search-type dtw,pearson,z-euclidean --workers max \
  --start-date "2025-06-16 08:00"
```

Useful flags: `--price`, `--stride`, `--plots`, `--skip-weekends`, `--max-price-gap` (absolute price units, not pips), `--start-index`, `--workers` (`max` or N; for similarity, one process per search-type), `--output-image png|skip`, `--output-table file|console|skip`, `--output-table-format csv|json`.

## Search types

| `--search-type` | Notes |
|-----------------|-------|
| `z-euclidean` | z-normalized Euclidean on `--price` |
| `z-manhattan` | z-normalized L1 on `--price` |
| `pearson` / `spearman` / `cosine` | correlation / cosine distances |
| `cid` | Complexity-Invariant Distance (z-normalized) |
| `dtw` / `softdtw` | alignment distances (z-normalized) |
| `logret` | z-Euclidean on log-returns of `--price` |
| `ohlc-joint` | Euclidean on `concat(z(close), z(range))` — no second z-norm |
| `candle-shape` | Euclidean on body/wick ratios — no second z-norm |
| `dtw-multiv` | multivariate DTW on z-normalized OHLC |

Struct metrics (`ohlc-joint`, `candle-shape`, `dtw-multiv`) ignore `--price`. Charts still overlay z-normalized close for visualization.

`--search-type` accepts one value or a comma-separated list (`dtw,pearson`); each type writes its own CSV/PNG. With multiple types, `--workers` runs them in parallel processes (capped by the number of types).

## Tests

```bash
pytest
# or: python tests/test.py
```

## Spec

See `TASK.md` for the original research brief (Ukrainian).
