"""Run all search types and estimate long / skip / short from max_dev_*."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.config import DEFAULT_WORKERS, OUTPUT_DIR, SEARCH_TYPES, resolve_workers
from src.prepare import load_ohlc
from src.similarity import find_similar_many, resolve_start_index

# Classification: upside vs downside dominance on the forward horizon.
DOMINANCE = 1.25  # need 25% larger excursion to call direction
MIN_MOVE = 0.05  # % — below this, treat as noise -> skip


def classify_row(high: float, low: float) -> str:
    up = float(high)
    down = abs(float(low))
    if up >= MIN_MOVE and up > down * DOMINANCE:
        return "long"
    if down >= MIN_MOVE and down > up * DOMINANCE:
        return "short"
    return "skip"


def weights_from_distance(distances: np.ndarray) -> np.ndarray:
    d = np.asarray(distances, dtype=float)
    # Closer matches weigh more; shift so min distance still has mass.
    inv = 1.0 / (d - d.min() + 1e-6)
    return inv / inv.sum()


def summarize(labels: list[str], distances: np.ndarray | None = None) -> dict:
    labels = list(labels)
    n = len(labels)
    counts = {k: labels.count(k) for k in ("long", "skip", "short")}
    equal = {k: counts[k] / n for k in counts} if n else {k: 0.0 for k in counts}
    if distances is not None and n:
        w = weights_from_distance(distances)
        w_counts = {k: 0.0 for k in ("long", "skip", "short")}
        for lab, wi in zip(labels, w):
            w_counts[lab] += float(wi)
        weighted = w_counts
    else:
        weighted = equal
    return {
        "n": n,
        "counts": counts,
        "equal": equal,
        "weighted": weighted,
        "mode": max(equal, key=equal.get) if n else "skip",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--start-date", type=str, default="2025-06-16 08:00")
    ap.add_argument(
        "--workers",
        type=str,
        default=DEFAULT_WORKERS,
        help="Parallel processes for search types: 'max' or positive int "
        f"(default: {DEFAULT_WORKERS})",
    )
    ap.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated search types to skip "
        "(e.g. softdtw,dtw-multiv)",
    )
    args = ap.parse_args()

    try:
        workers = resolve_workers(args.workers)
    except (TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    symbol = "EURUSD"
    timeframe = "1h"
    window = args.window
    horizon = args.horizon
    top_k = args.top_k
    start_date = args.start_date
    price = "close"
    excluded = {p.strip() for p in args.exclude.split(",") if p.strip()}
    unknown = sorted(excluded - set(SEARCH_TYPES))
    if unknown:
        raise SystemExit(f"Unknown --exclude type(s): {', '.join(unknown)}")
    search_types = sorted(st for st in SEARCH_TYPES if st not in excluded)
    if not search_types:
        raise SystemExit("No search types left after --exclude")
    workers_n = min(workers, len(search_types))

    print(f"Loading {symbol} {timeframe}...")
    ohlc = load_ohlc(timeframe=timeframe, symbol=symbol)
    q_start, quantized, actual = resolve_start_index(
        ohlc, start_date, timeframe, window
    )
    print(f"query: {start_date} -> {actual} (index={q_start})")
    print(
        f"rule: long if high > {DOMINANCE}*|low| and high>={MIN_MOVE}%; "
        f"short symmetric; else skip"
    )
    print(
        f"Running {len(search_types)} search types "
        f"(workers={workers_n})...",
        flush=True,
    )

    t0 = time.perf_counter()
    typed_results = find_similar_many(
        ohlc,
        search_types,
        workers=workers_n,
        window=window,
        stride=1,
        top_k=top_k,
        query_start=q_start,
        horizon=horizon,
        price=price,
    )
    total_elapsed = time.perf_counter() - t0

    rows_out: list[dict] = []
    per_type: dict[str, dict] = {}

    for i, (st, results) in enumerate(typed_results, start=1):
        labels = [
            classify_row(r.max_dev_high, r.max_dev_low)
            for r in results.itertuples(index=False)
        ]
        dist = results["distance"].to_numpy(float)
        summary = summarize(labels, dist)
        per_type[st] = summary
        print(
            f"[{i}/{len(search_types)}] {st}  "
            f"long={summary['equal']['long']:.0%} "
            f"skip={summary['equal']['skip']:.0%} "
            f"short={summary['equal']['short']:.0%}  mode={summary['mode']}",
            flush=True,
        )
        for r, lab in zip(results.itertuples(index=False), labels):
            rows_out.append(
                {
                    "search_type": st,
                    "rank": int(r.rank),
                    "distance": float(r.distance),
                    "match_start_time": str(r.match_start_time),
                    "match_end_time": str(r.match_end_time),
                    "max_dev_high": float(r.max_dev_high),
                    "max_dev_low": float(r.max_dev_low),
                    "label": lab,
                }
            )

    # Ensemble: equal vote across all matches of all types
    all_labels = [r["label"] for r in rows_out]
    # Distance scales differ by metric — use equal weights for cross-type ensemble
    ensemble_equal = summarize(all_labels, None)
    # Per-type mode vote
    modes = [per_type[st]["mode"] for st in per_type]
    mode_vote = summarize(modes, None)

    # Overlap: match windows appearing in >=2 search types
    key_counts: dict[str, int] = {}
    key_labels: dict[str, list[str]] = {}
    for r in rows_out:
        key = r["match_start_time"]
        key_counts[key] = key_counts.get(key, 0) + 1
        key_labels.setdefault(key, []).append(r["label"])
    multi = {
        k: summarize(labs, None)
        for k, labs in key_labels.items()
        if key_counts[k] >= 2
    }
    consensus_labels = [s["mode"] for s in multi.values()]
    overlap_summary = summarize(consensus_labels, None)

    report = {
        "meta": {
            "symbol": symbol,
            "timeframe": timeframe,
            "window": window,
            "horizon": horizon,
            "top_k": top_k,
            "start_date": start_date,
            "query_bar": str(actual),
            "price": price,
            "workers": workers_n,
            "excluded": sorted(excluded),
            "search_types": search_types,
            "dominance": DOMINANCE,
            "min_move_pct": MIN_MOVE,
            "elapsed_s": round(total_elapsed, 2),
            "rule": (
                f"long if max_dev_high > {DOMINANCE}*|max_dev_low| "
                f"and max_dev_high>={MIN_MOVE}; short symmetric; else skip"
            ),
        },
        "per_type": {
            st: {
                "counts": per_type[st]["counts"],
                "equal_prob": {k: round(v, 4) for k, v in per_type[st]["equal"].items()},
                "weighted_prob": {
                    k: round(v, 4) for k, v in per_type[st]["weighted"].items()
                },
                "mode": per_type[st]["mode"],
            }
            for st in per_type
        },
        "ensemble": {
            "all_matches_equal": {
                "n": ensemble_equal["n"],
                "counts": ensemble_equal["counts"],
                "prob": {k: round(v, 4) for k, v in ensemble_equal["equal"].items()},
                "mode": ensemble_equal["mode"],
            },
            "per_type_mode_vote": {
                "n": mode_vote["n"],
                "counts": mode_vote["counts"],
                "prob": {k: round(v, 4) for k, v in mode_vote["equal"].items()},
                "mode": mode_vote["mode"],
            },
            "overlap_ge2_types": {
                "n_unique_windows": overlap_summary["n"],
                "counts": overlap_summary["counts"],
                "prob": {k: round(v, 4) for k, v in overlap_summary["equal"].items()},
                "mode": overlap_summary["mode"],
            },
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp(actual).strftime("%Y%m%d-%H%M")
    excl_tag = (
        "_excl-" + "-".join(sorted(excluded)) if excluded else ""
    )
    stem = f"{symbol}_{timeframe}_{ts}_window{window}{excl_tag}_direction"
    matches_path = OUTPUT_DIR / f"{stem}_matches.csv"
    report_path = OUTPUT_DIR / f"{stem}_report.json"
    pd.DataFrame(rows_out).to_csv(matches_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nTotal elapsed: {total_elapsed:.1f}s")
    print("\n=== Ensemble (all matches, equal) ===")
    print(report["ensemble"]["all_matches_equal"])
    print("=== Per-type mode vote ===")
    print(report["ensemble"]["per_type_mode_vote"])
    print("=== Overlap (>=2 types) ===")
    print(report["ensemble"]["overlap_ge2_types"])
    print(f"\nWrote {matches_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
