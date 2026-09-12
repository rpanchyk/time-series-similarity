"""Time Series Similarity — entry point."""

from __future__ import annotations

import argparse

from src.config import (
    DEFAULT_DECIMALS,
    DEFAULT_HORIZON,
    DEFAULT_MAX_PRICE_GAP,
    DEFAULT_OUTPUT_IMAGE,
    DEFAULT_OUTPUT_TABLE,
    DEFAULT_OUTPUT_TABLE_FORMAT,
    DEFAULT_PLOTS,
    DEFAULT_PRICE,
    DEFAULT_STRIDE,
    DEFAULT_TOP_K,
    DEFAULT_WINDOW,
    DEFAULT_WORKERS,
    OUTPUT_IMAGE_MODES,
    OUTPUT_TABLE_FORMATS,
    OUTPUT_TABLE_MODES,
    PRICE_COLUMNS,
    SEARCH_TYPES,
    TIMEFRAMES,
    ticks_path,
    ohlc_path,
    parse_search_types,
    resolve_workers,
)
from src.output import plot_matches, print_results_table, save_results_table
from src.prepare import load_ohlc, prepare_ohlc
from src.similarity import find_similar_many, resolve_start_index


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forex time-series similarity")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prepare",
        action="store_true",
        help="Build data/ohlc/<SYMBOL>_<timeframe>.csv from ticks",
    )
    mode.add_argument(
        "--similarity",
        action="store_true",
        help="Find similar windows in OHLC data",
    )
    p.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Instrument symbol (required)",
    )
    p.add_argument(
        "--search-type",
        type=str,
        default=None,
        help="Search type(s) for --similarity: one value or comma-separated list. "
        f"One of: {', '.join(sorted(SEARCH_TYPES))}. "
        "Example: dtw or dtw,pearson,z-euclidean",
    )
    p.add_argument(
        "--timeframe",
        choices=list(TIMEFRAMES),
        required=True,
        help=f"Bar timeframe for prepare/similarity. One of: {', '.join(TIMEFRAMES)}",
    )
    p.add_argument(
        "--price",
        choices=list(PRICE_COLUMNS),
        default=DEFAULT_PRICE,
        help=f"OHLC column used for similarity matching (default: {DEFAULT_PRICE})",
    )
    p.add_argument(
        "--workers",
        type=str,
        default=DEFAULT_WORKERS,
        help="Parallel workers: --prepare tick shards; --similarity one process "
        f"per search-type. 'max' (all CPU cores) or positive integer "
        f"(default: {DEFAULT_WORKERS})",
    )
    p.add_argument(
        "--decimals",
        type=str,
        default=DEFAULT_DECIMALS,
        help="Price decimal places for --prepare: 'ticks' (match tick file) "
        f"or non-negative integer N (default: {DEFAULT_DECIMALS})",
    )
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"Bars after each match to report/plot (default: {DEFAULT_HORIZON})",
    )
    p.add_argument(
        "--plots",
        type=int,
        default=DEFAULT_PLOTS,
        help=f"Max matches to draw on the chart (default: {DEFAULT_PLOTS})",
    )
    p.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Query window start datetime; quantized to --timeframe "
        "(default: last window). Example: 2024-01-15 10:00",
    )
    p.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Start index of query window (alternative to --start-date)",
    )
    p.add_argument(
        "--skip-weekends",
        action="store_true",
        default=False,
        help="With --similarity: exclude windows that cross Fri→Mon weekend gaps "
        "(default: false)",
    )
    p.add_argument(
        "--max-price-gap",
        type=float,
        default=DEFAULT_MAX_PRICE_GAP,
        help="With --similarity: reject windows where |open[i+1]-close[i]| "
        f"exceeds this absolute price gap (not pips; default: {DEFAULT_MAX_PRICE_GAP} = off)",
    )
    p.add_argument(
        "--output-image",
        choices=list(OUTPUT_IMAGE_MODES),
        default=DEFAULT_OUTPUT_IMAGE,
        help="With --similarity: chart output — png (write PNG) or skip "
        f"(default: {DEFAULT_OUTPUT_IMAGE})",
    )
    p.add_argument(
        "--output-table",
        choices=list(OUTPUT_TABLE_MODES),
        default=DEFAULT_OUTPUT_TABLE,
        help="With --similarity: table output — file (write disk), console "
        f"(print only), or skip (default: {DEFAULT_OUTPUT_TABLE})",
    )
    p.add_argument(
        "--output-table-format",
        choices=list(OUTPUT_TABLE_FORMATS),
        default=DEFAULT_OUTPUT_TABLE_FORMAT,
        help="With --similarity: table format for --output-table file|console "
        f"(default: {DEFAULT_OUTPUT_TABLE_FORMAT})",
    )

    args = p.parse_args(argv)

    if args.similarity and not args.search_type:
        p.error("--similarity requires --search-type")
    if args.prepare:
        # Similarity-only flags rejected under --prepare.
        prepare_forbidden = (
            ("search_type", None, "--search-type"),
            ("start_date", None, "--start-date"),
            ("start_index", None, "--start-index"),
            ("skip_weekends", False, "--skip-weekends"),
            ("max_price_gap", DEFAULT_MAX_PRICE_GAP, "--max-price-gap"),
            ("output_image", DEFAULT_OUTPUT_IMAGE, "--output-image"),
            ("output_table", DEFAULT_OUTPUT_TABLE, "--output-table"),
            (
                "output_table_format",
                DEFAULT_OUTPUT_TABLE_FORMAT,
                "--output-table-format",
            ),
        )
        for attr, default, flag in prepare_forbidden:
            if getattr(args, attr) != default:
                p.error(f"{flag} is only valid with --similarity")
    if args.similarity and args.start_date is not None and args.start_index is not None:
        p.error("Use either --start-date or --start-index, not both")
    if args.similarity and args.plots < 1:
        p.error("--plots must be >= 1")
    if args.max_price_gap < 0:
        p.error("--max-price-gap must be >= 0")
    if (
        args.similarity
        and args.output_table == "skip"
        and args.output_table_format != DEFAULT_OUTPUT_TABLE_FORMAT
    ):
        p.error("--output-table-format is only valid when --output-table is not skip")

    if args.search_type is not None:
        try:
            args.search_type = parse_search_types(args.search_type)
        except ValueError as exc:
            p.error(str(exc))

    # Validate --decimals: 'ticks' or non-negative int (used by --prepare)
    dec = str(args.decimals).strip().lower()
    if dec == "ticks":
        args.decimals = "ticks"
    else:
        if not dec.isdigit():
            p.error("--decimals must be 'ticks' or a non-negative integer")
        args.decimals = int(dec)

    # Validate --workers: 'max' or positive int
    try:
        args.workers = resolve_workers(args.workers)
    except (TypeError, ValueError) as exc:
        p.error(str(exc) if str(exc) else "--workers must be 'max' or a positive integer")

    args.symbol = args.symbol.upper()

    return args


def cmd_prepare(args: argparse.Namespace) -> None:
    ticks_file = ticks_path(args.symbol)
    if not ticks_file.exists():
        raise SystemExit(
            f"Error: ticks file not found: {ticks_file}\n"
            f"Expected file: data/ticks/{args.symbol}_ticks.csv"
        )
    prepare_ohlc(
        symbol=args.symbol,
        timeframe=args.timeframe,
        workers=args.workers,
        decimals=args.decimals,
    )


def cmd_similarity(args: argparse.Namespace) -> None:
    ohlc_file = ohlc_path(timeframe=args.timeframe, symbol=args.symbol)
    if not ohlc_file.exists():
        raise SystemExit(
            f"Error: OHLC data not found: {ohlc_file}\n"
            f"Run prepare first, e.g.: "
            f"python main.py --prepare --symbol {args.symbol} --timeframe {args.timeframe}"
        )

    print(f"Loading OHLC data (symbol={args.symbol}, timeframe={args.timeframe})...")
    ohlc = load_ohlc(timeframe=args.timeframe, symbol=args.symbol)
    print(
        f"Loaded {len(ohlc):,} bars "
        f"({ohlc['datetime'].iloc[0]} -> {ohlc['datetime'].iloc[-1]})"
    )

    query_start = args.start_index
    if args.start_date is not None:
        query_start, quantized, actual = resolve_start_index(
            ohlc,
            start_date=args.start_date,
            timeframe=args.timeframe,
            window=args.window,
        )
        print(
            f"start-date={args.start_date} -> quantized={quantized} "
            f"-> bar={actual} (index={query_start})"
        )

    search_types: list[str] = args.search_type
    workers_n = min(args.workers, len(search_types))
    print(
        f"Similarity search-types={','.join(search_types)} "
        f"symbol={args.symbol} "
        f"timeframe={args.timeframe} price={args.price} "
        f"window={args.window}, stride={args.stride}, "
        f"top_k={args.top_k}, horizon={args.horizon}, "
        f"skip_weekends={args.skip_weekends}, "
        f"max_price_gap={args.max_price_gap}, "
        f"workers={workers_n}..."
    )
    if workers_n > 1 and len(search_types) > 1:
        print(
            f"Running {len(search_types)} search types "
            f"in parallel ({workers_n} processes)..."
        )

    typed_results = find_similar_many(
        ohlc,
        search_types,
        workers=args.workers,
        window=args.window,
        stride=args.stride,
        top_k=args.top_k,
        query_start=query_start,
        horizon=args.horizon,
        price=args.price,
        skip_weekends=args.skip_weekends,
        max_price_gap=args.max_price_gap,
    )

    for i, (search_type, results) in enumerate(typed_results, start=1):
        search_type_desc = SEARCH_TYPES[search_type]
        print(
            f"\n[{i}/{len(search_types)}] search-type={search_type}: "
            f"{search_type_desc}"
        )
        if args.output_table == "console":
            print("\nResults:")
            print_results_table(
                results,
                symbol=args.symbol,
                table_format=args.output_table_format,
            )
        elif args.output_table == "file":
            save_results_table(
                results,
                timeframe=args.timeframe,
                search_type=search_type,
                price=args.price,
                symbol=args.symbol,
                table_format=args.output_table_format,
            )
        if args.output_image == "png":
            plot_matches(
                ohlc,
                results,
                window=args.window,
                timeframe=args.timeframe,
                search_type=search_type,
                price=args.price,
                symbol=args.symbol,
                horizon=args.horizon,
                plots=args.plots,
            )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.prepare:
        cmd_prepare(args)
    elif args.similarity:
        cmd_similarity(args)


if __name__ == "__main__":
    main()
