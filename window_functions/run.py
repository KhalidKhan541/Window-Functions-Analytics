#!/usr/bin/env python3
"""
CLI entry point for the Window Functions & Advanced Analytics project.

Usage:
    python -m window_functions.run generate --n-rows 3000000 --output output
    python -m window_functions.run analyze  --input output/enriched_data.csv --output output --top-n 25
    python -m window_functions.run validate --input output/enriched_data.csv
    python -m window_functions.run full     --output output --n-rows 3000000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from window_functions.src.pipeline import (
    PipelineConfig,
    PipelineError,
    build_summary,
    generate_synthetic_data,
    get_top_products,
    run_all_window_analyses,
    save_results,
    _validate_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_yaml_config(path: str) -> dict:
    """Load a YAML configuration file.

    Args:
        path: Path to YAML file.

    Returns:
        Parsed configuration as a dictionary.
    """
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed; skipping YAML config")
        return {}

    config_path = Path(path)
    if not config_path.exists():
        logger.warning("Config file not found: %s", path)
        return {}

    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    logger.info("Loaded config from %s", path)
    return data


def _build_config(args: argparse.Namespace, yaml_config: dict) -> PipelineConfig:
    """Merge CLI arguments with YAML config to build a PipelineConfig.

    CLI arguments take precedence over YAML values.

    Args:
        args: Parsed CLI arguments.
        yaml_config: Parsed YAML configuration dictionary.

    Returns:
        Merged PipelineConfig instance.
    """
    merged: dict = {**yaml_config}

    for key in ("n_rows", "output", "seed", "top_n", "config"):
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            merged[key] = cli_val

    output_dir = merged.get("output", "output")

    return PipelineConfig(
        n_rows=int(merged.get("n_rows", 3_000_000)),
        output_dir=str(output_dir),
        seed=int(merged.get("seed", 42)),
        top_n=int(merged.get("top_n", 25)),
        validation_config_path=merged.get("config"),
    )


def cmd_generate(args: argparse.Namespace) -> None:
    """Handle the 'generate' subcommand."""
    t0 = time.perf_counter()
    config = _build_config(args, _load_yaml_config(getattr(args, "config", "")))

    logger.info("Generating %d rows", config.n_rows)
    df = generate_synthetic_data(config.n_rows, config.seed)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "raw_data.csv"
    df.to_csv(path, index=False)

    elapsed = time.perf_counter() - t0
    logger.info("Saved raw data to %s (%.2f MB) in %.3fs", path, path.stat().st_size / 1e6, elapsed)
    print(f"Generated {len(df):,} rows -> {path} ({elapsed:.3f}s)")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Handle the 'analyze' subcommand."""
    t0 = time.perf_counter()
    config = _build_config(args, _load_yaml_config(getattr(args, "config", "")))

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    logger.info("Loading data from %s", input_path)
    df = pd.read_csv(input_path, parse_dates=["transaction_date"])
    load_time = time.perf_counter() - t0
    logger.info("Loaded %d rows in %.3fs", len(df), load_time)

    t1 = time.perf_counter()
    enriched = run_all_window_analyses(df, config)
    analyze_time = time.perf_counter() - t1
    logger.info("Analysis completed in %.3fs", analyze_time)

    t2 = time.perf_counter()
    summary = build_summary(enriched)
    top_products = get_top_products(summary, config.top_n)
    save_results(enriched, summary, top_products, config.output_dir)
    save_time = time.perf_counter() - t2

    total = time.perf_counter() - t0
    print(
        f"Analysis complete.\n"
        f"  Rows enriched:  {len(enriched):,}\n"
        f"  Products:       {len(summary):,}\n"
        f"  Top-N:          {len(top_products)}\n"
        f"  Load:           {load_time:.3f}s\n"
        f"  Analyze:        {analyze_time:.3f}s\n"
        f"  Save:           {save_time:.3f}s\n"
        f"  Total:          {total:.3f}s"
    )


def cmd_validate(args: argparse.Namespace) -> None:
    """Handle the 'validate' subcommand."""
    t0 = time.perf_counter()
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    logger.info("Loading data from %s", input_path)
    df = pd.read_csv(input_path, parse_dates=["transaction_date"])
    load_time = time.perf_counter() - t0

    config_path = getattr(args, "config", None)
    t1 = time.perf_counter()
    try:
        _validate_dataset(df, config_path)
        validate_time = time.perf_counter() - t1
        print(
            f"Validation PASSED.\n"
            f"  Rows:    {len(df):,}\n"
            f"  Columns: {len(df.columns)}\n"
            f"  Load:    {load_time:.3f}s\n"
            f"  Check:   {validate_time:.3f}s"
        )
    except ValueError as exc:
        validate_time = time.perf_counter() - t1
        print(
            f"Validation FAILED: {exc}\n"
            f"  Load:  {load_time:.3f}s\n"
            f"  Check: {validate_time:.3f}s"
        )
        sys.exit(1)


def cmd_full(args: argparse.Namespace) -> None:
    """Handle the 'full' subcommand: generate -> validate -> analyze."""
    from window_functions.src.pipeline import run_pipeline

    t0 = time.perf_counter()
    config = _build_config(args, _load_yaml_config(getattr(args, "config", "")))

    try:
        results = run_pipeline(config)
    except PipelineError as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)

    total = time.perf_counter() - t0
    print("Pipeline complete.\n")
    print(f"  Timing breakdown:")
    for stage, secs in results.timings.items():
        print(f"    {stage:<12} {secs:.3f}s")
    print(f"    {'wall-clock':<12} {total:.3f}s\n")

    if results.enriched_df is not None:
        print(f"  Enriched rows:   {len(results.enriched_df):,}")
    if results.summary_df is not None:
        print(f"  Summary rows:    {len(results.summary_df):,}")
    if results.top_products_df is not None:
        print(f"  Top products:    {len(results.top_products_df)}")
    print(f"\n  Output dir: {config.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="window_analytics",
        description="Window Functions & Advanced Analytics CLI",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML configuration file",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # generate
    gen = sub.add_parser("generate", help="Generate synthetic dataset")
    gen.add_argument("--n-rows", type=int, default=3_000_000, help="Number of rows (default: 3M)")
    gen.add_argument("--output", type=str, default="output", help="Output directory")
    gen.add_argument("--seed", type=int, default=42, help="Random seed")

    # analyze
    ana = sub.add_parser("analyze", help="Run window function analysis on existing data")
    ana.add_argument("--input", type=str, required=True, help="Path to input CSV")
    ana.add_argument("--output", type=str, default="output", help="Output directory")
    ana.add_argument("--top-n", type=int, default=25, help="Number of top products to report")

    # validate
    val = sub.add_parser("validate", help="Validate a dataset against expected schema")
    val.add_argument("--input", type=str, required=True, help="Path to input CSV")

    # full
    ful = sub.add_parser("full", help="Run full pipeline: generate -> validate -> analyze")
    ful.add_argument("--output", type=str, default="output", help="Output directory")
    ful.add_argument("--n-rows", type=int, default=3_000_000, help="Number of rows (default: 3M)")

    return parser


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point.

    Args:
        argv: Command-line arguments. Defaults to sys.argv.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "generate": cmd_generate,
        "analyze": cmd_analyze,
        "validate": cmd_validate,
        "full": cmd_full,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
