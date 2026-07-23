"""
End-to-end pipeline for window function analytics.

Orchestrates data generation, validation, window function analyses, and result persistence.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the analytics pipeline."""

    n_rows: int = 3_000_000
    output_dir: str = "output"
    seed: int = 42
    top_n: int = 25
    validation_config_path: str | None = None

    def __post_init__(self) -> None:
        self.output_dir = str(Path(self.output_dir).resolve())


@dataclass
class PipelineResults:
    """Container for all pipeline outputs."""

    enriched_df: pd.DataFrame | None = None
    summary_df: pd.DataFrame | None = None
    top_products_df: pd.DataFrame | None = None
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enriched_shape": self.enriched_df.shape if self.enriched_df is not None else None,
            "summary_shape": self.summary_df.shape if self.summary_df is not None else None,
            "top_products_shape": self.top_products_df.shape if self.top_products_df is not None else None,
            "timings": self.timings,
        }


class PipelineError(Exception):
    """Raised when a pipeline stage fails."""


def generate_synthetic_data(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic e-commerce transaction dataset.

    Args:
        n_rows: Number of rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: transaction_id, product_id, customer_id, category,
        quantity, unit_price, discount_pct, transaction_date, region.
    """
    logger.info("Generating %d rows of synthetic data (seed=%d)", n_rows, seed)
    rng = np.random.default_rng(seed)

    n_products = max(500, n_rows // 500)
    n_customers = max(10_000, n_rows // 100)

    categories = [
        "Electronics", "Clothing", "Home & Garden", "Sports", "Books",
        "Automotive", "Health", "Toys", "Grocery", "Office",
    ]
    regions = ["North", "South", "East", "West", "Central"]

    df = pd.DataFrame({
        "transaction_id": np.arange(1, n_rows + 1),
        "product_id": rng.integers(1, n_products + 1, size=n_rows),
        "customer_id": rng.integers(1, n_customers + 1, size=n_rows),
        "category": rng.choice(categories, size=n_rows),
        "quantity": rng.integers(1, 25, size=n_rows),
        "unit_price": np.round(rng.uniform(1.0, 500.0, size=n_rows), 2),
        "discount_pct": np.round(rng.uniform(0.0, 0.40, size=n_rows), 4),
        "transaction_date": pd.date_range(
            start="2024-01-01", periods=n_rows, freq="28s"
        ),
        "region": rng.choice(regions, size=n_rows),
    })

    df["revenue"] = np.round(df["quantity"] * df["unit_price"] * (1 - df["discount_pct"]), 2)
    df["transaction_date"] = df["transaction_date"].astype("datetime64[ns]")

    logger.info("Generated dataset with %d rows, %d columns", *df.shape)
    return df


def _rankings_window(df: pd.DataFrame) -> pd.DataFrame:
    """Rank products within categories by total revenue using ROW_NUMBER, RANK, DENSE_RANK."""
    logger.info("Computing rankings window functions")
    prod_rev = (
        df.groupby(["category", "product_id"])["revenue"]
        .sum()
        .reset_index()
        .rename(columns={"revenue": "total_product_revenue"})
    )

    for col in ("row_num", "rank", "dense_rank"):
        prod_rev[col] = 0

    prod_rev["row_num"] = prod_rev.groupby("category")["total_product_revenue"].rank(
        method="first", ascending=False
    ).astype(int)
    prod_rev["rank"] = prod_rev.groupby("category")["total_product_revenue"].rank(
        method="average", ascending=False
    ).astype(int)
    prod_rev["dense_rank"] = prod_rev.groupby("category")["total_product_revenue"].rank(
        method="dense", ascending=False
    ).astype(int)

    df = df.merge(
        prod_rev[["category", "product_id", "row_num", "rank", "dense_rank"]],
        on=["category", "product_id"],
        how="left",
    )
    df["row_num"] = df["row_num"].fillna(0).astype(int)
    df["rank"] = df["rank"].fillna(0).astype(int)
    df["dense_rank"] = df["dense_rank"].fillna(0).astype(int)
    return df


def _lag_lead_window(df: pd.DataFrame) -> pd.DataFrame:
    """Compute LAG/LEAD on revenue partitioned by product_id, ordered by transaction_date."""
    logger.info("Computing LAG/LEAD window functions")
    df = df.sort_values(["product_id", "transaction_date"]).reset_index(drop=True)

    df["prev_revenue"] = df.groupby("product_id")["revenue"].shift(1)
    df["next_revenue"] = df.groupby("product_id")["revenue"].shift(-1)
    df["revenue_change"] = df["revenue"] - df["prev_revenue"]
    df["pct_revenue_change"] = np.where(
        df["prev_revenue"].abs() > 0,
        np.round(df["revenue_change"] / df["prev_revenue"], 4),
        np.nan,
    )

    df["prev_revenue"] = df["prev_revenue"].round(2)
    df["revenue_change"] = df["revenue_change"].round(2)
    return df


def _cumulative_window(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative sum, min, max of revenue per product."""
    logger.info("Computing cumulative window functions")
    df["cumulative_revenue"] = df.groupby("product_id")["revenue"].cumsum().round(2)
    df["running_avg_price"] = (
        df.groupby("product_id")["unit_price"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
        .round(2)
    )
    return df


def _moving_average_window(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Compute rolling average of revenue per product."""
    logger.info("Computing rolling average (window=%d) window function", window)
    df["rolling_avg_revenue"] = (
        df.groupby("product_id")["revenue"]
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .round(2)
    )
    return df


def _ntile_window(df: pd.DataFrame) -> pd.DataFrame:
    """Assign NTILE buckets for revenue distribution within each category."""
    logger.info("Computing NTILE window function")
    prod_rev = (
        df.groupby(["category", "product_id"])["revenue"]
        .sum()
        .reset_index()
        .rename(columns={"revenue": "total_product_revenue"})
    )
    prod_rev["ntile_4"] = prod_rev.groupby("category")["total_product_revenue"].transform(
        lambda x: pd.qcut(x, q=4, labels=[1, 2, 3, 4], duplicates="drop").astype(int)
    )
    prod_rev["ntile_10"] = prod_rev.groupby("category")["total_product_revenue"].transform(
        lambda x: pd.qcut(x, q=min(10, len(x.unique())), labels=False, duplicates="drop") + 1
    )

    df = df.merge(
        prod_rev[["category", "product_id", "ntile_4", "ntile_10"]],
        on=["category", "product_id"],
        how="left",
    )
    return df


def _percent_of_total_window(df: pd.DataFrame) -> pd.DataFrame:
    """Compute each product's percentage of its category's total revenue."""
    logger.info("Computing percent-of-total window function")
    cat_rev = df.groupby("category")["revenue"].transform("sum")
    df["pct_of_category_revenue"] = np.round(
        df["revenue"] / cat_rev.where(cat_rev > 0) * 100, 4
    )
    return df


def _sessionization_window(df: pd.DataFrame, timeout_minutes: int = 30) -> pd.DataFrame:
    """Detect new sessions per customer based on a timeout gap."""
    logger.info("Computing sessionization window (timeout=%dm)", timeout_minutes)
    df = df.sort_values(["customer_id", "transaction_date"]).reset_index(drop=True)

    prev_date = df.groupby("customer_id")["transaction_date"].shift(1)
    gap_seconds = (df["transaction_date"] - prev_date).dt.total_seconds()
    df["is_new_session"] = (gap_seconds > timeout_minutes * 60) | prev_date.isna()
    df["session_id"] = df.groupby("customer_id")["is_new_session"].cumsum().astype(int)
    return df


def run_all_window_analyses(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Apply every window function analysis to the dataset.

    Args:
        df: Raw transaction DataFrame.
        config: Pipeline configuration.

    Returns:
        Enriched DataFrame with all window function columns.
    """
    df = _rankings_window(df)
    df = _lag_lead_window(df)
    df = _cumulative_window(df)
    df = _moving_average_window(df)
    df = _ntile_window(df)
    df = _percent_of_total_window(df)
    df = _sessionization_window(df)
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-product analytics summary.

    Args:
        df: Enriched DataFrame.

    Returns:
        Summary DataFrame aggregated to product level.
    """
    logger.info("Building analytics summary")
    summary = (
        df.groupby(["category", "product_id", "region"])
        .agg(
            total_revenue=("revenue", "sum"),
            avg_revenue=("revenue", "mean"),
            median_revenue=("revenue", "median"),
            transaction_count=("transaction_id", "count"),
            unique_customers=("customer_id", "nunique"),
            avg_quantity=("quantity", "mean"),
            avg_discount=("discount_pct", "mean"),
            revenue_std=("revenue", "std"),
            first_transaction=("transaction_date", "min"),
            last_transaction=("transaction_date", "max"),
        )
        .reset_index()
    )
    summary = summary.round(4)
    summary["days_active"] = (
        summary["last_transaction"] - summary["first_transaction"]
    ).dt.days

    summary = summary.sort_values("total_revenue", ascending=False).reset_index(drop=True)
    return summary


def get_top_products(summary: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """Return the top-N products by total revenue.

    Args:
        summary: Product-level summary DataFrame.
        n: Number of top products to return.

    Returns:
        Top-N products DataFrame.
    """
    return summary.head(n).copy()


def save_results(
    enriched_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    top_products_df: pd.DataFrame,
    output_dir: str,
) -> dict[str, str]:
    """Persist all result DataFrames to CSV.

    Args:
        enriched_df: Full dataset with window function columns.
        summary_df: Product-level summary.
        top_products_df: Top-N products.
        output_dir: Directory to write output files.

    Returns:
        Mapping of result name to file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    for name, frame in [
        ("enriched_data", enriched_df),
        ("analysis_summary", summary_df),
        ("top_products", top_products_df),
    ]:
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = str(path)
        logger.info("Saved %s to %s", name, path)

    return paths


def run_pipeline(config: PipelineConfig) -> PipelineResults:
    """Execute the full analytics pipeline.

    Steps:
        1. Generate synthetic data
        2. Validate (if config provided)
        3. Run all window function analyses
        4. Build summary and top-N products
        5. Save all results

    Args:
        config: Pipeline configuration.

    Returns:
        PipelineResults containing all outputs and timings.

    Raises:
        PipelineError: If any stage fails.
    """
    timings: dict[str, float] = {}
    results = PipelineResults()

    try:
        t0 = time.perf_counter()
        df = generate_synthetic_data(config.n_rows, config.seed)
        timings["generate"] = round(time.perf_counter() - t0, 3)
        logger.info("Data generation: %.3fs", timings["generate"])
    except Exception as exc:
        raise PipelineError(f"Data generation failed: {exc}") from exc

        # validation

    if config.validation_config_path:
        try:
            t0 = time.perf_counter()
            _validate_dataset(df, config.validation_config_path)
            timings["validate"] = round(time.perf_counter() - t0, 3)
            logger.info("Validation: %.3fs", timings["validate"])
        except Exception as exc:
            raise PipelineError(f"Validation failed: {exc}") from exc

    try:
        t0 = time.perf_counter()
        enriched = run_all_window_analyses(df, config)
        timings["analyze"] = round(time.perf_counter() - t0, 3)
        results.enriched_df = enriched
        logger.info("Analysis: %.3fs", timings["analyze"])
    except Exception as exc:
        raise PipelineError(f"Analysis failed: {exc}") from exc

    try:
        t0 = time.perf_counter()
        summary = build_summary(enriched)
        top_products = get_top_products(summary, config.top_n)
        timings["summarize"] = round(time.perf_counter() - t0, 3)
        results.summary_df = summary
        results.top_products_df = top_products
        logger.info("Summary: %.3fs", timings["summarize"])
    except Exception as exc:
        raise PipelineError(f"Summarization failed: {exc}") from exc

    try:
        t0 = time.perf_counter()
        save_results(enriched, summary, top_products, config.output_dir)
        timings["save"] = round(time.perf_counter() - t0, 3)
        timings["total"] = round(sum(v for k, v in timings.items() if k != "total"), 3)
        results.timings = timings
        logger.info("Save: %.3fs", timings["save"])
        logger.info("Pipeline complete: %.3fs total", timings["total"])
    except Exception as exc:
        raise PipelineError(f"Saving results failed: {exc}") from exc

    return results


def _validate_dataset(df: pd.DataFrame, config_path: str | None) -> None:
    """Validate DataFrame columns and basic constraints against a config dict.

    Args:
        df: DataFrame to validate.
        config_path: Path to YAML validation config (unused; uses built-in rules).

    Raises:
        ValueError: If required columns are missing or constraints are violated.
    """
    required = {"transaction_id", "product_id", "customer_id", "category",
                "quantity", "unit_price", "discount_pct", "transaction_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty")

    if df["quantity"].min() < 1:
        raise ValueError("quantity contains values < 1")

    if (df["discount_pct"] < 0).any() or (df["discount_pct"] > 1).any():
        raise ValueError("discount_pct must be between 0 and 1")

    logger.info("Dataset validation passed (%d rows, %d cols)", *df.shape)


__all__ = [
    "PipelineConfig",
    "PipelineResults",
    "PipelineError",
    "generate_synthetic_data",
    "run_all_window_analyses",
    "build_summary",
    "get_top_products",
    "save_results",
    "run_pipeline",
]
