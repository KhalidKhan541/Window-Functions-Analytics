"""Advanced sales analytics module using window functions.

Combines window functions into real-world analytics pipelines: running totals,
cumulative metrics, period-over-period comparisons, percentile rankings,
multi-window moving averages, and customer segmentation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class AnalysisError(Exception):
    """Raised when an analysis operation fails."""


class WindowFunctions:
    """Lightweight window-function helpers for pandas DataFrames.

    Provides scalar aggregation windows (cumsum, rank, etc.) without
    requiring SQL or third-party dependencies.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")
        self._df = df

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def _require_columns(self, columns: List[str], caller: str) -> None:
        missing = set(columns) - set(self._df.columns)
        if missing:
            raise AnalysisError(
                f"{caller}: required columns not found: {sorted(missing)}"
            )

    def _require_numeric(self, column: str, caller: str) -> None:
        if not pd.api.types.is_numeric_dtype(self._df[column]):
            raise AnalysisError(f"{caller}: column '{column}' must be numeric")

    def cumulative_sum(
        self, column: str, group_by: Optional[List[str]] = None
    ) -> pd.Series:
        if group_by:
            self._require_columns(group_by, "cumulative_sum")
        self._require_numeric(column, "cumulative_sum")
        if group_by:
            return self._df.groupby(group_by)[column].cumsum()
        return self._df[column].cumsum()

    def rank(
        self,
        column: str,
        method: str = "dense",
        ascending: bool = True,
        group_by: Optional[List[str]] = None,
    ) -> pd.Series:
        if group_by:
            self._require_columns(group_by, "rank")
        self._require_numeric(column, "rank")
        if group_by:
            return self._df.groupby(group_by)[column].rank(
                method=method, ascending=ascending
            )
        return self._df[column].rank(method=method, ascending=ascending)

    def lag(
        self, column: str, periods: int = 1, group_by: Optional[List[str]] = None
    ) -> pd.Series:
        if group_by:
            self._require_columns(group_by, "lag")
        self._require_numeric(column, "lag")
        if group_by:
            return self._df.groupby(group_by)[column].shift(periods)
        return self._df[column].shift(periods)

    def lead(
        self, column: str, periods: int = 1, group_by: Optional[List[str]] = None
    ) -> pd.Series:
        if group_by:
            self._require_columns(group_by, "lead")
        self._require_numeric(column, "lead")
        if group_by:
            return self._df.groupby(group_by)[column].shift(-periods)
        return self._df[column].shift(-periods)

    def rolling_mean(
        self, column: str, window: int, group_by: Optional[List[str]] = None
    ) -> pd.Series:
        if group_by:
            self._require_columns(group_by, "rolling_mean")
        self._require_numeric(column, "rolling_mean")
        if group_by:
            return (
                self._df.groupby(group_by)[column]
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
        return self._df[column].rolling(window, min_periods=1).mean()

    def percentile_rank(
        self, column: str, group_by: Optional[List[str]] = None
    ) -> pd.Series:
        if group_by:
            self._require_columns(group_by, "percentile_rank")
        self._require_numeric(column, "percentile_rank")
        if group_by:
            return self._df.groupby(group_by)[column].rank(pct=True)
        return self._df[column].rank(pct=True)

    def ntile(
        self, column: str, n: int, group_by: Optional[List[str]] = None
    ) -> pd.Series:
        if group_by:
            self._require_columns(group_by, "ntile")
        self._require_numeric(column, "ntile")
        if group_by:
            return self._df.groupby(group_by)[column].transform(
                lambda s: pd.qcut(s, q=n, labels=False, duplicates="drop") + 1
            )
        return pd.qcut(self._df[column], q=n, labels=False, duplicates="drop") + 1


class SalesAnalyzer:
    """Advanced sales analytics using window functions.

    Expects a DataFrame with at least:

    - ``date`` – datetime or date-like column
    - ``region`` – categorical region identifier
    - ``product`` – product name or ID
    - ``revenue`` – numeric revenue column
    - ``quantity`` – numeric quantity column
    - ``customer_id`` – customer identifier

    Additional columns are preserved through transformations.
    """

    _REQUIRED_COLS = ("date", "region", "product", "revenue", "quantity", "customer_id")

    def __init__(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")
        self._validate_required(df)
        self.df: pd.DataFrame = df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.wf: WindowFunctions = WindowFunctions(self.df)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_required(df: pd.DataFrame) -> None:
        missing = [c for c in SalesAnalyzer._REQUIRED_COLS if c not in df.columns]
        if missing:
            raise AnalysisError(
                f"DataFrame missing required columns: {sorted(missing)}"
            )
        for col in ("revenue", "quantity"):
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise AnalysisError(f"Column '{col}' must be numeric")

    # ------------------------------------------------------------------
    # Public analyses
    # ------------------------------------------------------------------

    def sales_summary(self) -> pd.DataFrame:
        """Running totals and cumulative metrics per region.

        Adds columns:
        - ``running_revenue`` – cumulative revenue per region (ordered by date)
        - ``running_quantity`` – cumulative quantity per region
        - ``revenue_pct_of_region`` – each row's revenue as % of region total
        - ``cumulative_avg_order_value`` – running average order value per region
        """
        df = self.df.sort_values(["region", "date"]).copy()

        df["running_revenue"] = self.wf.cumulative_sum("revenue", group_by=["region"])
        df["running_quantity"] = self.wf.cumulative_sum("quantity", group_by=["region"])

        region_totals = df.groupby("region")["revenue"].transform("sum")
        df["revenue_pct_of_region"] = np.where(
            region_totals > 0, df["running_revenue"] / region_totals * 100, 0.0
        )

        df["cumulative_avg_order_value"] = df["running_revenue"] / df["running_quantity"]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        return df

    def top_n_products(self, n: int = 10) -> pd.DataFrame:
        """Top N products by revenue using dense_rank().

        Returns a DataFrame with one row per product, ranked by total
        revenue within each region.

        Args:
            n: Number of top products to return per region.

        Returns:
            DataFrame with columns: region, product, total_revenue,
            total_quantity, avg_order_value, rank.
        """
        if n <= 0:
            raise AnalysisError(f"n must be a positive integer, got {n}")

        agg = (
            self.df.groupby(["region", "product"], as_index=False)
            .agg(total_revenue=("revenue", "sum"), total_quantity=("quantity", "sum"))
        )
        agg["avg_order_value"] = np.where(
            agg["total_quantity"] > 0,
            agg["total_revenue"] / agg["total_quantity"],
            0.0,
        )
        agg["rank"] = agg.groupby("region")["total_revenue"].rank(
            method="dense", ascending=False
        )
        return agg[agg["rank"] <= n].sort_values(["region", "rank"]).reset_index(drop=True)

    def period_comparison(self) -> pd.DataFrame:
        """Period-over-period comparison with LAG.

        Computes month-over-month changes per region.

        Adds columns:
        - ``prev_month_revenue`` – previous month's revenue
        - ``revenue_change`` – absolute change from previous month
        - ``revenue_change_pct`` – percentage change from previous month
        - ``prev_month_quantity`` – previous month's quantity
        - ``quantity_change_pct`` – quantity percentage change
        """
        monthly = (
            self.df.set_index("date")
            .groupby("region")
            .resample("ME")
            .agg(revenue=("revenue", "sum"), quantity=("quantity", "sum"))
            .reset_index()
        )

        wf = WindowFunctions(monthly)
        monthly["prev_month_revenue"] = wf.lag("revenue", periods=1, group_by=["region"])
        monthly["revenue_change"] = monthly["revenue"] - monthly["prev_month_revenue"]
        monthly["revenue_change_pct"] = np.where(
            monthly["prev_month_revenue"].abs() > 0,
            monthly["revenue_change"] / monthly["prev_month_revenue"] * 100,
            0.0,
        )

        monthly["prev_month_quantity"] = wf.lag("quantity", periods=1, group_by=["region"])
        monthly["quantity_change_pct"] = np.where(
            monthly["prev_month_quantity"].abs() > 0,
            (monthly["quantity"] - monthly["prev_month_quantity"])
            / monthly["prev_month_quantity"]
            * 100,
            0.0,
        )

        monthly.replace([np.inf, -np.inf], np.nan, inplace=True)
        return monthly

    def regional_percentiles(self) -> pd.DataFrame:
        """Percentile ranking per region.

        Adds columns:
        - ``revenue_percentile`` – percentile rank of revenue within region
        - ``quantity_percentile`` – percentile rank of quantity within region
        """
        df = self.df.copy()
        df["revenue_percentile"] = self.wf.percentile_rank(
            "revenue", group_by=["region"]
        )
        df["quantity_percentile"] = self.wf.percentile_rank(
            "quantity", group_by=["region"]
        )
        return df

    def moving_averages(
        self, windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Multi-window moving averages on revenue, grouped by region.

        For each window size *w* adds a column ``revenue_ma_{w}``.

        Args:
            windows: List of window sizes in rows. Defaults to [7, 14, 30].

        Returns:
            DataFrame sorted by region and date with moving average columns.
        """
        if windows is None:
            windows = [7, 14, 30]
        if not windows or not all(isinstance(w, int) and w > 0 for w in windows):
            raise AnalysisError("windows must be a non-empty list of positive integers")

        df = self.df.sort_values(["region", "date"]).copy()
        for w in windows:
            col = f"revenue_ma_{w}"
            df[col] = (
                df.groupby("region")["revenue"]
                .rolling(w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
        return df

    def customer_segmentation(self) -> pd.DataFrame:
        """NTILE-based customer segmentation (quartiles).

        Segments customers by total spending into 4 quartiles per region.

        Adds columns:
        - ``total_customer_revenue`` – total revenue per customer
        - ``segment`` – quartile label (1 = lowest spenders, 4 = highest)
        """
        customer_rev = (
            self.df.groupby(["region", "customer_id"], as_index=False)
            .agg(total_customer_revenue=("revenue", "sum"))
        )

        wf = WindowFunctions(customer_rev)
        customer_rev["segment"] = wf.ntile(
            "total_customer_revenue", n=4, group_by=["region"]
        )

        label_map = {1: "Low", 2: "Medium-Low", 3: "Medium-High", 4: "High"}
        customer_rev["segment_label"] = customer_rev["segment"].map(label_map)

        return customer_rev.sort_values(["region", "segment"], ascending=[True, False])

    def full_analysis(self) -> Dict[str, pd.DataFrame]:
        """Run all analyses and return results dict.

        Returns:
            Dictionary mapping analysis name to its resulting DataFrame.

        Raises:
            AnalysisError: If any analysis step fails.
        """
        analyses = {
            "sales_summary": self.sales_summary,
            "top_n_products": self.top_n_products,
            "period_comparison": self.period_comparison,
            "regional_percentiles": self.regional_percentiles,
            "moving_averages": self.moving_averages,
            "customer_segmentation": self.customer_segmentation,
        }
        results: Dict[str, pd.DataFrame] = {}
        for name, func in analyses.items():
            try:
                results[name] = func()
            except Exception as exc:
                raise AnalysisError(f"Analysis '{name}' failed: {exc}") from exc
        return results

    def __repr__(self) -> str:
        return (
            f"SalesAnalyzer(rows={len(self.df)}, "
            f"cols={list(self.df.columns)})"
        )
