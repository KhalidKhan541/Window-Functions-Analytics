"""
SQL-style window functions implemented in pandas.

Provides vectorized, performant implementations of common SQL window operations
for use with large-scale DataFrames (3M+ rows). All methods return a new DataFrame
and never modify the original.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WindowFunctions:
    """SQL-style window functions implemented in pandas.

    Examples
    --------
    >>> wf = WindowFunctions(sales_df)
    >>> result = wf.rank(partition_by=["region"], order_by="revenue", ascending=False)
    >>> result = wf.running_total(column="revenue", partition_by=["region"], order_by="date")
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self._validate()

    def _validate(self) -> None:
        """Validate that input DataFrame is non-empty and has expected structure."""
        if self.df.empty:
            raise ValueError("Input DataFrame is empty.")
        logger.debug(
            "WindowFunctions initialized with %d rows and columns: %s",
            len(self.df),
            list(self.df.columns),
        )

    def _require_column(self, col: str) -> None:
        """Raise if a column is missing from the DataFrame."""
        if col not in self.df.columns:
            raise KeyError(
                f"Column '{col}' not found. Available columns: {list(self.df.columns)}"
            )

    def _require_columns(self, cols: List[str]) -> None:
        """Raise if any of the listed columns are missing."""
        for c in cols:
            self._require_column(c)

    def _resolve_col_name(self, col_name: Optional[str], default: str) -> str:
        """Return *col_name* if provided, else *default*."""
        return col_name if col_name is not None else default

    # ------------------------------------------------------------------
    # Core window functions
    # ------------------------------------------------------------------

    def row_number(
        self,
        partition_by: List[str],
        order_by: str,
        col_name: str = "row_num",
    ) -> pd.DataFrame:
        """ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...).

        Assigns a sequential integer to each row within a partition,
        starting at 1 and incrementing by 1.
        """
        result = self.df.copy()
        self._require_columns(partition_by + [order_by])
        result[col_name] = (
            result.groupby(partition_by, sort=False)
            .cumcount()
            .add(1)
            .astype("int32")
        )
        return result

    def rank(
        self,
        partition_by: List[str],
        order_by: str,
        ascending: bool = True,
        col_name: str = "rank",
    ) -> pd.DataFrame:
        """RANK() OVER (PARTITION BY ... ORDER BY ...) — with gaps.

        Rows that compare equal receive the same rank, and the next
        rank after ties is incremented by the number of tied rows.
        """
        result = self.df.copy()
        self._require_columns(partition_by + [order_by])
        result[col_name] = (
            result.groupby(partition_by, sort=False)[order_by]
            .rank(method="min", ascending=ascending)
            .astype("int32")
        )
        return result

    def dense_rank(
        self,
        partition_by: List[str],
        order_by: str,
        ascending: bool = True,
        col_name: str = "dense_rank",
    ) -> pd.DataFrame:
        """DENSE_RANK() — no gaps in ranking.

        Same as RANK() but consecutive ranks are always sequential
        regardless of ties.
        """
        result = self.df.copy()
        self._require_columns(partition_by + [order_by])
        result[col_name] = (
            result.groupby(partition_by, sort=False)[order_by]
            .rank(method="dense", ascending=ascending)
            .astype("int32")
        )
        return result

    def ntile(
        self,
        partition_by: List[str],
        order_by: str,
        n: int,
        col_name: str = "ntile",
    ) -> pd.DataFrame:
        """NTILE(n) — divide each partition into *n* buckets.

        Each bucket is assigned an integer from 1 to *n*.  Buckets are
        sized as evenly as possible; earlier buckets may be one row larger.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        result = self.df.copy()
        self._require_columns(partition_by + [order_by])

        def _assign_ntile(group: pd.DataFrame) -> pd.Series:
            total = len(group)
            bucket_size = total / n
            return pd.Series(
                np.floor(np.arange(total) / bucket_size).astype(int).clip(upper=n - 1)
                + 1,
                index=group.index,
            )

        result[col_name] = (
            result.sort_values(order_by)
            .groupby(partition_by, sort=False)
            .apply(_assign_ntile)
            .droplevel(0)
            .reindex(result.index)
            .astype("int32")
        )
        return result

    def lag(
        self,
        column: str,
        partition_by: List[str],
        order_by: str,
        periods: int = 1,
        col_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """LAG(column, periods) OVER (PARTITION BY ... ORDER BY ...).

        Returns the value from *periods* rows before the current row
        within the partition.  Rows outside the partition produce NaN.
        """
        col_name = self._resolve_col_name(col_name, f"{column}_lag_{periods}")
        result = self.df.copy()
        self._require_columns(partition_by + [order_by, column])
        result[col_name] = (
            result.sort_values(order_by)
            .groupby(partition_by, sort=False)[column]
            .shift(periods)
        )
        return result

    def lead(
        self,
        column: str,
        partition_by: List[str],
        order_by: str,
        periods: int = 1,
        col_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """LEAD(column, periods) OVER (PARTITION BY ... ORDER BY ...).

        Returns the value from *periods* rows after the current row
        within the partition.  Rows outside the partition produce NaN.
        """
        col_name = self._resolve_col_name(col_name, f"{column}_lead_{periods}")
        result = self.df.copy()
        self._require_columns(partition_by + [order_by, column])
        result[col_name] = (
            result.sort_values(order_by)
            .groupby(partition_by, sort=False)[column]
            .shift(-periods)
        )
        return result

    # ------------------------------------------------------------------
    # Aggregate window functions
    # ------------------------------------------------------------------

    def running_total(
        self,
        column: str,
        partition_by: List[str],
        order_by: str,
    ) -> pd.DataFrame:
        """SUM(column) OVER (PARTITION BY ... ORDER BY ... ROWS UNBOUNDED PRECEDING).

        Cumulative sum within each partition ordered by *order_by*.
        """
        result = self.df.copy()
        self._require_columns(partition_by + [order_by, column])
        result[f"{column}_running_sum"] = (
            result.sort_values(order_by)
            .groupby(partition_by, sort=False)[column]
            .cumsum()
        )
        return result

    def moving_average(
        self,
        column: str,
        partition_by: List[str],
        order_by: str,
        window_size: int = 7,
    ) -> pd.DataFrame:
        """AVG(column) OVER (PARTITION BY ... ORDER BY ... ROWS BETWEEN N PRECEDING AND CURRENT ROW).

        Rolling (trailing) mean of the last *window_size* rows (inclusive
        of the current row).
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        result = self.df.copy()
        self._require_columns(partition_by + [order_by, column])
        result[f"{column}_ma_{window_size}"] = (
            result.sort_values(order_by)
            .groupby(partition_by, sort=False)[column]
            .transform(lambda s: s.rolling(window_size, min_periods=1).mean())
        )
        return result

    def cumulative_avg(
        self,
        column: str,
        partition_by: List[str],
        order_by: str,
    ) -> pd.DataFrame:
        """Cumulative average (running mean) within each partition."""
        result = self.df.copy()
        self._require_columns(partition_by + [order_by, column])
        result[f"{column}_cum_avg"] = (
            result.sort_values(order_by)
            .groupby(partition_by, sort=False)[column]
            .transform(lambda s: s.expanding().mean())
        )
        return result

    def cumulative_count(
        self,
        partition_by: List[str],
        order_by: str,
    ) -> pd.DataFrame:
        """Cumulative row count within each partition (1-based)."""
        result = self.df.copy()
        self._require_columns(partition_by + [order_by])
        result["cum_count"] = (
            result.sort_values(order_by)
            .groupby(partition_by, sort=False)
            .cumcount()
            .add(1)
            .astype("int32")
        )
        return result

    # ------------------------------------------------------------------
    # Percentage / ratio functions
    # ------------------------------------------------------------------

    def percent_of_total(
        self,
        column: str,
        partition_by: List[str],
    ) -> pd.DataFrame:
        """column / SUM(column) OVER (PARTITION BY ...) * 100.

        Expresses each row's value as a percentage of its partition total.
        """
        result = self.df.copy()
        self._require_columns(partition_by + [column])
        partition_sum = result.groupby(partition_by, sort=False)[column].transform("sum")
        result[f"{column}_pct_of_total"] = np.where(
            partition_sum != 0,
            result[column] / partition_sum * 100,
            np.nan,
        )
        return result

    def cumulative_percent(
        self,
        column: str,
        partition_by: List[str],
        order_by: str,
    ) -> pd.DataFrame:
        """Cumulative percentage of partition total.

        Each row shows what fraction of the partition's total has been
        accumulated up to and including that row.
        """
        result = self.df.copy()
        self._require_columns(partition_by + [order_by, column])
        grouped = result.sort_values(order_by).groupby(partition_by, sort=False)[column]
        partition_sum = grouped.transform("sum")
        cumsum = grouped.cumsum()
        result[f"{column}_cum_pct"] = np.where(
            partition_sum != 0,
            cumsum / partition_sum * 100,
            np.nan,
        )
        return result

    # ------------------------------------------------------------------
    # Period-over-period analysis
    # ------------------------------------------------------------------

    def period_over_period(
        self,
        column: str,
        order_by: str,
        col_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """(current - previous) / previous * 100 using LAG.

        Percentage change from the immediately preceding row (unpartitioned).
        """
        col_name = self._resolve_col_name(col_name, f"{column}_pct_change")
        result = self.df.copy()
        self._require_columns([order_by, column])
        sorted_df = result.sort_values(order_by)
        prev = sorted_df[column].shift(1)
        result[col_name] = np.where(
            prev != 0,
            (result[column] - prev) / prev * 100,
            np.nan,
        )
        return result

    def yoy_growth(
        self,
        column: str,
        partition_by: List[str],
        order_by: str,
    ) -> pd.DataFrame:
        """Year-over-year (period-over-period) growth percentage.

        LAG(column, 1) within each partition.
        """
        result = self.df.copy()
        self._require_columns(partition_by + [order_by, column])
        grouped = (
            result.sort_values(order_by)
            .groupby(partition_by, sort=False)[column]
        )
        prev = grouped.shift(1)
        result[f"{column}_yoy_pct"] = np.where(
            prev != 0,
            (result[column] - prev) / prev * 100,
            np.nan,
        )
        return result

    # ------------------------------------------------------------------
    # Percentile / distribution functions
    # ------------------------------------------------------------------

    def percentile_rank(
        self,
        column: str,
        partition_by: List[str],
    ) -> pd.DataFrame:
        """PERCENT_RANK() OVER (PARTITION BY ...).

        Returns a value in [0, 1] representing where the row falls in the
        partition.  The lowest value maps to 0.
        """
        result = self.df.copy()
        self._require_columns(partition_by + [column])
        result["percent_rank"] = result.groupby(partition_by, sort=False)[column].rank(
            method="min", pct=True
        )
        return result

    def cume_dist(
        self,
        column: str,
        partition_by: List[str],
        order_by: str,
    ) -> pd.DataFrame:
        """CUME_DIST() — cumulative distribution.

        Fraction of rows in the partition whose values are less than or
        equal to the current row's value.
        """
        result = self.df.copy()
        self._require_columns(partition_by + [order_by, column])
        result["cume_dist"] = (
            result.sort_values(order_by)
            .groupby(partition_by, sort=False)[column]
            .rank(method="max", pct=True)
        )
        return result
