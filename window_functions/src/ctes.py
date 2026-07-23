"""CTE (Common Table Expression) simulation module for pandas.

Provides a fluent interface to build SQL-like CTE pipelines using pandas
method chaining. Useful for composing multi-step data transformations that
mirror SQL CTE syntax.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd


class CTEBuilderError(Exception):
    """Raised when a CTE builder operation fails."""


class CTEBuilder:
    """Simulate SQL CTEs with pandas method chaining.

    Enables defining named intermediate DataFrames (CTEs) and composing
    SQL-like operations (SELECT, WHERE, GROUP BY, etc.) in a fluent style.

    Example::

        result = (
            CTEBuilder(raw_sales)
            .cte("region_totals", lambda df: df.groupby("region")["revenue"].sum().reset_index())
            .cte("high_regions", lambda df: df[df["revenue"] > 1_000_000])
            .select(["region", "revenue"])
            .order_by(["revenue"], ascending=False)
            .limit(10)
            .result()
        )
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")
        self._df: pd.DataFrame = df.copy()
        self._ctes: Dict[str, pd.DataFrame] = {}
        self._chain: List[Callable[[pd.DataFrame], pd.DataFrame]] = []
        self._active_cte: Optional[str] = None

    def cte(self, name: str, func: Callable[[pd.DataFrame], pd.DataFrame]) -> CTEBuilder:
        """Define a CTE. func receives a DataFrame and returns a DataFrame.

        The CTE is evaluated lazily during ``result()``. If a CTE with the
        same name already exists, it is overwritten.

        Args:
            name: Unique name for this CTE.
            func: Callable that transforms a DataFrame into the CTE result.

        Returns:
            self (for chaining).

        Raises:
            CTEBuilderError: If *func* is not callable or returns a non-DataFrame.
        """
        if not callable(func):
            raise CTEBuilderError(f"CTE 'func' must be callable, got {type(func).__name__}")
        if not isinstance(name, str) or not name.strip():
            raise CTEBuilderError("CTE name must be a non-empty string")

        def _apply(df: pd.DataFrame, _f: Callable = func, _n: str = name) -> pd.DataFrame:
            try:
                result = _f(df)
            except Exception as exc:
                raise CTEBuilderError(f"CTE '{_n}' raised an error: {exc}") from exc
            if not isinstance(result, pd.DataFrame):
                raise CTEBuilderError(
                    f"CTE '{_n}' must return a DataFrame, got {type(result).__name__}"
                )
            return result

        self._chain.append(_apply)
        self._active_cte = name
        return self

    def select(self, columns: List[str]) -> CTEBuilder:
        """SELECT columns from the current result.

        Args:
            columns: Column names to keep.

        Returns:
            self (for chaining).

        Raises:
            CTEBuilderError: If *columns* is not a non-empty list of strings.
        """
        self._validate_column_list(columns, "select")

        def _apply(df: pd.DataFrame, _cols: List[str] = columns) -> pd.DataFrame:
            missing = set(_cols) - set(df.columns)
            if missing:
                raise CTEBuilderError(
                    f"SELECT columns not found in DataFrame: {sorted(missing)}"
                )
            return df[_cols].copy()

        self._chain.append(_apply)
        return self

    def where(self, condition: pd.Series) -> CTEBuilder:
        """WHERE clause.

        Args:
            condition: Boolean pandas Series aligned with the current DataFrame.

        Returns:
            self (for chaining).

        Raises:
            CTEBuilderError: If *condition* is not a boolean Series.
        """
        self._validate_series(condition, "where")

        def _apply(df: pd.DataFrame, _cond: pd.Series = condition) -> pd.DataFrame:
            if len(_cond) != len(df):
                raise CTEBuilderError(
                    f"WHERE condition length ({len(_cond)}) does not match "
                    f"DataFrame length ({len(df)})"
                )
            return df.loc[_cond].copy()

        self._chain.append(_apply)
        return self

    def group_by(
        self, columns: List[str], agg_dict: Dict[str, str]
    ) -> CTEBuilder:
        """GROUP BY with aggregation.

        Args:
            columns: Columns to group by.
            agg_dict: Mapping of column name to aggregation function name
                      (e.g. ``{"revenue": "sum", "quantity": "mean"}``).

        Returns:
            self (for chaining).

        Raises:
            CTEBuilderError: If inputs are invalid.
        """
        self._validate_column_list(columns, "group_by")
        if not isinstance(agg_dict, dict) or not agg_dict:
            raise CTEBuilderError("agg_dict must be a non-empty dictionary")

        def _apply(
            df: pd.DataFrame,
            _cols: List[str] = columns,
            _agg: Dict[str, str] = agg_dict,
        ) -> pd.DataFrame:
            missing_cols = set(_cols) - set(df.columns)
            if missing_cols:
                raise CTEBuilderError(
                    f"GROUP BY columns not found: {sorted(missing_cols)}"
                )
            missing_agg = set(_agg.keys()) - set(df.columns)
            if missing_agg:
                raise CTEBuilderError(
                    f"Aggregation columns not found: {sorted(missing_agg)}"
                )
            grouped = df.groupby(_cols, as_index=False).agg(_agg)
            return grouped

        self._chain.append(_apply)
        return self

    def having(self, condition: pd.Series) -> CTEBuilder:
        """HAVING clause (filter after group_by).

        Typically used after ``group_by`` to filter aggregated results.

        Args:
            condition: Boolean pandas Series to filter on.

        Returns:
            self (for chaining).

        Raises:
            CTEBuilderError: If *condition* is not a boolean Series.
        """
        self._validate_series(condition, "having")

        def _apply(df: pd.DataFrame, _cond: pd.Series = condition) -> pd.DataFrame:
            if len(_cond) != len(df):
                raise CTEBuilderError(
                    f"HAVING condition length ({len(_cond)}) does not match "
                    f"DataFrame length ({len(df)})"
                )
            return df.loc[_cond].copy()

        self._chain.append(_apply)
        return self

    def order_by(
        self,
        columns: List[str],
        ascending: Union[bool, List[bool]] = True,
    ) -> CTEBuilder:
        """ORDER BY.

        Args:
            columns: Column(s) to sort by.
            ascending: Sort direction. If a single bool, applies to all columns.

        Returns:
            self (for chaining).

        Raises:
            CTEBuilderError: If inputs are invalid.
        """
        self._validate_column_list(columns, "order_by")

        def _apply(
            df: pd.DataFrame,
            _cols: List[str] = columns,
            _asc: Union[bool, List[bool]] = ascending,
        ) -> pd.DataFrame:
            missing = set(_cols) - set(df.columns)
            if missing:
                raise CTEBuilderError(
                    f"ORDER BY columns not found: {sorted(missing)}"
                )
            return df.sort_values(by=_cols, ascending=_asc, ignore_index=True).copy()

        self._chain.append(_apply)
        return self

    def limit(self, n: int) -> CTEBuilder:
        """LIMIT.

        Args:
            n: Maximum number of rows to return.

        Returns:
            self (for chaining).

        Raises:
            CTEBuilderError: If *n* is not a positive integer.
        """
        if not isinstance(n, int) or n <= 0:
            raise CTEBuilderError(f"LIMIT must be a positive integer, got {n}")

        def _apply(df: pd.DataFrame, _n: int = n) -> pd.DataFrame:
            return df.head(_n).copy()

        self._chain.append(_apply)
        return self

    def result(self) -> pd.DataFrame:
        """Execute all CTEs and return the final result.

        Applies every operation in the chain sequentially to produce the
        final DataFrame.

        Returns:
            The resulting DataFrame.

        Raises:
            CTEBuilderError: If the chain is empty or any step fails.
        """
        if not self._chain:
            return self._df.copy()

        current = self._df.copy()
        for step in self._chain:
            try:
                current = step(current)
            except CTEBuilderError:
                raise
            except Exception as exc:
                raise CTEBuilderError(f"Step in CTE chain failed: {exc}") from exc
        return current

    def reset(self) -> CTEBuilder:
        """Reset the builder to the original DataFrame, clearing the chain.

        Returns:
            self (for chaining).
        """
        self._chain.clear()
        self._active_cte = None
        return self

    @property
    def columns(self) -> List[str]:
        """Return column names of the current underlying DataFrame."""
        return list(self._df.columns)

    @property
    def shape(self) -> tuple:
        """Return ``(rows, columns)`` of the current underlying DataFrame."""
        return self._df.shape

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_column_list(columns: Any, method: str) -> None:
        if not isinstance(columns, list) or not columns:
            raise CTEBuilderError(f"{method}: columns must be a non-empty list of strings")
        if not all(isinstance(c, str) for c in columns):
            raise CTEBuilderError(f"{method}: all column names must be strings")

    @staticmethod
    def _validate_series(series: Any, method: str) -> None:
        if not isinstance(series, pd.Series):
            raise CTEBuilderError(
                f"{method}: condition must be a pandas Series, got {type(series).__name__}"
            )
        if not pd.api.types.is_bool_dtype(series):
            raise CTEBuilderError(f"{method}: condition must be a boolean Series")

    def __repr__(self) -> str:
        chain_len = len(self._chain)
        return (
            f"CTEBuilder(rows={self._df.shape[0]}, cols={self._df.shape[1]}, "
            f"chain_steps={chain_len})"
        )
