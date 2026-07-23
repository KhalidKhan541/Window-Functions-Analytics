"""Pandas DataFrame validation layer for analytics pipelines."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate DataFrames before analysis.

    Collects errors and warnings during validation so that callers can
    inspect the full report before deciding whether to proceed or clean.

    Args:
        df: The DataFrame to validate.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.errors: List[str] = []
        self.warnings: List[str] = []

    # ------------------------------------------------------------------
    # Schema checks
    # ------------------------------------------------------------------

    def validate_schema(self, required_columns: Dict[str, str]) -> bool:
        """Check required columns exist with correct dtypes.

        Args:
            required_columns: Mapping of column name to expected dtype string
                (e.g. ``{"sale_id": "int64", "date": "datetime64"}``).

        Returns:
            True if all checks pass.
        """
        ok = True
        for col, expected in required_columns.items():
            if col not in self.df.columns:
                self.errors.append(f"Missing required column: '{col}'")
                ok = False
                continue
            actual = str(self.df[col].dtype)
            if expected not in actual:
                self.errors.append(
                    f"Column '{col}' dtype mismatch: expected ~'{expected}', got '{actual}'"
                )
                ok = False
        return ok

    # ------------------------------------------------------------------
    # Null / uniqueness checks
    # ------------------------------------------------------------------

    def validate_no_nulls(self, columns: List[str]) -> bool:
        """Check for null values in the given columns.

        Args:
            columns: Column names to inspect.

        Returns:
            True if no nulls are found.
        """
        ok = True
        for col in columns:
            if col not in self.df.columns:
                self.errors.append(f"Column '{col}' not found for null check")
                ok = False
                continue
            n_nulls = int(self.df[col].isna().sum())
            if n_nulls > 0:
                pct = n_nulls / len(self.df) * 100
                self.errors.append(
                    f"Column '{col}' has {n_nulls:,} nulls ({pct:.2f}%)"
                )
                ok = False
        return ok

    def validate_unique(self, columns: List[str]) -> bool:
        """Check uniqueness constraint on one or more columns.

        Args:
            columns: Column names that should form a unique key.

        Returns:
            True if all values (or combinations) are unique.
        """
        ok = True
        subset = self.df[columns]
        n_before = len(subset)
        n_after = len(subset.drop_duplicates())
        n_dupes = n_before - n_after
        if n_dupes > 0:
            self.errors.append(
                f"Columns {columns} have {n_dupes:,} duplicate rows"
            )
            ok = False
        return ok

    # ------------------------------------------------------------------
    # Range / boundary checks
    # ------------------------------------------------------------------

    def validate_ranges(
        self, column_ranges: Dict[str, Tuple[float, float]]
    ) -> bool:
        """Check that numeric columns fall within expected ranges.

        Args:
            column_ranges: Mapping of column name to ``(min, max)`` tuple.

        Returns:
            True if all values are within bounds.
        """
        ok = True
        for col, (lo, hi) in column_ranges.items():
            if col not in self.df.columns:
                self.errors.append(f"Column '{col}' not found for range check")
                ok = False
                continue
            col_min = self.df[col].min()
            col_max = self.df[col].max()
            if col_min < lo:
                self.errors.append(
                    f"Column '{col}' below range: min={col_min}, expected >= {lo}"
                )
                ok = False
            if col_max > hi:
                self.errors.append(
                    f"Column '{col}' above range: max={col_max}, expected <= {hi}"
                )
                ok = False
        return ok

    def validate_dates(
        self,
        date_column: str,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
    ) -> bool:
        """Validate a datetime column.

        Checks that the column is datetime-typed, has no nulls, and (optionally)
        falls within the specified bounds.

        Args:
            date_column: Name of the datetime column.
            min_date: Earliest acceptable date as ISO string.
            max_date: Latest acceptable date as ISO string.

        Returns:
            True if valid.
        """
        ok = True
        if date_column not in self.df.columns:
            self.errors.append(f"Date column '{date_column}' not found")
            return False

        if not pd.api.types.is_datetime64_any_dtype(self.df[date_column]):
            self.errors.append(
                f"Column '{date_column}' is not datetime type: {self.df[date_column].dtype}"
            )
            ok = False
            return ok

        n_nulls = int(self.df[date_column].isna().sum())
        if n_nulls > 0:
            self.errors.append(
                f"Date column '{date_column}' has {n_nulls:,} nulls"
            )
            ok = False

        if min_date is not None:
            cutoff = pd.Timestamp(min_date)
            below = int((self.df[date_column] < cutoff).sum())
            if below > 0:
                self.errors.append(
                    f"{below:,} rows in '{date_column}' before {min_date}"
                )
                ok = False

        if max_date is not None:
            cutoff = pd.Timestamp(max_date)
            above = int((self.df[date_column] > cutoff).sum())
            if above > 0:
                self.errors.append(
                    f"{above:,} rows in '{date_column}' after {max_date}"
                )
                ok = False

        return ok

    # ------------------------------------------------------------------
    # Referential integrity
    # ------------------------------------------------------------------

    def validate_referential(
        self, df2: pd.DataFrame, on: str
    ) -> bool:
        """Check referential integrity between self.df and df2.

        Every value in ``self.df[on]`` must appear in ``df2[on]``.

        Args:
            df2: The parent / reference DataFrame.
            on: Foreign-key column name.

        Returns:
            True if all references resolve.
        """
        ok = True
        if on not in self.df.columns:
            self.errors.append(f"Column '{on}' not found in source DataFrame")
            return False
        if on not in df2.columns:
            self.errors.append(f"Column '{on}' not found in reference DataFrame")
            return False

        source_vals = set(self.df[on].unique())
        ref_vals = set(df2[on].unique())
        orphans = source_vals - ref_vals
        if orphans:
            n_orphans = len(orphans)
            self.warnings.append(
                f"{n_orphans} values in '{on}' have no match in reference DataFrame"
            )
            ok = False
        return ok

    # ------------------------------------------------------------------
    # Composite validation
    # ------------------------------------------------------------------

    def full_validation(self, config: dict) -> Dict[str, Any]:
        """Run all validations defined in *config* and return a report.

        Config keys:
            - ``schema``: ``Dict[str, str]`` passed to :meth:`validate_schema`
            - ``no_nulls``: ``List[str]`` passed to :meth:`validate_no_nulls`
            - ``unique``: ``List[str]`` passed to :meth:`validate_unique`
            - ``ranges``: ``Dict[str, Tuple[float,float]]`` passed to :meth:`validate_ranges`
            - ``dates``: ``Dict`` with keys ``column``, ``min_date``, ``max_date``
            - ``referential``: ``Dict`` with keys ``df2``, ``on``

        Args:
            config: Validation configuration dictionary.

        Returns:
            Report dict with ``ok``, ``errors``, ``warnings``, and ``checks_run``.
        """
        self.errors.clear()
        self.warnings.clear()
        checks_run: List[str] = []

        if "schema" in config:
            self.validate_schema(config["schema"])
            checks_run.append("schema")

        if "no_nulls" in config:
            self.validate_no_nulls(config["no_nulls"])
            checks_run.append("no_nulls")

        if "unique" in config:
            self.validate_unique(config["unique"])
            checks_run.append("unique")

        if "ranges" in config:
            self.validate_ranges(config["ranges"])
            checks_run.append("ranges")

        if "dates" in config:
            dc = config["dates"]
            self.validate_dates(
                dc["column"],
                min_date=dc.get("min_date"),
                max_date=dc.get("max_date"),
            )
            checks_run.append("dates")

        if "referential" in config:
            ref = config["referential"]
            self.validate_referential(ref["df2"], ref["on"])
            checks_run.append("referential")

        report = {
            "ok": len(self.errors) == 0,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checks_run": checks_run,
        }

        if report["ok"]:
            logger.info("Validation passed (%d checks)", len(checks_run))
        else:
            logger.warning(
                "Validation failed: %d errors, %d warnings",
                len(self.errors),
                len(self.warnings),
            )

        return report

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def clean(self) -> pd.DataFrame:
        """Return a cleaned copy of the DataFrame.

        Steps:
            1. Remove exact duplicate rows.
            2. Drop columns that are entirely null.
            3. Forward-fill then back-fill remaining nulls in numeric columns.

        Returns:
            Cleaned DataFrame.
        """
        cleaned = self.df.copy()
        n_before = len(cleaned)

        cleaned = cleaned.drop_duplicates()
        n_dupes = n_before - len(cleaned)
        if n_dupes > 0:
            logger.info("Removed %d duplicate rows", n_dupes)
            self.warnings.append(f"Removed {n_dupes} duplicate rows")

        null_cols = [c for c in cleaned.columns if cleaned[c].isna().all()]
        if null_cols:
            cleaned = cleaned.drop(columns=null_cols)
            logger.info("Dropped all-null columns: %s", null_cols)
            self.warnings.append(f"Dropped all-null columns: {null_cols}")

        num_cols = cleaned.select_dtypes(include=[np.number]).columns
        n_nulls_before = int(cleaned[num_cols].isna().sum().sum())
        if n_nulls_before > 0:
            cleaned[num_cols] = cleaned[num_cols].ffill().bfill()
            logger.info("Filled %d nulls in numeric columns", n_nulls_before)
            self.warnings.append(
                f"Filled {n_nulls_before} nulls in numeric columns via ffill/bfill"
            )

        logger.info(
            "Cleaning complete: %d -> %d rows", n_before, len(cleaned)
        )
        return cleaned
