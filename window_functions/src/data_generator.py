"""Synthetic sales data generator for window function analytics."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CATEGORY_CONFIG = {
    "Electronics": {"price_range": (50.0, 500.0), "margin": 0.35, "weight": 0.25},
    "Clothing": {"price_range": (10.0, 200.0), "margin": 0.55, "weight": 0.25},
    "Food": {"price_range": (5.0, 50.0), "margin": 0.40, "weight": 0.20},
    "Furniture": {"price_range": (30.0, 400.0), "margin": 0.30, "weight": 0.15},
    "Software": {"price_range": (20.0, 300.0), "margin": 0.70, "weight": 0.15},
}

REGIONS = ["North", "South", "East", "West", "Central"]
CHANNELS = ["Online", "Retail", "Wholesale"]
PRIORITIES = ["Low", "Medium", "High"]


class SalesDataGenerator:
    """Generate realistic synthetic sales data for 3M+ rows.

    Produces a DataFrame with seasonal patterns, category-specific price
    distributions, and realistic correlations between fields.

    Args:
        n_rows: Number of rows to generate. Defaults to 3_000_000.
        seed: Random seed for reproducibility. Defaults to 42.
    """

    def __init__(self, n_rows: int = 3_000_000, seed: int = 42):
        self.n_rows = n_rows
        self.seed = seed
        np.random.seed(seed)
        self._categories = list(CATEGORY_CONFIG.keys())
        self._category_weights = np.array(
            [CATEGORY_CONFIG[c]["weight"] for c in self._categories]
        )
        self._category_weights /= self._category_weights.sum()

    def generate(self) -> pd.DataFrame:
        """Generate full sales dataset.

        Returns:
            DataFrame with columns: sale_id, date, region, product_id,
            product_category, customer_id, quantity, unit_price, discount,
            revenue, cost, channel, priority.
        """
        logger.info("Generating %d rows of synthetic sales data", self.n_rows)

        dates = self._generate_dates()
        quantities = self._generate_seasonal_quantities(dates)
        categories = self._assign_categories()
        unit_prices = self._generate_prices(categories)
        discounts = self._generate_discounts()

        revenue = quantities * unit_prices * (1.0 - discounts)
        margins = np.array([CATEGORY_CONFIG[c]["margin"] for c in categories])
        cost = revenue * margins

        df = pd.DataFrame(
            {
                "sale_id": np.arange(1, self.n_rows + 1, dtype=np.int64),
                "date": dates,
                "region": np.random.choice(REGIONS, size=self.n_rows),
                "product_id": [
                    f"P{np.random.randint(1, 501):03d}" for _ in range(self.n_rows)
                ],
                "product_category": categories,
                "customer_id": [
                    f"C{np.random.randint(1, 50001):05d}" for _ in range(self.n_rows)
                ],
                "quantity": quantities,
                "unit_price": np.round(unit_prices, 2),
                "discount": np.round(discounts, 4),
                "revenue": np.round(revenue, 2),
                "cost": np.round(cost, 2),
                "channel": np.random.choice(CHANNELS, size=self.n_rows),
                "priority": np.random.choice(
                    PRIORITIES, size=self.n_rows, p=[0.2, 0.5, 0.3]
                ),
            }
        )

        logger.info(
            "Generated DataFrame: %d rows, %d columns, memory=%.1f MB",
            len(df),
            len(df.columns),
            df.memory_usage(deep=True).sum() / 1e6,
        )
        return df

    def _generate_dates(self) -> np.ndarray:
        """Generate dates spanning 2019-2024 with seasonal patterns.

        Q4 (Oct-Dec) and summer (Jun-Aug) receive higher probability weights
        to simulate holiday and seasonal demand spikes.

        Returns:
            Array of datetime64 values.
        """
        start = np.datetime64("2019-01-01")
        end = np.datetime64("2024-12-31")
        total_days = (end - start).astype("timedelta64[D]").astype(int) + 1

        day_weights = np.ones(total_days, dtype=np.float64)
        for day_offset in range(total_days):
            current_date = start + np.timedelta64(day_offset, "D")
            month = int(current_date.astype("datetime64[M]").astype(int) % 12) + 1
            if month in (6, 7, 8):
                day_weights[day_offset] = 1.3
            elif month in (10, 11, 12):
                day_weights[day_offset] = 1.8
            elif month in (1, 2):
                day_weights[day_offset] = 0.7

        day_weights /= day_weights.sum()
        day_indices = np.random.choice(total_days, size=self.n_rows, p=day_weights)
        return start + day_indices.astype("timedelta64[D]")

    def _generate_seasonal_quantities(self, dates: np.ndarray) -> np.ndarray:
        """Generate quantities with seasonal spikes.

        Base quantity is drawn from a log-normal distribution. Months in Q4
        and summer receive a multiplicative boost.

        Args:
            dates: Array of sale dates.

        Returns:
            Integer quantity array clipped to [1, 100].
        """
        base = np.random.lognormal(mean=2.5, sigma=0.8, size=self.n_rows)
        months = dates.astype("datetime64[M]").astype(int) % 12 + 1

        seasonal_mult = np.ones(self.n_rows, dtype=np.float64)
        seasonal_mult[np.isin(months, [6, 7, 8])] = 1.4
        seasonal_mult[np.isin(months, [10, 11, 12])] = 1.7
        seasonal_mult[np.isin(months, [1, 2])] = 0.8

        quantities = (base * seasonal_mult).astype(int)
        return np.clip(quantities, 1, 100)

    def _assign_categories(self) -> np.ndarray:
        """Assign product categories with configured weights.

        Returns:
            Array of category strings.
        """
        indices = np.random.choice(
            len(self._categories), size=self.n_rows, p=self._category_weights
        )
        return np.array(self._categories)[indices]

    def _generate_prices(self, categories: np.ndarray) -> np.ndarray:
        """Generate unit prices drawn from category-specific uniform ranges.

        Args:
            categories: Array of category labels.

        Returns:
            Float price array.
        """
        prices = np.empty(self.n_rows, dtype=np.float64)
        for cat in self._categories:
            mask = categories == cat
            lo, hi = CATEGORY_CONFIG[cat]["price_range"]
            prices[mask] = np.random.uniform(lo, hi, size=mask.sum())
        return prices

    def _generate_discounts(self) -> np.ndarray:
        """Generate discount fractions.

        Discounts follow a beta distribution skewed toward zero (most sales
        have small or no discounts).

        Returns:
            Float discount array in [0, 0.3].
        """
        raw = np.random.beta(a=2.0, b=8.0, size=self.n_rows)
        return np.clip(raw, 0.0, 0.3)
