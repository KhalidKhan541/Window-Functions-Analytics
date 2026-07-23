# Window Functions & Advanced Analytics

Production-grade implementation of SQL-style window functions in Python/pandas for large-scale analytics. Processes 3M+ row sales datasets with vectorized operations, a pandas validation layer, and CTE simulation.

## Architecture

```
window_functions/
├── run.py                    # CLI entry point (generate / analyze / validate / full)
├── src/
│   ├── window_funcs.py       # 20 SQL window functions (ROW_NUMBER, LAG/LEAD, NTILE, etc.)
│   ├── ctes.py               # CTE simulation with method chaining
│   ├── analysis.py           # Sales analytics combining window functions
│   ├── data_generator.py     # 3M+ row synthetic sales data generator
│   ├── validation.py         # DataFrame validation layer
│   └── pipeline.py           # End-to-end orchestration
├── configs/
│   └── default.yaml          # Default configuration
├── data/                     # Generated datasets
└── outputs/                  # Analysis results
```

## Window Functions Implemented

| Function | SQL Equivalent | Description |
|----------|---------------|-------------|
| `row_number()` | `ROW_NUMBER() OVER (...)` | Sequential row numbering |
| `rank()` | `RANK() OVER (...)` | Ranking with gaps |
| `dense_rank()` | `DENSE_RANK() OVER (...)` | Ranking without gaps |
| `ntile()` | `NTILE(n) OVER (...)` | Divide into n equal buckets |
| `lag()` | `LAG(col, n) OVER (...)` | Value from n rows before |
| `lead()` | `LEAD(col, n) OVER (...)` | Value from n rows after |
| `running_total()` | `SUM() OVER (... ROWS UNBOUNDED PRECEDING)` | Cumulative sum |
| `moving_average()` | `AVG() OVER (... ROWS BETWEEN N PRECEDING AND CURRENT)` | Rolling average |
| `percent_of_total()` | `col / SUM() OVER (PARTITION BY ...)` | Percentage of partition total |
| `percentile_rank()` | `PERCENT_RANK() OVER (...)` | Percentile position |
| `cume_dist()` | `CUME_DIST() OVER (...)` | Cumulative distribution |
| `yoy_growth()` | LAG-based YoY calculation | Year-over-year growth % |
| `period_over_period()` | LAG-based comparison | Period-over-period change % |
| `cumulative_avg()` | Expanding window mean | Running average |
| `cumulative_count()` | Expanding window count | Running count |

## Quick Start

```bash
pip install -r requirements.txt

# Generate 3M+ rows and run full analysis
python -m window_functions.run full --n-rows 3000000 --output outputs/

# Generate only
python -m window_functions.run generate --n-rows 1000000 --output data/

# Analyze existing data
python -m window_functions.run analyze --input data/sales_data.csv --output outputs/

# Validate data
python -m window_functions.run validate --input data/sales_data.csv
```

## Output Files

| File | Description |
|------|-------------|
| `raw_data.csv` | 3M+ row synthetic sales dataset |
| `enriched_data.csv` | Original data + all window function columns |
| `analysis_summary.csv` | Aggregated product-level metrics |
| `top_products.csv` | Top-N products by revenue (dense_rank) |

## Features

- **Vectorized Operations**: All window functions use `groupby().transform()`, `shift()`, `expanding()` — no row-by-row apply
- **3M+ Rows**: Handles millions of rows efficiently with numpy vectorized generation
- **Validation Layer**: Schema checks, null detection, uniqueness, range validation, referential integrity
- **CTE Simulation**: SQL-style CTEs with pandas method chaining
- **Seasonal Patterns**: Realistic synthetic data with Q4 spikes, category-specific pricing

## Dependencies

- numpy
- pandas
- pyyaml
