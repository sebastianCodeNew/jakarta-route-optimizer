"""Trip generation (commuter production) modeling per H3 cell."""

import pandas as pd

from jro.exceptions import DemandModelError
from jro.logging_setup import logger


def compute_trip_production(
    cells_df: pd.DataFrame,
    labor_df: pd.DataFrame,
) -> pd.DataFrame:
    """Computes daily commuter trip production (origins O_i) per H3 cell.

    Formula:
        O_i = Employed_i * CommuterRate(kabkota_i)

    Args:
        cells_df: DataFrame with ['cell_id', 'employed', 'kabkota_code'].
        labor_df: DataFrame with BPS labor statistics ['kabkota_code', 'commuter_rate'].

    Returns:
        DataFrame with columns: ['cell_id', 'production']
    """
    logger.info("Computing commuter trip production for %d cells", len(cells_df))

    merged = cells_df.merge(
        labor_df[["kabkota_code", "commuter_rate"]],
        on="kabkota_code",
        how="left",
    )

    if merged["commuter_rate"].isna().any():
        logger.warning("Some cells have missing commuter rates; filling with average rate 0.18")
        merged["commuter_rate"] = merged["commuter_rate"].fillna(0.18)

    merged["production"] = (merged["employed"] * merged["commuter_rate"]).clip(lower=0.0)

    total_trips = merged["production"].sum()
    if total_trips <= 0:
        raise DemandModelError("Total trip production is zero or negative")

    logger.info("Total daily commute trip production: %s trips/day", f"{total_trips:,.0f}")
    return merged[["cell_id", "production"]]
