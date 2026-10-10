"""Congestion profile aggregation and fallback calibration."""

import sqlite3
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from jro.logging_setup import logger


class CongestionProfile(NamedTuple):
    """Speed adjustment factors per road class during peak hours."""

    speed_ratio_by_class: dict[str, float]
    bus_speed_factor: float
    motorcycle_speed_factor: float
    source: str


LITERATURE_PEAK_SPEED_RATIOS: dict[str, float] = {
    "trunk": 0.55,
    "primary": 0.45,
    "secondary": 0.40,
    "tertiary": 0.48,
    "unclassified": 0.55,
    "residential": 0.60,
}


def build_congestion_profile(
    db_path: Path | None = None,
) -> CongestionProfile:
    """Computes median speed ratios from database or literature baseline.

    Args:
        db_path: Path to SQLite observations database.

    Returns:
        CongestionProfile instance.
    """
    db_file = db_path or Path("data/processed/congestion_budget.db")
    ratios = LITERATURE_PEAK_SPEED_RATIOS.copy()
    source = "literature_jakarta_study"

    if db_file.exists():
        try:
            with sqlite3.connect(db_file) as conn:
                df = pd.read_sql_query(
                    """
                    SELECT highway_class, speed_ratio
                    FROM congestion_observations
                    WHERE speed_ratio IS NOT NULL
                    """,
                    conn,
                )
                if len(df) >= 10:
                    medians = df.groupby("highway_class")["speed_ratio"].median().to_dict()
                    for hw, med in medians.items():
                        ratios[hw] = float(med)
                    source = f"tomtom_live_db_({len(df)}_samples)"
                    logger.info(
                        "Built empirical congestion profile from %d TomTom observations", len(df)
                    )
        except (sqlite3.Error, OSError) as e:
            logger.warning(
                "Could not read observations from %s: %s; using literature fallback", db_file, e
            )

    return CongestionProfile(
        speed_ratio_by_class=ratios,
        bus_speed_factor=0.85,
        motorcycle_speed_factor=1.35,
        source=source,
    )
