"""Loader and validator for BPS labor, commuter, and inter-regency OD datasets."""

from pathlib import Path

import pandas as pd

from jro.exceptions import DataSourceError
from jro.logging_setup import logger

REQUIRED_LABOR_COLUMNS = {
    "kabkota_code",
    "kabkota_name",
    "province",
    "population_total",
    "population_15plus",
    "employed_count",
    "commuter_rate",
    "transit_share_reported",
}

REQUIRED_OD_COLUMNS = {
    "origin_code",
    "origin_name",
    "destination_code",
    "destination_name",
    "commuter_flow_daily",
}


class BpsDataLoader:
    """Loads and validates official BPS Jabodetabek datasets."""

    def __init__(self, raw_dir: Path) -> None:
        self.bps_dir = raw_dir / "bps"

    def load_labor_data(self) -> pd.DataFrame:
        """Loads and validates regency/city labor and commuter rate data."""
        file_path = self.bps_dir / "bps_kabkota_labor.csv"
        if not file_path.exists():
            raise DataSourceError(f"BPS labor file not found: {file_path}")

        df = pd.read_csv(file_path)
        missing = REQUIRED_LABOR_COLUMNS - set(df.columns)
        if missing:
            raise DataSourceError(f"BPS labor file is missing required columns: {missing}")

        logger.info("Loaded BPS labor statistics for %d regencies/cities", len(df))
        return df

    def load_od_matrix(self) -> pd.DataFrame:
        """Loads and validates the inter-regency BPS commuter flow matrix."""
        file_path = self.bps_dir / "bps_commuter_od_2023.csv"
        if not file_path.exists():
            raise DataSourceError(f"BPS commuter OD file not found: {file_path}")

        df = pd.read_csv(file_path)
        missing = REQUIRED_OD_COLUMNS - set(df.columns)
        if missing:
            raise DataSourceError(f"BPS commuter OD file is missing required columns: {missing}")

        logger.info(
            "Loaded BPS commuter OD flows (%d OD pairs, total %s trips/day)",
            len(df),
            f"{df['commuter_flow_daily'].sum():,}",
        )
        return df
