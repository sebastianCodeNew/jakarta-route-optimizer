"""Loader for PPID TransJakarta and KRL ridership validation numbers."""

from pathlib import Path

import pandas as pd

from jro.exceptions import DataSourceError
from jro.logging_setup import logger

REQUIRED_RIDERSHIP_COLUMNS = {
    "route_id",
    "route_name",
    "service_type",
    "daily_ridership_reported",
    "source_ref",
}


class RidershipDataLoader:
    """Loads validation benchmarks for transit corridor passenger counts."""

    def __init__(self, raw_dir: Path) -> None:
        self.bps_dir = raw_dir / "bps"

    def load_validation_ridership(self) -> pd.DataFrame:
        """Loads officially reported ridership numbers for model validation."""
        file_path = self.bps_dir / "ridership_validation.csv"
        if not file_path.exists():
            raise DataSourceError(f"Ridership validation file not found: {file_path}")

        df = pd.read_csv(file_path)
        missing = REQUIRED_RIDERSHIP_COLUMNS - set(df.columns)
        if missing:
            raise DataSourceError(
                f"Ridership validation file is missing required columns: {missing}"
            )

        logger.info("Loaded ridership validation dataset with %d corridors/stations", len(df))
        return df
