"""TransJakarta official GTFS feed ingestion."""

import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from jro.exceptions import DataSourceError
from jro.io.http import download_file
from jro.io.manifest import ManifestManager
from jro.logging_setup import logger

DEFAULT_TJ_GTFS_URL = "https://gtfs.transjakarta.co.id/files/file_gtfs.zip"


class TransJakartaGtfsIngestor:
    """Ingests, extracts, and caches the official TransJakarta GTFS feed."""

    def __init__(self, raw_dir: Path, manifest: ManifestManager) -> None:
        self.raw_dir = raw_dir / "gtfs_tj"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest

    def ingest(
        self,
        url: str = DEFAULT_TJ_GTFS_URL,
        force_download: bool = False,
    ) -> Path:
        """Downloads TransJakarta GTFS feed if not cached or if forced.

        Args:
            url: TransJakarta GTFS zip URL.
            force_download: If True, bypasses manifest validity check.

        Returns:
            Path to downloaded ZIP file.
        """
        zip_path = self.raw_dir / "transjakarta_gtfs.zip"
        manifest_key = "raw_transjakarta_gtfs"

        if not force_download and self.manifest.is_valid(manifest_key) and zip_path.exists():
            logger.info("TransJakarta GTFS feed is already cached and verified at %s", zip_path)
            return zip_path

        logger.info("Fetching official TransJakarta GTFS from %s", url)
        try:
            download_file(url=url, destination=zip_path)
        except Exception as e:
            # Check if an existing copy exists as fallback
            if zip_path.exists():
                logger.warning(
                    "Download failed (%s), using existing local GTFS copy at %s", e, zip_path
                )
            else:
                raise DataSourceError(f"Failed to ingest TransJakarta GTFS feed: {e}") from e

        # Validate that the file is indeed a valid ZIP file containing GTFS tables
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                namelist = set(zf.namelist())
                required = {"routes.txt", "stops.txt", "trips.txt", "stop_times.txt"}
                missing = required - namelist
                if missing:
                    raise DataSourceError(f"GTFS zip is missing required tables: {missing}")
                logger.info(
                    "GTFS zip verified valid with %d files: %s", len(namelist), sorted(namelist)
                )
        except zipfile.BadZipFile as e:
            raise DataSourceError(f"Downloaded file is not a valid zip archive: {e}") from e

        self.manifest.record(
            key=manifest_key,
            file_path=zip_path,
            source_url=url,
            license_str="CC BY 4.0",
            metadata={"ingested_at": datetime.now(UTC).isoformat()},
        )
        return zip_path

    def load_table(self, table_name: str) -> pd.DataFrame:
        """Loads a specific table from the GTFS zip archive as DataFrame."""
        zip_path = self.raw_dir / "transjakarta_gtfs.zip"
        if not zip_path.exists():
            raise DataSourceError(f"TransJakarta GTFS zip not found at {zip_path}")

        file_in_zip = f"{table_name}.txt" if not table_name.endswith(".txt") else table_name
        with zipfile.ZipFile(zip_path, "r") as zf:
            if file_in_zip not in zf.namelist():
                raise DataSourceError(f"Table {file_in_zip} not found in {zip_path}")
            with zf.open(file_in_zip) as f:
                return pd.read_csv(f, dtype=str)
