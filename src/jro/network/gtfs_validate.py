"""Validation utility for GTFS feeds."""

import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from jro.exceptions import ValidationError
from jro.logging_setup import logger


def validate_gtfs_feed(gtfs_path: Path) -> dict[str, Any]:
    """Validates structural and referential integrity of a GTFS feed.

    Args:
        gtfs_path: Path to GTFS zip file.

    Returns:
        Validation report dictionary with summary metrics and error list.
    """
    logger.info("Validating GTFS feed integrity for %s", gtfs_path)
    report: dict[str, Any] = {
        "file": str(gtfs_path),
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    if not gtfs_path.exists():
        raise ValidationError(f"GTFS file not found: {gtfs_path}")

    try:
        with zipfile.ZipFile(gtfs_path, "r") as zf:
            file_list = set(zf.namelist())
            required = {"agency.txt", "stops.txt", "routes.txt", "trips.txt", "stop_times.txt"}
            missing = required - file_list
            if missing:
                report["is_valid"] = False
                report["errors"].append(f"Missing required GTFS files: {sorted(missing)}")
                return report

            def load_df(name: str) -> pd.DataFrame:
                if name in file_list:
                    with zf.open(name) as f:
                        return pd.read_csv(f, dtype=str)
                return pd.DataFrame()

            stops = load_df("stops.txt")
            routes = load_df("routes.txt")
            trips = load_df("trips.txt")
            stop_times = load_df("stop_times.txt")

            report["stats"] = {
                "stops_count": len(stops),
                "routes_count": len(routes),
                "trips_count": len(trips),
                "stop_times_count": len(stop_times),
            }

            # Coordinate validation
            stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
            stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
            invalid_coords = stops[
                stops["stop_lat"].isna()
                | stops["stop_lon"].isna()
                | (stops["stop_lat"] < -8.0)
                | (stops["stop_lat"] > -5.0)
                | (stops["stop_lon"] < 105.0)
                | (stops["stop_lon"] > 108.5)
            ]
            if not invalid_coords.empty:
                report["warnings"].append(
                    f"Found {len(invalid_coords)} stops with coordinates outside Jabodetabek boundary"
                )

            # Referential integrity
            trip_routes = set(trips["route_id"])
            all_routes = set(routes["route_id"])
            orphan_trips = trip_routes - all_routes
            if orphan_trips:
                report["errors"].append(
                    f"Found {len(orphan_trips)} trips referencing non-existent routes"
                )
                report["is_valid"] = False

            st_trips = set(stop_times["trip_id"])
            all_trips = set(trips["trip_id"])
            orphan_st = st_trips - all_trips
            if orphan_st:
                report["errors"].append(
                    f"Found {len(orphan_st)} stop_times referencing non-existent trips"
                )
                report["is_valid"] = False

    except (OSError, zipfile.BadZipFile, KeyError, ValueError) as e:
        report["is_valid"] = False
        report["errors"].append(f"Unexpected error inspecting GTFS zip: {e}")

    logger.info(
        "GTFS Validation result for %s: valid=%s, %d errors, %d warnings",
        gtfs_path.name,
        report["is_valid"],
        len(report["errors"]),
        len(report["warnings"]),
    )
    return report
