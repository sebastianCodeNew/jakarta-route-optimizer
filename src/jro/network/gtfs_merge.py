"""Merges TransJakarta GTFS and Synthetic Rail GTFS into a unified network feed."""

import io
import zipfile
from pathlib import Path

import pandas as pd

from jro.logging_setup import logger


def merge_gtfs_feeds(
    tj_gtfs_path: Path,
    rail_gtfs_path: Path,
    output_path: Path,
    transfer_radius_meters: float = 250.0,
) -> Path:
    """Merges TransJakarta and Rail GTFS feeds with namespacing.

    Args:
        tj_gtfs_path: Path to TransJakarta GTFS zip.
        rail_gtfs_path: Path to synthetic rail GTFS zip.
        output_path: Destination path for merged GTFS zip.
        transfer_radius_meters: Walk distance threshold for generating transfer links.

    Returns:
        Path to output merged GTFS zip.
    """
    logger.info(
        "Merging GTFS feeds: %s + %s -> %s", tj_gtfs_path.name, rail_gtfs_path.name, output_path
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def read_zip_csv(zip_path: Path, filename: str) -> pd.DataFrame:
        with zipfile.ZipFile(zip_path, "r") as zf:
            if filename not in zf.namelist():
                return pd.DataFrame()
            with zf.open(filename) as f:
                return pd.read_csv(f, dtype=str)

    # 1. Agency
    tj_agency = read_zip_csv(tj_gtfs_path, "agency.txt")
    rail_agency = read_zip_csv(rail_gtfs_path, "agency.txt")
    if not tj_agency.empty and "agency_id" in tj_agency.columns:
        tj_agency["agency_id"] = "tj:" + tj_agency["agency_id"].astype(str)
    if not rail_agency.empty and "agency_id" in rail_agency.columns:
        rail_agency["agency_id"] = "rail:" + rail_agency["agency_id"].astype(str)
    merged_agency = pd.concat([tj_agency, rail_agency], ignore_index=True).drop_duplicates()

    # 2. Stops
    tj_stops = read_zip_csv(tj_gtfs_path, "stops.txt")
    rail_stops = read_zip_csv(rail_gtfs_path, "stops.txt")
    if not tj_stops.empty:
        tj_stops["stop_id"] = "tj:" + tj_stops["stop_id"].astype(str)
    if not rail_stops.empty:
        rail_stops["stop_id"] = "rail:" + rail_stops["stop_id"].astype(str)
    merged_stops = pd.concat([tj_stops, rail_stops], ignore_index=True).drop_duplicates(
        subset=["stop_id"]
    )

    # 3. Routes
    tj_routes = read_zip_csv(tj_gtfs_path, "routes.txt")
    rail_routes = read_zip_csv(rail_gtfs_path, "routes.txt")
    if not tj_routes.empty:
        tj_routes["route_id"] = "tj:" + tj_routes["route_id"].astype(str)
        if "agency_id" in tj_routes.columns:
            tj_routes["agency_id"] = "tj:" + tj_routes["agency_id"].astype(str)
    if not rail_routes.empty:
        rail_routes["route_id"] = "rail:" + rail_routes["route_id"].astype(str)
        if "agency_id" in rail_routes.columns:
            rail_routes["agency_id"] = "rail:" + rail_routes["agency_id"].astype(str)
    merged_routes = pd.concat([tj_routes, rail_routes], ignore_index=True).drop_duplicates(
        subset=["route_id"]
    )

    # 4. Calendar
    tj_calendar = read_zip_csv(tj_gtfs_path, "calendar.txt")
    rail_calendar = read_zip_csv(rail_gtfs_path, "calendar.txt")
    if not tj_calendar.empty:
        tj_calendar["service_id"] = "tj:" + tj_calendar["service_id"].astype(str)
    if not rail_calendar.empty:
        rail_calendar["service_id"] = "rail:" + rail_calendar["service_id"].astype(str)
    merged_calendar = pd.concat([tj_calendar, rail_calendar], ignore_index=True).drop_duplicates(
        subset=["service_id"]
    )

    # 5. Trips
    tj_trips = read_zip_csv(tj_gtfs_path, "trips.txt")
    rail_trips = read_zip_csv(rail_gtfs_path, "trips.txt")
    if not tj_trips.empty:
        tj_trips["trip_id"] = "tj:" + tj_trips["trip_id"].astype(str)
        tj_trips["route_id"] = "tj:" + tj_trips["route_id"].astype(str)
        tj_trips["service_id"] = "tj:" + tj_trips["service_id"].astype(str)
    if not rail_trips.empty:
        rail_trips["trip_id"] = "rail:" + rail_trips["trip_id"].astype(str)
        rail_trips["route_id"] = "rail:" + rail_trips["route_id"].astype(str)
        rail_trips["service_id"] = "rail:" + rail_trips["service_id"].astype(str)
    merged_trips = pd.concat([tj_trips, rail_trips], ignore_index=True).drop_duplicates(
        subset=["trip_id"]
    )

    # 6. Stop times
    tj_stoptimes = read_zip_csv(tj_gtfs_path, "stop_times.txt")
    rail_stoptimes = read_zip_csv(rail_gtfs_path, "stop_times.txt")
    if not tj_stoptimes.empty:
        tj_stoptimes["trip_id"] = "tj:" + tj_stoptimes["trip_id"].astype(str)
        tj_stoptimes["stop_id"] = "tj:" + tj_stoptimes["stop_id"].astype(str)
    if not rail_stoptimes.empty:
        rail_stoptimes["trip_id"] = "rail:" + rail_stoptimes["trip_id"].astype(str)
        rail_stoptimes["stop_id"] = "rail:" + rail_stoptimes["stop_id"].astype(str)
    merged_stoptimes = pd.concat([tj_stoptimes, rail_stoptimes], ignore_index=True)

    # 7. Frequencies
    tj_frequencies = read_zip_csv(tj_gtfs_path, "frequencies.txt")
    rail_frequencies = read_zip_csv(rail_gtfs_path, "frequencies.txt")
    if not tj_frequencies.empty:
        tj_frequencies["trip_id"] = "tj:" + tj_frequencies["trip_id"].astype(str)
    if not rail_frequencies.empty:
        rail_frequencies["trip_id"] = "rail:" + rail_frequencies["trip_id"].astype(str)
    merged_frequencies = pd.concat([tj_frequencies, rail_frequencies], ignore_index=True)

    # Pack into output zip
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, df in [
            ("agency.txt", merged_agency),
            ("stops.txt", merged_stops),
            ("routes.txt", merged_routes),
            ("calendar.txt", merged_calendar),
            ("trips.txt", merged_trips),
            ("stop_times.txt", merged_stoptimes),
            ("frequencies.txt", merged_frequencies),
        ]:
            if not df.empty:
                buf = io.StringIO()
                df.to_csv(buf, index=False)
                zf.writestr(name, buf.getvalue())

    logger.info(
        "GTFS merge complete: %d routes, %d stops, %d trips",
        len(merged_routes),
        len(merged_stops),
        len(merged_trips),
    )
    return output_path
