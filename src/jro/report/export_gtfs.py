"""Exports recommended bus route into a compliant GTFS zip archive."""

import csv
import io
import zipfile
from pathlib import Path

from jro.candidates.generator import CandidateRoute
from jro.evaluation.operations import RouteOperations
from jro.logging_setup import logger


def export_route_to_gtfs(
    route: CandidateRoute,
    operations: RouteOperations,
    output_zip_path: Path,
) -> Path:
    """Packages candidate route into a standard GTFS zip archive.

    Args:
        route: Recommended CandidateRoute instance.
        operations: Operating specifications for the route.
        output_zip_path: Target zip file path.

    Returns:
        Path to output zip file.
    """
    logger.info("Exporting GTFS feed for route %s to %s", route.route_id, output_zip_path)
    output_zip_path = output_zip_path.resolve()
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)

    agency_rows = [
        ["agency_id", "agency_name", "agency_url", "agency_timezone", "agency_lang"],
        [
            "TRANSJAKARTA",
            "PT Transportasi Jakarta",
            "https://transjakarta.co.id",
            "Asia/Jakarta",
            "id",
        ],
    ]

    calendar_rows = [
        [
            "service_id",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "start_date",
            "end_date",
        ],
        ["DAILY_SERVICE", "1", "1", "1", "1", "1", "1", "1", "20240101", "20301231"],
    ]

    route_type = "3"  # Bus
    routes_rows = [
        ["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"],
        [route.route_id, "TRANSJAKARTA", route.route_id, route.route_name, route_type],
    ]

    trip_fwd = f"TRIP_{route.route_id}_FWD"
    trip_rev = f"TRIP_{route.route_id}_REV"
    trips_rows = [
        ["route_id", "service_id", "trip_id", "trip_headsign", "direction_id"],
        [route.route_id, "DAILY_SERVICE", trip_fwd, route.destination_name, "0"],
        [route.route_id, "DAILY_SERVICE", trip_rev, route.origin_name, "1"],
    ]

    headway_secs = max(300, int(operations.headway_minutes * 60))
    frequencies_rows = [
        ["trip_id", "start_time", "end_time", "headway_secs"],
        [trip_fwd, "05:00:00", "22:00:00", str(headway_secs)],
        [trip_rev, "05:00:00", "22:00:00", str(headway_secs)],
    ]

    stops_rows = [["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type"]]
    stop_times_rows = [["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]]

    curr_min = 300  # 05:00 AM
    for s in route.stops:
        stops_rows.append([s.stop_id, s.stop_name, f"{s.latitude:.6f}", f"{s.longitude:.6f}", "0"])
        arr_str = f"{curr_min // 60:02d}:{curr_min % 60:02d}:00"
        dep_str = f"{(curr_min + 1) // 60:02d}:{(curr_min + 1) % 60:02d}:00"
        stop_times_rows.append([trip_fwd, arr_str, dep_str, s.stop_id, str(s.sequence)])
        curr_min += 2

    # Reverse direction
    curr_min = 300
    for seq, s in enumerate(reversed(route.stops), start=1):
        arr_str = f"{curr_min // 60:02d}:{curr_min % 60:02d}:00"
        dep_str = f"{(curr_min + 1) // 60:02d}:{(curr_min + 1) % 60:02d}:00"
        stop_times_rows.append([trip_rev, arr_str, dep_str, s.stop_id, str(seq)])
        curr_min += 2

    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, rows in [
            ("agency.txt", agency_rows),
            ("calendar.txt", calendar_rows),
            ("routes.txt", routes_rows),
            ("trips.txt", trips_rows),
            ("stops.txt", stops_rows),
            ("stop_times.txt", stop_times_rows),
            ("frequencies.txt", frequencies_rows),
        ]:
            out = io.StringIO()
            writer = csv.writer(out)
            writer.writerows(rows)
            zf.writestr(name, out.getvalue())

    logger.info("GTFS export complete: %s (%d stops)", output_zip_path.name, len(route.stops))
    return output_zip_path
