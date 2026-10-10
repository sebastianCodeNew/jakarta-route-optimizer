"""Stop placement and spatial snapping along corridor trajectories."""

from typing import NamedTuple

import h3

from jro.network.road_graph import haversine_distance_km


class RouteStop(NamedTuple):
    """Bus stop location along candidate route."""

    stop_id: str
    stop_name: str
    latitude: float
    longitude: float
    sequence: int
    cell_id: str


def place_stops_along_path(
    waypoints: list[tuple[float, float]],
    route_code: str,
    target_spacing_km: float = 0.50,
    h3_resolution: int = 8,
) -> list[RouteStop]:
    """Places evenly spaced stops along a sequence of waypoints.

    Args:
        waypoints: List of (lat, lon) coordinates defining the route path.
        route_code: Unique prefix for stop identification.
        target_spacing_km: Desired distance between consecutive stops (e.g. 0.5 km = 500 m).
        h3_resolution: H3 resolution for cell index assignment.

    Returns:
        List of RouteStop objects.
    """
    if len(waypoints) < 2:
        return []

    stops: list[RouteStop] = []

    # First stop at origin
    orig_lat, orig_lon = waypoints[0]
    orig_cell = h3.latlng_to_cell(orig_lat, orig_lon, h3_resolution)
    stops.append(
        RouteStop(
            stop_id=f"{route_code}_S01",
            stop_name=f"Halte {route_code} Terminus 1",
            latitude=orig_lat,
            longitude=orig_lon,
            sequence=1,
            cell_id=orig_cell,
        )
    )

    # Accumulate distance along segments and interpolate intermediate stops
    accumulated_dist = 0.0
    seq = 2

    for i in range(len(waypoints) - 1):
        lat1, lon1 = waypoints[i]
        lat2, lon2 = waypoints[i + 1]
        seg_dist = haversine_distance_km(lat1, lon1, lat2, lon2)

        if seg_dist <= 0:
            continue

        dist_left = seg_dist
        curr_ratio = 0.0

        while (accumulated_dist + dist_left) >= target_spacing_km:
            needed = target_spacing_km - accumulated_dist
            ratio_step = needed / seg_dist
            curr_ratio += ratio_step

            interp_lat = lat1 + curr_ratio * (lat2 - lat1)
            interp_lon = lon1 + curr_ratio * (lon2 - lon1)
            cell = h3.latlng_to_cell(interp_lat, interp_lon, h3_resolution)

            stops.append(
                RouteStop(
                    stop_id=f"{route_code}_S{seq:02d}",
                    stop_name=f"Halte {route_code} #{seq}",
                    latitude=interp_lat,
                    longitude=interp_lon,
                    sequence=seq,
                    cell_id=cell,
                )
            )
            seq += 1
            dist_left -= needed
            accumulated_dist = 0.0

        accumulated_dist += dist_left

    # Last stop at destination terminus
    dest_lat, dest_lon = waypoints[-1]
    dest_cell = h3.latlng_to_cell(dest_lat, dest_lon, h3_resolution)
    stops.append(
        RouteStop(
            stop_id=f"{route_code}_S{seq:02d}",
            stop_name=f"Halte {route_code} Terminus 2",
            latitude=dest_lat,
            longitude=dest_lon,
            sequence=seq,
            cell_id=dest_cell,
        )
    )

    return stops
