"""Interactive Folium map generation for route recommendations."""

from pathlib import Path

import folium

from jro.candidates.generator import CandidateRoute
from jro.logging_setup import logger


def generate_interactive_route_map(
    route: CandidateRoute,
    output_html_path: Path,
    title: str = "JRO Recommended Bus Route",
) -> Path:
    """Renders interactive Leaflet/Folium map with route alignment and stops.

    Args:
        route: CandidateRoute instance.
        output_html_path: Path where output HTML map will be saved.
        title: Map title string.

    Returns:
        Path to saved HTML map.
    """
    logger.info("Generating interactive Folium map for %s...", route.route_name)
    output_html_path = output_html_path.resolve()
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    # Center map on route mid-point
    mid_idx = len(route.waypoints) // 2
    center_lat, center_lon = route.waypoints[mid_idx]

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="OpenStreetMap",
    )

    # Route polyline
    route_coords = [[wp[0], wp[1]] for wp in route.waypoints]
    color = "#0066cc" if route.vehicle_profile == "non_brt" else "#e67e22"

    folium.PolyLine(
        locations=route_coords,
        color=color,
        weight=6,
        opacity=0.85,
        tooltip=f"<b>{route.route_name}</b><br>Length: {route.length_km} km ({route.vehicle_profile.upper()})",
    ).add_to(m)

    # Terminus 1 (Origin)
    folium.Marker(
        location=[route.waypoints[0][0], route.waypoints[0][1]],
        popup=f"<b>Terminus 1:</b> {route.origin_name}",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(m)

    # Terminus 2 (Destination / Transit Hub)
    folium.Marker(
        location=[route.waypoints[-1][0], route.waypoints[-1][1]],
        popup=f"<b>Terminus 2:</b> {route.destination_name} (Transit Hub)",
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
    ).add_to(m)

    # Stops along route
    for stop in route.stops[1:-1]:
        folium.CircleMarker(
            location=[stop.latitude, stop.longitude],
            radius=5,
            color=color,
            fill=True,
            fill_color="#ffffff",
            fill_opacity=0.9,
            popup=f"<b>Stop #{stop.sequence}:</b> {stop.stop_name}<br>H3 Cell: {stop.cell_id}",
        ).add_to(m)

    m.save(str(output_html_path))
    logger.info("Interactive map saved at %s", output_html_path)
    return output_html_path
