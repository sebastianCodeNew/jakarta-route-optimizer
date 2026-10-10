"""Baseline multi-modal travel time matrices calculation and caching."""

from pathlib import Path
from typing import NamedTuple

import numpy as np

from jro.congestion.profile import CongestionProfile
from jro.io.storage import load_numpy, save_numpy
from jro.logging_setup import logger


class BaselineMatrices(NamedTuple):
    """Container for baseline network travel time matrices (minutes)."""

    transit_travel_time: np.ndarray
    car_travel_time: np.ndarray
    motorcycle_travel_time: np.ndarray
    distance_km: np.ndarray


def compute_baseline_matrices(
    cell_coords: list[tuple[float, float]],
    transit_stops: list[tuple[float, float, str]],
    congestion_profile: CongestionProfile,
    cache_dir: Path | None = None,
    force_recompute: bool = False,
) -> BaselineMatrices:
    """Computes or loads cached baseline travel time matrices for study area cells.

    Args:
        cell_coords: List of (lat, lon) coordinates for each H3 cell centroid.
        transit_stops: List of (lat, lon, stop_id) for existing transit network stops.
        congestion_profile: Peak congestion speed profile.
        cache_dir: Optional directory to store/load cached matrices.
        force_recompute: Whether to bypass cache and force recomputation.

    Returns:
        BaselineMatrices containing transit, car, motorcycle times, and distances.
    """
    n = len(cell_coords)
    logger.info("Computing baseline travel time matrices for %d cells", n)

    if cache_dir and not force_recompute:
        tr_path = cache_dir / "base_transit_tt.npy"
        car_path = cache_dir / "base_car_tt.npy"
        mc_path = cache_dir / "base_mc_tt.npy"
        dist_path = cache_dir / "base_dist_km.npy"

        if tr_path.exists() and car_path.exists() and mc_path.exists() and dist_path.exists():
            logger.info("Loading cached baseline matrices from %s", cache_dir)
            return BaselineMatrices(
                transit_travel_time=load_numpy(tr_path),
                car_travel_time=load_numpy(car_path),
                motorcycle_travel_time=load_numpy(mc_path),
                distance_km=load_numpy(dist_path),
            )

    lats = np.array([c[0] for c in cell_coords], dtype=np.float32)
    lons = np.array([c[1] for c in cell_coords], dtype=np.float32)

    # 1. Distance matrix (km) via vectorized Haversine approximation in local projection
    lat_diff = (lats[:, np.newaxis] - lats[np.newaxis, :]) * 111.0
    lon_diff = (lons[:, np.newaxis] - lons[np.newaxis, :]) * 110.5
    dist_km = np.sqrt(lat_diff**2 + lon_diff**2, dtype=np.float32)
    del lat_diff, lon_diff
    # Minimum self-distance within hexagon cell
    np.fill_diagonal(dist_km, 0.5)

    # 2. Road travel speeds
    avg_speed_ratio = congestion_profile.speed_ratio_by_class.get("primary", 0.45)
    car_speed_kmh = max(12.0, 38.0 * avg_speed_ratio)
    mc_speed_kmh = min(40.0, car_speed_kmh * congestion_profile.motorcycle_speed_factor)

    car_tt = (dist_km / car_speed_kmh * 60.0).astype(np.float32)
    mc_tt = (dist_km / mc_speed_kmh * 60.0).astype(np.float32)

    # 3. Transit baseline travel time:
    if transit_stops:
        stop_lats = np.array([s[0] for s in transit_stops], dtype=np.float32)
        stop_lons = np.array([s[1] for s in transit_stops], dtype=np.float32)

        d_lat_stop = (lats[:, np.newaxis] - stop_lats[np.newaxis, :]) * 111.0
        d_lon_stop = (lons[:, np.newaxis] - stop_lons[np.newaxis, :]) * 110.5
        d_to_stops = np.sqrt(d_lat_stop**2 + d_lon_stop**2, dtype=np.float32)
        del d_lat_stop, d_lon_stop
        min_access_km = np.min(d_to_stops, axis=1)  # shape (N,)
        del d_to_stops
    else:
        min_access_km = np.full(n, 2.5, dtype=np.float32)

    walk_access_min = (min_access_km * (60.0 / 4.5)).astype(np.float32)

    transit_speed_kmh = 22.0
    in_transit_min = (dist_km / transit_speed_kmh * 60.0).astype(np.float32)
    transit_tt = (
        walk_access_min[:, np.newaxis] + in_transit_min + walk_access_min[np.newaxis, :] + 8.0
    ).astype(np.float32)
    del in_transit_min

    # Cap self trips
    np.fill_diagonal(transit_tt, 5.0)

    matrices = BaselineMatrices(
        transit_travel_time=transit_tt,
        car_travel_time=car_tt,
        motorcycle_travel_time=mc_tt,
        distance_km=dist_km,
    )

    if cache_dir:
        save_numpy(matrices.transit_travel_time, cache_dir / "base_transit_tt.npy")
        save_numpy(matrices.car_travel_time, cache_dir / "base_car_tt.npy")
        save_numpy(matrices.motorcycle_travel_time, cache_dir / "base_mc_tt.npy")
        save_numpy(matrices.distance_km, cache_dir / "base_dist_km.npy")

    logger.info("Baseline travel time matrices generated successfully")
    return matrices
