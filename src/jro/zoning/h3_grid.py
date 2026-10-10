"""H3 spatial hexagonal indexing for Greater Jakarta (Jabodetabek)."""

from collections.abc import Sequence

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from jro.config import BBox
from jro.logging_setup import logger


def generate_study_area_grid(
    bbox: BBox,
    resolution: int = 8,
) -> gpd.GeoDataFrame:
    """Generates a complete hexagonal H3 grid covering the bounding box.

    Args:
        bbox: Bounding box of the study area.
        resolution: H3 hexagon resolution (default 8, ~0.74 km2).

    Returns:
        GeoDataFrame with columns: ['cell_id', 'latitude', 'longitude', 'geometry']
        in EPSG:4326.
    """
    logger.info(
        "Generating H3 grid at resolution %d for bbox [%.2f, %.2f, %.2f, %.2f]",
        resolution,
        bbox.min_lon,
        bbox.min_lat,
        bbox.max_lon,
        bbox.max_lat,
    )

    # Sample grid points to cover the entire bounding box
    step_lat = 0.008
    step_lon = 0.008
    lats = np.arange(bbox.min_lat, bbox.max_lat + step_lat, step_lat)
    lons = np.arange(bbox.min_lon, bbox.max_lon + step_lon, step_lon)

    cells_set: set[str] = set()
    for lat in lats:
        for lon in lons:
            cell = h3.latlng_to_cell(float(lat), float(lon), resolution)
            cells_set.add(cell)

    logger.info("Generated %d unique H3 cells", len(cells_set))
    return cells_to_geodataframe(sorted(cells_set))


def cells_to_geodataframe(cell_ids: Sequence[str]) -> gpd.GeoDataFrame:
    """Converts a sequence of H3 cell IDs into a GeoDataFrame with hexagon polygons."""
    records = []
    geometries = []

    for cell in cell_ids:
        lat, lon = h3.cell_to_latlng(cell)
        boundary = h3.cell_to_boundary(cell)
        # boundary is list of (lat, lng), shapely Polygon expects (x=lng, y=lat)
        poly = Polygon([(pt[1], pt[0]) for pt in boundary])
        records.append({"cell_id": cell, "latitude": lat, "longitude": lon})
        geometries.append(poly)

    gdf = gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")
    return gdf


def assign_regency_by_proximity(
    cells_gdf: gpd.GeoDataFrame,
    regency_centers: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Assigns each H3 cell to the nearest Jabodetabek regency/city center.

    Args:
        cells_gdf: GeoDataFrame of H3 cells with ['latitude', 'longitude'].
        regency_centers: DataFrame with ['kabkota_code', 'kabkota_name', 'latitude', 'longitude'].

    Returns:
        Augmented GeoDataFrame with ['kabkota_code', 'kabkota_name'].
    """
    reg_coords = regency_centers[["latitude", "longitude"]].to_numpy()
    cell_coords = cells_gdf[["latitude", "longitude"]].to_numpy()

    # Compute Euclidean distance in degree space (sufficient for coarse administrative zone assignment)
    # Shape: (N_cells, N_regencies)
    diff = cell_coords[:, np.newaxis, :] - reg_coords[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    nearest_idx = np.argmin(dist_sq, axis=1)

    result_gdf = cells_gdf.copy()
    result_gdf["kabkota_code"] = regency_centers["kabkota_code"].iloc[nearest_idx].values
    result_gdf["kabkota_name"] = regency_centers["kabkota_name"].iloc[nearest_idx].values
    return result_gdf


# Canonical Jabodetabek administrative center coordinates for proximity mapping
REGENCY_CENTERS_DATA = pd.DataFrame(
    [
        {
            "kabkota_code": 3171,
            "kabkota_name": "Jakarta Pusat",
            "latitude": -6.1805,
            "longitude": 106.8284,
        },
        {
            "kabkota_code": 3172,
            "kabkota_name": "Jakarta Utara",
            "latitude": -6.1384,
            "longitude": 106.8640,
        },
        {
            "kabkota_code": 3173,
            "kabkota_name": "Jakarta Barat",
            "latitude": -6.1683,
            "longitude": 106.7588,
        },
        {
            "kabkota_code": 3174,
            "kabkota_name": "Jakarta Selatan",
            "latitude": -6.2615,
            "longitude": 106.8106,
        },
        {
            "kabkota_code": 3175,
            "kabkota_name": "Jakarta Timur",
            "latitude": -6.2250,
            "longitude": 106.9004,
        },
        {
            "kabkota_code": 3201,
            "kabkota_name": "Kabupaten Bogor",
            "latitude": -6.4800,
            "longitude": 106.8400,
        },
        {
            "kabkota_code": 3271,
            "kabkota_name": "Kota Bogor",
            "latitude": -6.5971,
            "longitude": 106.8060,
        },
        {
            "kabkota_code": 3276,
            "kabkota_name": "Kota Depok",
            "latitude": -6.4025,
            "longitude": 106.7942,
        },
        {
            "kabkota_code": 3216,
            "kabkota_name": "Kabupaten Bekasi",
            "latitude": -6.2500,
            "longitude": 107.1500,
        },
        {
            "kabkota_code": 3275,
            "kabkota_name": "Kota Bekasi",
            "latitude": -6.2383,
            "longitude": 106.9756,
        },
        {
            "kabkota_code": 3603,
            "kabkota_name": "Kabupaten Tangerang",
            "latitude": -6.2000,
            "longitude": 106.4700,
        },
        {
            "kabkota_code": 3671,
            "kabkota_name": "Kota Tangerang",
            "latitude": -6.1783,
            "longitude": 106.6319,
        },
        {
            "kabkota_code": 3674,
            "kabkota_name": "Kota Tangerang Selatan",
            "latitude": -6.2889,
            "longitude": 106.7179,
        },
    ]
)
