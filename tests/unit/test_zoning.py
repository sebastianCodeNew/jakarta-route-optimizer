"""Unit tests for H3 spatial indexing and zoning."""

from jro.config import BBox
from jro.zoning.h3_grid import (
    REGENCY_CENTERS_DATA,
    assign_regency_by_proximity,
    generate_study_area_grid,
)


def test_h3_grid_generation_mini_bbox() -> None:
    """Tests H3 grid generation over a small test bounding box."""
    mini_bbox = BBox(min_lon=106.80, min_lat=-6.22, max_lon=106.84, max_lat=-6.18)
    gdf = generate_study_area_grid(mini_bbox, resolution=8)

    assert not gdf.empty
    assert "cell_id" in gdf.columns
    assert "geometry" in gdf.columns
    assert all(geom.is_valid for geom in gdf.geometry)


def test_regency_assignment() -> None:
    """Tests assigning regencies based on proximity."""
    mini_bbox = BBox(min_lon=106.80, min_lat=-6.22, max_lon=106.84, max_lat=-6.18)
    gdf = generate_study_area_grid(mini_bbox, resolution=8)

    assigned = assign_regency_by_proximity(gdf, REGENCY_CENTERS_DATA)
    assert "kabkota_code" in assigned.columns
    assert "kabkota_name" in assigned.columns
    assert assigned["kabkota_code"].notna().all()
