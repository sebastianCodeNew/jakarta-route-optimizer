"""Unit tests for synthetic GTFS build, merging, and validation."""

import zipfile

from jro.network.gtfs_merge import merge_gtfs_feeds
from jro.network.gtfs_validate import validate_gtfs_feed
from jro.network.rail_gtfs_builder import build_synthetic_rail_gtfs


def test_synthetic_rail_gtfs_builder_and_validation(app_config, tmp_path) -> None:
    """Tests building synthetic rail GTFS and validating its structural compliance."""
    rail_zip = tmp_path / "synthetic_rail.zip"
    build_synthetic_rail_gtfs(app_config.rail, rail_zip)

    assert rail_zip.exists()

    with zipfile.ZipFile(rail_zip, "r") as zf:
        names = set(zf.namelist())
        expected = {
            "agency.txt",
            "calendar.txt",
            "routes.txt",
            "trips.txt",
            "stops.txt",
            "stop_times.txt",
            "frequencies.txt",
        }
        assert expected.issubset(names)

    report = validate_gtfs_feed(rail_zip)
    assert report["is_valid"]
    assert report["stats"]["stops_count"] > 30
    assert report["stats"]["routes_count"] >= 5


def test_gtfs_merge(app_config, tmp_path) -> None:
    """Tests merging two GTFS feeds with namespace prefixing."""
    rail_zip = tmp_path / "rail.zip"
    build_synthetic_rail_gtfs(app_config.rail, rail_zip)

    merged_zip = tmp_path / "merged.zip"
    merge_gtfs_feeds(rail_zip, rail_zip, merged_zip)

    assert merged_zip.exists()
    report = validate_gtfs_feed(merged_zip)
    assert report["is_valid"]
