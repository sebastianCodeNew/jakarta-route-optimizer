"""Unit tests for configuration models and validation."""

import pytest
from pydantic import ValidationError

from jro.config import BBox, ObjectiveWeights, VehicleProfile, load_config


def test_load_config_valid(config_dir) -> None:
    """Verifies that the standard configuration directory parses without error."""
    cfg = load_config(config_dir)
    assert cfg.settings.h3_resolution == 8
    assert "mikrotrans" in cfg.vehicles
    assert "non_brt" in cfg.vehicles
    assert (
        abs(
            cfg.weights.vkt_reduction
            + cfg.weights.emission_reduction
            + cfg.weights.ridership
            + cfg.weights.accessibility_gain
            + cfg.weights.equity_transit_desert
            - 1.0
        )
        < 1e-4
    )


def test_bbox_validation() -> None:
    """Tests bounding box ordering constraints."""
    # Valid
    bbox = BBox(min_lon=106.5, min_lat=-6.5, max_lon=107.0, max_lat=-6.0)
    assert bbox.min_lon < bbox.max_lon

    # Invalid: min_lon >= max_lon
    with pytest.raises(ValidationError):
        BBox(min_lon=107.5, min_lat=-6.5, max_lon=106.5, max_lat=-6.0)

    # Invalid: min_lat >= max_lat
    with pytest.raises(ValidationError):
        BBox(min_lon=106.5, min_lat=-5.5, max_lon=107.0, max_lat=-6.5)


def test_objective_weights_must_sum_to_one() -> None:
    """Tests that objective weights must sum to 1.0."""
    # Valid sum = 1.0
    w = ObjectiveWeights(
        vkt_reduction=0.30,
        emission_reduction=0.25,
        ridership=0.20,
        accessibility_gain=0.15,
        equity_transit_desert=0.10,
    )
    assert w.vkt_reduction == 0.30

    # Invalid sum != 1.0
    with pytest.raises(ValidationError):
        ObjectiveWeights(
            vkt_reduction=0.50,
            emission_reduction=0.50,
            ridership=0.20,
            accessibility_gain=0.15,
            equity_transit_desert=0.10,
        )


def test_vehicle_profile_length_bounds() -> None:
    """Tests min/max length validation on vehicle profile."""
    with pytest.raises(ValidationError):
        VehicleProfile(
            name="mikrotrans",
            display_name="Test",
            capacity=11,
            dwell_seconds=15.0,
            layover_minutes=5.0,
            allowed_highway=("primary",),
            min_length_km=25.0,  # Invalid: min > max
            max_length_km=10.0,
            operating_cost_per_km_idr=6500.0,
            emission_factor_key="mikrotrans",
        )
