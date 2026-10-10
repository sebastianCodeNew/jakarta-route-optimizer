"""Unit tests for bus fleet operations and capacity formulas."""

from jro.candidates.generator import CandidateRoute
from jro.candidates.stop_placement import RouteStop
from jro.config import VehicleProfile
from jro.evaluation.operations import compute_route_operations


def test_compute_route_operations_scaling() -> None:
    """Verifies operational formulas for 1 bus versus 10 buses."""
    stops = [
        RouteStop(f"S{i}", f"Stop {i}", -6.2 + i * 0.01, 106.8 + i * 0.01, i, f"cell_{i}")
        for i in range(1, 11)
    ]
    route = CandidateRoute(
        route_id="TEST_R01",
        route_name="Test Route",
        vehicle_profile="non_brt",
        length_km=15.0,
        origin_name="Terminus A",
        destination_name="Terminus B",
        stops=stops,
        waypoints=[(-6.2, 106.8), (-6.1, 106.9)],
        stop_cell_ids=[s.cell_id for s in stops],
        overlap_ratio=0.10,
    )
    profile = VehicleProfile(
        name="non_brt",
        display_name="Non-BRT Bus",
        capacity=60,
        dwell_seconds=30.0,
        layover_minutes=8.0,
        allowed_highway=("primary",),
        min_length_km=10.0,
        max_length_km=35.0,
        operating_cost_per_km_idr=14000.0,
        emission_factor_key="bus_medium_large",
    )

    # N = 1 unit scenario
    ops_1 = compute_route_operations(route, profile, fleet_size=1, congested_road_speed_kmh=20.0)
    assert ops_1.fleet_size == 1
    assert ops_1.headway_minutes == ops_1.cycle_time_minutes
    assert ops_1.wait_time_minutes == 20.0  # capped by commuter wait threshold
    assert ops_1.hourly_capacity_passengers < 50.0  # severely constrained

    # N = 10 units scenario
    ops_10 = compute_route_operations(route, profile, fleet_size=10, congested_road_speed_kmh=20.0)
    assert ops_10.fleet_size == 10
    assert abs(ops_10.headway_minutes - (ops_1.cycle_time_minutes / 10.0)) < 0.1
    assert ops_10.hourly_capacity_passengers > ops_1.hourly_capacity_passengers * 9.0
