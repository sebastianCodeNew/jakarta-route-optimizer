"""Unit tests for vectorized min-plus evaluator and capacity constraints."""

import numpy as np

from jro.candidates.generator import CandidateRoute
from jro.candidates.stop_placement import RouteStop
from jro.evaluation.fast_eval import evaluate_route_fast
from jro.evaluation.operations import RouteOperations


def test_fast_eval_monotonicity_and_capacity_cap(sample_od_matrices, mode_choice_model) -> None:
    """Tests that new transit route never worsens travel time and respects daily capacity."""
    base_transit_tt, base_mc_tt, base_car_tt, dist_km, od_demand = sample_od_matrices

    cell_ids = [f"cell_{i}" for i in range(5)]
    cell_id_to_idx = {cid: idx for idx, cid in enumerate(cell_ids)}

    stops = [
        RouteStop(f"S{i}", f"Stop {i}", -6.2 + i * 0.01, 106.8 + i * 0.01, i + 1, cell_ids[i])
        for i in range(5)
    ]
    route = CandidateRoute(
        route_id="TEST_CORR",
        route_name="Test Corridor",
        vehicle_profile="mikrotrans",
        length_km=15.0,
        origin_name="Stop 0",
        destination_name="Stop 4",
        stops=stops,
        waypoints=[(-6.2, 106.8), (-6.16, 106.84)],
        stop_cell_ids=cell_ids,
        overlap_ratio=0.15,
    )

    ops = RouteOperations(
        fleet_size=8,
        cycle_time_minutes=60.0,
        headway_minutes=7.5,
        wait_time_minutes=3.75,
        hourly_capacity_passengers=88.0,
        daily_capacity_passengers=924.0,  # constrained daily cap
        commercial_speed_kmh=22.0,
        daily_bus_vkt=384.0,
        daily_operating_cost_idr=2496000.0,
    )

    eval_res = evaluate_route_fast(
        route=route,
        operations=ops,
        base_transit_tt=base_transit_tt,
        base_mc_tt=base_mc_tt,
        base_car_tt=base_car_tt,
        dist_km=dist_km,
        od_demand=od_demand,
        cell_id_to_idx=cell_id_to_idx,
        mode_choice=mode_choice_model,
    )

    # 1. Monotonicity: new travel time must be <= baseline travel time everywhere
    assert np.all(eval_res.new_transit_travel_time <= base_transit_tt + 1e-6)

    # 2. Capacity constraint: served riders must not exceed daily capacity
    assert eval_res.daily_riders_served <= ops.daily_capacity_passengers + 1e-4

    # 3. Mode shifts are non-negative
    assert eval_res.shifted_motorcycle_trips >= 0.0
    assert eval_res.shifted_car_trips >= 0.0
