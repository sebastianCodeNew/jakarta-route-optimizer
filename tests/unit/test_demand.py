"""Unit tests for travel demand: gravity, IPF, and mode choice."""

import numpy as np
import pandas as pd

from jro.demand.generation import compute_trip_production
from jro.demand.gravity import doubly_constrained_gravity
from jro.demand.ipf import apply_biproportional_ipf


def test_compute_trip_production() -> None:
    """Tests commuter trip production calculation."""
    cells_df = pd.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3"],
            "employed": [1000.0, 2500.0, 500.0],
            "kabkota_code": [3171, 3174, 3275],
        }
    )
    labor_df = pd.DataFrame(
        {
            "kabkota_code": [3171, 3174, 3275],
            "commuter_rate": [0.18, 0.20, 0.25],
        }
    )
    prod_df = compute_trip_production(cells_df, labor_df)
    assert len(prod_df) == 3
    # c1: 1000 * 0.18 = 180
    assert abs(prod_df.loc[prod_df["cell_id"] == "c1", "production"].iloc[0] - 180.0) < 1e-3


def test_doubly_constrained_gravity_conservation() -> None:
    """Verifies that doubly-constrained gravity model strictly conserves origin flows."""
    origins = np.array([500.0, 1200.0, 800.0])
    destinations = np.array([700.0, 1000.0, 800.0])
    cost_matrix = np.array(
        [
            [10.0, 25.0, 45.0],
            [20.0, 12.0, 30.0],
            [40.0, 28.0, 15.0],
        ]
    )

    flow_matrix = doubly_constrained_gravity(
        origins=origins,
        destinations=destinations,
        cost_matrix=cost_matrix,
        beta=0.04,
        max_iter=50,
    )

    # Check origin production conservation: sum_j T_ij == O_i
    row_sums = np.sum(flow_matrix, axis=1)
    assert np.allclose(row_sums, origins, rtol=1e-3)
    assert np.all(flow_matrix >= 0.0)


def test_ipf_convergence() -> None:
    """Tests that IPF bi-proportional scaling matches regional targets."""
    # 4 cells, 2 regencies (cells 0,1 in reg 1; cells 2,3 in reg 2)
    od = np.array(
        [
            [10.0, 15.0, 20.0, 25.0],
            [12.0, 18.0, 22.0, 28.0],
            [14.0, 16.0, 30.0, 35.0],
            [15.0, 20.0, 32.0, 40.0],
        ]
    )
    orig_regs = [1, 1, 2, 2]
    dest_regs = [1, 1, 2, 2]
    targets = {
        (1, 1): 100.0,
        (1, 2): 200.0,
        (2, 1): 150.0,
        (2, 2): 300.0,
    }

    calibrated = apply_biproportional_ipf(
        od_matrix=od,
        origin_regency_indices=orig_regs,
        dest_regency_indices=dest_regs,
        bps_regency_targets=targets,
    )

    # Check sub-blocks
    block_1_1 = np.sum(calibrated[:2, :2])
    block_1_2 = np.sum(calibrated[:2, 2:])
    block_2_1 = np.sum(calibrated[2:, :2])
    block_2_2 = np.sum(calibrated[2:, 2:])

    assert abs(block_1_1 - 100.0) < 1.0
    assert abs(block_1_2 - 200.0) < 1.0
    assert abs(block_2_1 - 150.0) < 1.0
    assert abs(block_2_2 - 300.0) < 1.0


def test_mode_choice_probabilities_and_shift(mode_choice_model) -> None:
    """Tests MNL utility, probability sum = 1.0, and positive transit shift."""
    mc_time = np.array([[20.0, 35.0], [30.0, 25.0]])
    car_time = np.array([[25.0, 45.0], [38.0, 32.0]])
    tr_time_base = np.array([[45.0, 60.0], [55.0, 50.0]])
    dist_km = np.array([[8.0, 15.0], [12.0, 10.0]])

    u_mc, u_car, u_tr_base = mode_choice_model.compute_utilities(
        motorcycle_time_min=mc_time,
        car_time_min=car_time,
        transit_ivt_min=tr_time_base * 0.7,
        transit_walk_min=tr_time_base * 0.2,
        transit_wait_min=tr_time_base * 0.1,
        transit_transfers=np.ones_like(mc_time),
        distance_km=dist_km,
    )
    base_shares = mode_choice_model.compute_shares(u_mc, u_car, u_tr_base)

    # Probabilities must sum to 1.0
    total_p = base_shares.p_motorcycle + base_shares.p_car + base_shares.p_transit
    assert np.allclose(total_p, 1.0)

    # Improved transit travel time (cut by 20 minutes)
    tr_time_improved = np.maximum(5.0, tr_time_base - 20.0)
    _, _, u_tr_imp = mode_choice_model.compute_utilities(
        motorcycle_time_min=mc_time,
        car_time_min=car_time,
        transit_ivt_min=tr_time_improved * 0.7,
        transit_walk_min=tr_time_improved * 0.2,
        transit_wait_min=tr_time_improved * 0.1,
        transit_transfers=np.zeros_like(mc_time),
        distance_km=dist_km,
    )
    imp_shares = mode_choice_model.compute_shares(u_mc, u_car, u_tr_imp)

    # Transit share must increase
    assert np.all(imp_shares.p_transit >= base_shares.p_transit)

    total_demand = np.array([[1000.0, 800.0], [600.0, 1200.0]])
    shift = mode_choice_model.evaluate_shift(total_demand, base_shares, imp_shares)
    assert np.all(shift.new_transit_riders >= 0.0)
    assert np.all(shift.shifted_motorcycle_trips >= 0.0)
    assert np.all(shift.shifted_car_trips >= 0.0)
