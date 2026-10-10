"""Calibration routines for Alternative Specific Constants (ASC)."""

import numpy as np

from jro.demand.mode_choice import MultinomialLogitModeChoice
from jro.logging_setup import logger


def calibrate_transit_asc(
    mode_choice: MultinomialLogitModeChoice,
    total_trips: np.ndarray,
    motorcycle_time_min: np.ndarray,
    car_time_min: np.ndarray,
    transit_ivt_min: np.ndarray,
    transit_walk_min: np.ndarray,
    transit_wait_min: np.ndarray,
    transit_transfers: np.ndarray,
    distance_km: np.ndarray,
    target_transit_share: float = 0.10,
    max_iter: int = 30,
    tol: float = 0.005,
) -> float:
    """Iteratively calibrates ASC_transit so aggregate transit share equals target_transit_share.

    Args:
        mode_choice: Mode choice model instance.
        total_trips: Matrix of OD flows.
        ... travel time & cost arrays ...
        target_transit_share: Target regional public transit mode share (default 10%).

    Returns:
        Calibrated asc_transit value.
    """
    logger.info(
        "Calibrating ASC_transit to reach target transit share %.1f%%", target_transit_share * 100
    )

    total_demand = np.sum(total_trips)
    if total_demand <= 0:
        return 0.0

    current_asc = 0.0
    lr = 0.8

    for it in range(max_iter):
        # Update param
        # ModeChoiceParams is frozen, so calculate utility directly with offset
        u_mc, u_car, u_tr_base = mode_choice.compute_utilities(
            motorcycle_time_min=motorcycle_time_min,
            car_time_min=car_time_min,
            transit_ivt_min=transit_ivt_min,
            transit_walk_min=transit_walk_min,
            transit_wait_min=transit_wait_min,
            transit_transfers=transit_transfers,
            distance_km=distance_km,
        )
        u_tr = u_tr_base + current_asc

        shares = mode_choice.compute_shares(u_mc, u_car, u_tr)
        modeled_transit_share = float(np.sum(total_trips * shares.p_transit) / total_demand)

        diff = target_transit_share - modeled_transit_share
        logger.debug(
            "Iter %d: ASC=%.3f, Modeled Share=%.3f%% (Target=%.3f%%, Diff=%.3f%%)",
            it + 1,
            current_asc,
            modeled_transit_share * 100,
            target_transit_share * 100,
            diff * 100,
        )

        if abs(diff) < tol:
            logger.info(
                "ASC calibration converged at %.4f (transit share %.2f%%)",
                current_asc,
                modeled_transit_share * 100,
            )
            break

        current_asc += lr * np.log(
            (target_transit_share / (1.0 - target_transit_share))
            / (max(modeled_transit_share, 1e-4) / (1.0 - min(modeled_transit_share, 0.999)))
        )

    return current_asc
