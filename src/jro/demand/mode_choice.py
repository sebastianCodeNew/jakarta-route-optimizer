"""Multinomial Logit (MNL) mode choice model and mode-shift evaluation."""

from typing import NamedTuple

import numpy as np

from jro.config import ModelParams


class ModeShares(NamedTuple):
    """Container for predicted mode probabilities."""

    p_motorcycle: np.ndarray
    p_car: np.ndarray
    p_transit: np.ndarray


class ModeShiftResult(NamedTuple):
    """Container for mode shift resulting from transit improvements."""

    new_transit_riders: np.ndarray
    shifted_motorcycle_trips: np.ndarray
    shifted_car_trips: np.ndarray


class MultinomialLogitModeChoice:
    """Evaluates commuter travel mode choice utilities and probabilities."""

    def __init__(self, params: ModelParams) -> None:
        self.params = params

    def compute_utilities(
        self,
        motorcycle_time_min: np.ndarray,
        car_time_min: np.ndarray,
        transit_ivt_min: np.ndarray,
        transit_walk_min: np.ndarray,
        transit_wait_min: np.ndarray,
        transit_transfers: np.ndarray,
        distance_km: np.ndarray,
        fare_idr: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates systematic utility for motorcycle, car, and transit."""
        mc = self.params.mode_choice
        costs = self.params.costs

        # Cost estimates
        cost_mc = distance_km * costs.motorcycle_cost_per_km_idr + costs.motorcycle_parking_idr
        cost_car = distance_km * costs.car_cost_per_km_idr + costs.car_parking_idr
        cost_transit = fare_idr if fare_idr is not None else costs.fare_non_brt_idr

        u_mc = mc.asc_motorcycle + mc.beta_ivt * motorcycle_time_min + mc.beta_cost * cost_mc

        u_car = mc.asc_car + mc.beta_ivt * car_time_min + mc.beta_cost * cost_car

        u_transit = (
            mc.asc_transit
            + mc.beta_ivt * transit_ivt_min
            + mc.beta_walk * transit_walk_min
            + mc.beta_wait * transit_wait_min
            + mc.beta_xfer * transit_transfers
            + mc.beta_cost * cost_transit
        )

        return u_mc, u_car, u_transit

    def compute_shares(
        self,
        u_mc: np.ndarray,
        u_car: np.ndarray,
        u_transit: np.ndarray,
    ) -> ModeShares:
        """Computes softmax probabilities across modes."""
        # Subtract max for numerical stability
        u_max = np.maximum(np.maximum(u_mc, u_car), u_transit)
        exp_mc = np.exp(u_mc - u_max)
        exp_car = np.exp(u_car - u_max)
        exp_transit = np.exp(u_transit - u_max)

        sum_exp = exp_mc + exp_car + exp_transit
        sum_exp[sum_exp <= 0] = 1e-9

        return ModeShares(
            p_motorcycle=exp_mc / sum_exp,
            p_car=exp_car / sum_exp,
            p_transit=exp_transit / sum_exp,
        )

    def evaluate_shift(
        self,
        total_trips: np.ndarray,
        base_shares: ModeShares,
        improved_shares: ModeShares,
    ) -> ModeShiftResult:
        """Calculates trips shifted from private modes to public transit.

        Args:
            total_trips: Matrix of total OD commute trips.
            base_shares: Baseline mode probabilities before new route.
            improved_shares: New mode probabilities after new route.

        Returns:
            ModeShiftResult containing newly attracted riders and diverted private trips.
        """
        delta_p_transit = np.maximum(0.0, improved_shares.p_transit - base_shares.p_transit)
        new_riders = total_trips * delta_p_transit

        # Proportional diversion from private modes
        private_total = base_shares.p_motorcycle + base_shares.p_car
        private_total[private_total <= 0] = 1e-9

        ratio_mc = base_shares.p_motorcycle / private_total
        ratio_car = base_shares.p_car / private_total

        shifted_mc = new_riders * ratio_mc
        shifted_car = new_riders * ratio_car

        return ModeShiftResult(
            new_transit_riders=new_riders,
            shifted_motorcycle_trips=shifted_mc,
            shifted_car_trips=shifted_car,
        )
