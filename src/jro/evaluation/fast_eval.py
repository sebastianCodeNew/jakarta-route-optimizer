"""Vectorized min-plus matrix updater and rapid scenario evaluator."""

from typing import NamedTuple

import numpy as np

from jro.candidates.generator import CandidateRoute
from jro.demand.mode_choice import ModeShares, MultinomialLogitModeChoice
from jro.evaluation.operations import RouteOperations


class RouteActiveSlice(NamedTuple):
    """Container for pre-sliced 1D active data for rapid fleet curves and Monte Carlo."""

    route_id: str
    cand_od: np.ndarray
    cand_ivt: np.ndarray
    cand_base_tr: np.ndarray
    cand_base_mc: np.ndarray
    cand_base_car: np.ndarray
    cand_dist: np.ndarray
    base_shares: ModeShares
    fare_idr: float | None


class FastEvaluationResult(NamedTuple):
    """Results from fast min-plus route evaluation."""

    route_id: str
    fleet_size: int
    daily_riders_unconstrained: float
    daily_riders_served: float
    capacity_utilization_ratio: float
    shifted_motorcycle_trips: float
    shifted_car_trips: float
    average_time_saved_minutes: float
    new_transit_travel_time: np.ndarray


def compute_route_in_vehicle_matrix(
    route: CandidateRoute,
    commercial_speed_kmh: float,
    base_transit_tt: np.ndarray,
    dist_km: np.ndarray,
    cell_id_to_idx: dict[str, int],
) -> np.ndarray | None:
    """Computes the in-vehicle transit time matrix along route stops (excluding wait time).

    Args:
        route: CandidateRoute with stops.
        commercial_speed_kmh: Commercial speed in km/h for the vehicle profile.
        base_transit_tt: Baseline transit travel time matrix (N, N).
        dist_km: Distance matrix in km (N, N).
        cell_id_to_idx: Mapping from cell_id to matrix row/col index.

    Returns:
        (N, N) float32 matrix with minimum in-vehicle transit time via route, or None if < 2 stops.
    """
    stop_indices = [cell_id_to_idx[s.cell_id] for s in route.stops if s.cell_id in cell_id_to_idx]

    clean_indices: list[int] = []
    for idx in stop_indices:
        if not clean_indices or clean_indices[-1] != idx:
            clean_indices.append(idx)

    k_stops = len(clean_indices)
    if k_stops < 2:
        return None

    speed_km_min = commercial_speed_kmh / 60.0
    stop_cum_dist = np.zeros(k_stops, dtype=np.float32)
    for i in range(1, k_stops):
        prev_idx = clean_indices[i - 1]
        curr_idx = clean_indices[i]
        seg_dist = dist_km[prev_idx, curr_idx]
        stop_cum_dist[i] = stop_cum_dist[i - 1] + seg_dist

    stop_cum_time = stop_cum_dist / max(0.1, speed_km_min)

    d_to_stops = dist_km[:, clean_indices]
    walk_to_stops = np.where(d_to_stops <= 2.0, d_to_stops * (60.0 / 4.5), np.inf).astype(
        np.float32
    )
    access_tt = np.minimum(walk_to_stops, base_transit_tt[:, clean_indices])

    d_from_stops = dist_km[clean_indices, :]
    walk_from_stops = np.where(d_from_stops <= 2.0, d_from_stops * (60.0 / 4.5), np.inf).astype(
        np.float32
    )
    egress_tt = np.minimum(walk_from_stops, base_transit_tt[clean_indices, :])

    vec_a = access_tt - stop_cum_time[np.newaxis, :]
    vec_b = egress_tt + stop_cum_time[:, np.newaxis]

    cum_min_a = np.minimum.accumulate(vec_a[:, :-1], axis=1)
    b_subset = vec_b[1:, :]

    n_cells = base_transit_tt.shape[0]
    min_ivt = np.full((n_cells, n_cells), np.inf, dtype=np.float32)
    for b in range(k_stops - 1):
        term = cum_min_a[:, b, np.newaxis] + b_subset[b, np.newaxis, :]
        np.minimum(min_ivt, term, out=min_ivt)
    del cum_min_a, b_subset

    # Reverse trip
    rev_cum_time = stop_cum_time[-1] - stop_cum_time
    rev_vec_a = access_tt - rev_cum_time[np.newaxis, :]
    rev_vec_b = egress_tt + rev_cum_time[:, np.newaxis]

    rev_cum_min_a = np.minimum.accumulate(rev_vec_a[:, ::-1][:, :-1], axis=1)
    rev_b_subset = rev_vec_b[::-1][1:, :]

    for b in range(k_stops - 1):
        term = rev_cum_min_a[:, b, np.newaxis] + rev_b_subset[b, np.newaxis, :]
        np.minimum(min_ivt, term, out=min_ivt)
    del rev_cum_min_a, rev_b_subset, access_tt, egress_tt

    return min_ivt


def evaluate_route_fast(
    route: CandidateRoute,
    operations: RouteOperations,
    base_transit_tt: np.ndarray,
    base_mc_tt: np.ndarray,
    base_car_tt: np.ndarray,
    dist_km: np.ndarray,
    od_demand: np.ndarray,
    cell_id_to_idx: dict[str, int],
    mode_choice: MultinomialLogitModeChoice,
    route_ivt_matrix: np.ndarray | None = None,
    return_full_matrix: bool | None = None,
) -> FastEvaluationResult:
    """Evaluates candidate route using vectorized min-plus travel time improvement.

    Args:
        route: CandidateRoute under evaluation.
        operations: RouteOperations computed for route and fleet size.
        base_transit_tt: Baseline transit travel time matrix. Shape: (N, N).
        base_mc_tt: Baseline motorcycle travel time matrix. Shape: (N, N).
        base_car_tt: Baseline car travel time matrix. Shape: (N, N).
        dist_km: Distance matrix in km. Shape: (N, N).
        od_demand: Daily commuter OD matrix. Shape: (N, N).
        cell_id_to_idx: Mapping from H3 cell_id to matrix index.
        mode_choice: MultinomialLogitModeChoice model instance.
        route_ivt_matrix: Optional precomputed in-vehicle time matrix for this route.
        return_full_matrix: Whether to populate new_transit_travel_time in result.
            Defaults to True if N <= 100, False otherwise to preserve memory.

    Returns:
        FastEvaluationResult instance.
    """
    need_full_matrix = (
        return_full_matrix if return_full_matrix is not None else (base_transit_tt.shape[0] <= 100)
    )

    min_ivt: np.ndarray | None
    if route_ivt_matrix is not None:
        min_ivt = route_ivt_matrix
    else:
        min_ivt = compute_route_in_vehicle_matrix(
            route=route,
            commercial_speed_kmh=operations.commercial_speed_kmh,
            base_transit_tt=base_transit_tt,
            dist_km=dist_km,
            cell_id_to_idx=cell_id_to_idx,
        )

    if min_ivt is None:
        return FastEvaluationResult(
            route_id=route.route_id,
            fleet_size=operations.fleet_size,
            daily_riders_unconstrained=0.0,
            daily_riders_served=0.0,
            capacity_utilization_ratio=0.0,
            shifted_motorcycle_trips=0.0,
            shifted_car_trips=0.0,
            average_time_saved_minutes=0.0,
            new_transit_travel_time=base_transit_tt.copy()
            if need_full_matrix
            else np.empty((0, 0), dtype=np.float32),
        )

    wait = operations.wait_time_minutes
    route_tt = min_ivt + wait
    improved_mask = route_tt < base_transit_tt

    if not np.any(improved_mask):
        return FastEvaluationResult(
            route_id=route.route_id,
            fleet_size=operations.fleet_size,
            daily_riders_unconstrained=0.0,
            daily_riders_served=0.0,
            capacity_utilization_ratio=0.0,
            shifted_motorcycle_trips=0.0,
            shifted_car_trips=0.0,
            average_time_saved_minutes=0.0,
            new_transit_travel_time=base_transit_tt.copy()
            if need_full_matrix
            else np.empty((0, 0), dtype=np.float32),
        )

    # Time savings for improved pairs
    time_diff_improved = base_transit_tt[improved_mask] - route_tt[improved_mask]
    avg_saved = float(np.mean(time_diff_improved))

    # Evaluate mode choice shift only on improved pairs with positive commuter demand
    active_mask = improved_mask & (od_demand > 0)
    if not np.any(active_mask):
        unconstrained_riders = 0.0
        served_riders = 0.0
        utilization = 0.0
        shifted_mc = 0.0
        shifted_car = 0.0
    else:
        act_mc_tt = base_mc_tt[active_mask]
        act_car_tt = base_car_tt[active_mask]
        act_base_tr = base_transit_tt[active_mask]
        act_new_tr = route_tt[active_mask]
        act_dist = dist_km[active_mask]
        act_od = od_demand[active_mask]

        u_mc_base, u_car_base, u_tr_base = mode_choice.compute_utilities(
            motorcycle_time_min=act_mc_tt,
            car_time_min=act_car_tt,
            transit_ivt_min=act_base_tr * 0.75,
            transit_walk_min=act_base_tr * 0.20,
            transit_wait_min=act_base_tr * 0.05,
            transit_transfers=np.ones_like(act_base_tr),
            distance_km=act_dist,
        )
        base_shares = mode_choice.compute_shares(u_mc_base, u_car_base, u_tr_base)

        fare = 0.0 if route.vehicle_profile == "mikrotrans" else None
        u_mc_new, u_car_new, u_tr_new = mode_choice.compute_utilities(
            motorcycle_time_min=act_mc_tt,
            car_time_min=act_car_tt,
            transit_ivt_min=act_new_tr * 0.75,
            transit_walk_min=act_new_tr * 0.20,
            transit_wait_min=act_new_tr * 0.05,
            transit_transfers=np.full_like(act_new_tr, 0.5),
            distance_km=act_dist,
            fare_idr=fare,
        )
        improved_shares = mode_choice.compute_shares(u_mc_new, u_car_new, u_tr_new)

        shift = mode_choice.evaluate_shift(act_od, base_shares, improved_shares)

        corridor_new = float(np.sum(shift.new_transit_riders))
        corridor_existing = float(np.sum(act_od * base_shares.p_transit))
        unconstrained_riders = (corridor_new + corridor_existing * 0.5) * 2.0
        daily_cap = operations.daily_capacity_passengers
        cap_factor = min(1.0, daily_cap / max(1.0, unconstrained_riders))
        served_riders = unconstrained_riders * cap_factor
        utilization = min(1.0, unconstrained_riders / max(1.0, daily_cap))
        shifted_mc = float(np.sum(shift.shifted_motorcycle_trips)) * cap_factor * 2.0
        shifted_car = float(np.sum(shift.shifted_car_trips)) * cap_factor * 2.0

    if need_full_matrix:
        new_transit_tt = np.minimum(base_transit_tt, route_tt)
    else:
        new_transit_tt = np.empty((0, 0), dtype=np.float32)

    return FastEvaluationResult(
        route_id=route.route_id,
        fleet_size=operations.fleet_size,
        daily_riders_unconstrained=round(unconstrained_riders, 1),
        daily_riders_served=round(served_riders, 1),
        capacity_utilization_ratio=round(utilization, 3),
        shifted_motorcycle_trips=round(shifted_mc, 1),
        shifted_car_trips=round(shifted_car, 1),
        average_time_saved_minutes=round(avg_saved, 2),
        new_transit_travel_time=new_transit_tt,
    )


def extract_route_active_slice(
    route: CandidateRoute,
    commercial_speed_kmh: float,
    base_transit_tt: np.ndarray,
    base_mc_tt: np.ndarray,
    base_car_tt: np.ndarray,
    dist_km: np.ndarray,
    od_demand: np.ndarray,
    cell_id_to_idx: dict[str, int],
    mode_choice: MultinomialLogitModeChoice,
    route_ivt_matrix: np.ndarray | None = None,
    min_wait_minutes: float = 1.0,
) -> RouteActiveSlice | None:
    """Extracts 1D vectors for active improvable cell pairs to accelerate evaluation."""
    min_ivt: np.ndarray | None
    if route_ivt_matrix is not None:
        min_ivt = route_ivt_matrix
    else:
        min_ivt = compute_route_in_vehicle_matrix(
            route=route,
            commercial_speed_kmh=commercial_speed_kmh,
            base_transit_tt=base_transit_tt,
            dist_km=dist_km,
            cell_id_to_idx=cell_id_to_idx,
        )
    if min_ivt is None:
        return None

    active_mask = ((min_ivt + min_wait_minutes) < base_transit_tt) & (od_demand > 0)
    if not np.any(active_mask):
        return None

    cand_od = od_demand[active_mask]
    cand_ivt = min_ivt[active_mask]
    cand_base_tr = base_transit_tt[active_mask]
    cand_base_mc = base_mc_tt[active_mask]
    cand_base_car = base_car_tt[active_mask]
    cand_dist = dist_km[active_mask]

    fare = 0.0 if route.vehicle_profile == "mikrotrans" else None
    u_mc_b, u_car_b, u_tr_b = mode_choice.compute_utilities(
        motorcycle_time_min=cand_base_mc,
        car_time_min=cand_base_car,
        transit_ivt_min=cand_base_tr * 0.75,
        transit_walk_min=cand_base_tr * 0.20,
        transit_wait_min=cand_base_tr * 0.05,
        transit_transfers=np.ones_like(cand_base_tr),
        distance_km=cand_dist,
    )
    b_shares = mode_choice.compute_shares(u_mc_b, u_car_b, u_tr_b)

    return RouteActiveSlice(
        route_id=route.route_id,
        cand_od=cand_od,
        cand_ivt=cand_ivt,
        cand_base_tr=cand_base_tr,
        cand_base_mc=cand_base_mc,
        cand_base_car=cand_base_car,
        cand_dist=cand_dist,
        base_shares=b_shares,
        fare_idr=fare,
    )


def evaluate_active_slice(
    route_id: str,
    fleet_size: int,
    wait_time_minutes: float,
    daily_capacity: float,
    slice_data: RouteActiveSlice,
    mode_choice: MultinomialLogitModeChoice,
    recompute_base_shares: bool = False,
) -> FastEvaluationResult:
    """Evaluates 1D active vector in microseconds for fleet scaling sweeps and Monte Carlo."""
    route_tt = slice_data.cand_ivt + wait_time_minutes
    sub_mask = route_tt < slice_data.cand_base_tr
    if not np.any(sub_mask):
        return FastEvaluationResult(
            route_id=route_id,
            fleet_size=fleet_size,
            daily_riders_unconstrained=0.0,
            daily_riders_served=0.0,
            capacity_utilization_ratio=0.0,
            shifted_motorcycle_trips=0.0,
            shifted_car_trips=0.0,
            average_time_saved_minutes=0.0,
            new_transit_travel_time=np.empty((0, 0), dtype=np.float32),
        )

    sub_dem = slice_data.cand_od[sub_mask]
    sub_mc = slice_data.cand_base_mc[sub_mask]
    sub_car = slice_data.cand_base_car[sub_mask]
    sub_btr = slice_data.cand_base_tr[sub_mask]
    sub_ntr = route_tt[sub_mask]
    sub_d = slice_data.cand_dist[sub_mask]

    if recompute_base_shares:
        u_mc_b, u_car_b, u_tr_b = mode_choice.compute_utilities(
            motorcycle_time_min=sub_mc,
            car_time_min=sub_car,
            transit_ivt_min=sub_btr * 0.75,
            transit_walk_min=sub_btr * 0.20,
            transit_wait_min=sub_btr * 0.05,
            transit_transfers=np.ones_like(sub_btr),
            distance_km=sub_d,
        )
        base_shares = mode_choice.compute_shares(u_mc_b, u_car_b, u_tr_b)
    else:
        base_shares = ModeShares(
            p_motorcycle=slice_data.base_shares.p_motorcycle[sub_mask],
            p_car=slice_data.base_shares.p_car[sub_mask],
            p_transit=slice_data.base_shares.p_transit[sub_mask],
        )

    u_mc_new, u_car_new, u_tr_new = mode_choice.compute_utilities(
        motorcycle_time_min=sub_mc,
        car_time_min=sub_car,
        transit_ivt_min=sub_ntr * 0.75,
        transit_walk_min=sub_ntr * 0.20,
        transit_wait_min=sub_ntr * 0.05,
        transit_transfers=np.full_like(sub_ntr, 0.5),
        distance_km=sub_d,
        fare_idr=slice_data.fare_idr,
    )
    improved_shares = mode_choice.compute_shares(u_mc_new, u_car_new, u_tr_new)
    shift = mode_choice.evaluate_shift(sub_dem, base_shares, improved_shares)

    corridor_new = float(np.sum(shift.new_transit_riders))
    corridor_existing = float(np.sum(sub_dem * base_shares.p_transit))
    unconstrained_riders = (corridor_new + corridor_existing * 0.5) * 2.0

    cap_factor = min(1.0, daily_capacity / max(1.0, unconstrained_riders))
    served_riders = unconstrained_riders * cap_factor
    utilization = min(1.0, unconstrained_riders / max(1.0, daily_capacity))
    shifted_mc = float(np.sum(shift.shifted_motorcycle_trips)) * cap_factor * 2.0
    shifted_car = float(np.sum(shift.shifted_car_trips)) * cap_factor * 2.0
    avg_saved = float(np.mean(sub_btr - sub_ntr))

    return FastEvaluationResult(
        route_id=route_id,
        fleet_size=fleet_size,
        daily_riders_unconstrained=round(unconstrained_riders, 1),
        daily_riders_served=round(served_riders, 1),
        capacity_utilization_ratio=round(utilization, 3),
        shifted_motorcycle_trips=round(shifted_mc, 1),
        shifted_car_trips=round(shifted_car, 1),
        average_time_saved_minutes=round(avg_saved, 2),
        new_transit_travel_time=np.empty((0, 0), dtype=np.float32),
    )
