import gc
from typing import NamedTuple

import numpy as np

from jro.candidates.generator import CandidateRoute, generate_candidate_routes
from jro.config import AppConfig
from jro.demand.mode_choice import MultinomialLogitModeChoice
from jro.evaluation.emissions import EmissionImpactResult, compute_emission_impacts
from jro.evaluation.fast_eval import (
    FastEvaluationResult,
    RouteActiveSlice,
    compute_route_in_vehicle_matrix,
    evaluate_active_slice,
    evaluate_route_fast,
    extract_route_active_slice,
)
from jro.evaluation.impact import RouteImpactScore, score_route_impact
from jro.evaluation.operations import RouteOperations, compute_route_operations
from jro.logging_setup import logger


class FleetCurvePoint(NamedTuple):
    """Data point on the fleet scaling curve (N = 1 to 30)."""

    fleet_size: int
    headway_minutes: float
    wait_time_minutes: float
    daily_capacity: float
    daily_riders_served: float
    capacity_utilization: float
    private_vkt_reduced: float
    net_co2_kg_saved: float
    composite_score: float


class EvaluatedRoute(NamedTuple):
    """Complete evaluation report for a candidate route."""

    route: CandidateRoute
    profile_name: str
    optimal_fleet_size: int
    operations: RouteOperations
    eval_result: FastEvaluationResult
    emissions: EmissionImpactResult
    impact_score: RouteImpactScore
    fleet_curve: list[FleetCurvePoint]
    equity_desert_ratio: float
    slice_data: RouteActiveSlice | None = None


class OptimizationSummary(NamedTuple):
    """Overall optimization results comparing profiles and ranking candidates."""

    ranked_candidates: list[EvaluatedRoute]
    best_mikrotrans: EvaluatedRoute
    best_non_brt: EvaluatedRoute
    overall_winner: EvaluatedRoute


def compute_fleet_scaling_curve(
    route: CandidateRoute,
    profile_name: str,
    base_transit_tt: np.ndarray,
    base_mc_tt: np.ndarray,
    base_car_tt: np.ndarray,
    dist_km: np.ndarray,
    od_demand: np.ndarray,
    cell_id_to_idx: dict[str, int],
    mode_choice: MultinomialLogitModeChoice,
    config: AppConfig,
    equity_desert_ratio: float,
    max_fleet: int = 30,
    route_ivt_matrix: np.ndarray | None = None,
    slice_data: RouteActiveSlice | None = None,
) -> list[FleetCurvePoint]:
    """Sweeps fleet size N from 1 to max_fleet to produce the impact scaling curve."""
    profile = config.vehicles[profile_name]
    curve: list[FleetCurvePoint] = []

    ivt_matrix = route_ivt_matrix
    if slice_data is None and ivt_matrix is None:
        ops_init = compute_route_operations(route, profile, fleet_size=1)
        ivt_matrix = compute_route_in_vehicle_matrix(
            route=route,
            commercial_speed_kmh=ops_init.commercial_speed_kmh,
            base_transit_tt=base_transit_tt,
            dist_km=dist_km,
            cell_id_to_idx=cell_id_to_idx,
        )

    for n in range(1, max_fleet + 1):
        ops = compute_route_operations(route, profile, fleet_size=n)
        if slice_data is not None:
            eval_res = evaluate_active_slice(
                route_id=route.route_id,
                fleet_size=n,
                wait_time_minutes=ops.wait_time_minutes,
                daily_capacity=ops.daily_capacity_passengers,
                slice_data=slice_data,
                mode_choice=mode_choice,
            )
        else:
            eval_res = evaluate_route_fast(
                route=route,
                operations=ops,
                base_transit_tt=base_transit_tt,
                base_mc_tt=base_mc_tt,
                base_car_tt=base_car_tt,
                dist_km=dist_km,
                od_demand=od_demand,
                cell_id_to_idx=cell_id_to_idx,
                mode_choice=mode_choice,
                route_ivt_matrix=ivt_matrix,
            )
        emis = compute_emission_impacts(eval_res, ops, route.length_km, config, profile_name)
        score = score_route_impact(
            route=route,
            eval_result=eval_res,
            emissions=emis,
            delta_accessibility=eval_res.daily_riders_served * 0.8,
            equity_desert_ratio=equity_desert_ratio,
            config=config,
        )
        curve.append(
            FleetCurvePoint(
                fleet_size=n,
                headway_minutes=ops.headway_minutes,
                wait_time_minutes=ops.wait_time_minutes,
                daily_capacity=ops.daily_capacity_passengers,
                daily_riders_served=eval_res.daily_riders_served,
                capacity_utilization=eval_res.capacity_utilization_ratio,
                private_vkt_reduced=emis.total_private_vkt_reduced,
                net_co2_kg_saved=emis.net_co2_kg_saved,
                composite_score=score.composite_score,
            )
        )
    return curve


def run_route_optimization(
    base_transit_tt: np.ndarray,
    base_mc_tt: np.ndarray,
    base_car_tt: np.ndarray,
    dist_km: np.ndarray,
    od_demand: np.ndarray,
    cell_ids: list[str],
    transit_desert_cells: set[str],
    config: AppConfig,
    default_fleet_size: int = 12,
) -> OptimizationSummary:
    """Evaluates and ranks all candidate routes across vehicle profiles.

    Args:
        base_transit_tt: Baseline transit travel time matrix.
        base_mc_tt: Baseline motorcycle travel time matrix.
        base_car_tt: Baseline car travel time matrix.
        dist_km: Distance matrix in km.
        od_demand: Matrix of commuter trips.
        cell_ids: List of H3 cell IDs corresponding to matrix rows/cols.
        transit_desert_cells: Set of cell IDs marked as transit deserts.
        config: System configuration.
        default_fleet_size: Fleet size for initial comparative ranking.

    Returns:
        OptimizationSummary instance with ranked candidates and profile winners.
    """
    logger.info("Starting multi-criteria route optimization across all profiles...")
    cell_id_to_idx = {cid: idx for idx, cid in enumerate(cell_ids)}
    mode_choice = MultinomialLogitModeChoice(config.model)

    evaluated_routes: list[EvaluatedRoute] = []

    for prof_name, profile in config.vehicles.items():
        candidates = generate_candidate_routes(
            profile=profile,
            max_overlap_ratio=config.constraints.max_overlap_ratio,
            h3_resolution=config.settings.h3_resolution,
        )

        for cand in candidates:
            # Calculate equity ratio: proportion of stops in transit desert cells
            stop_cells_set = set(cand.stop_cell_ids)
            desert_stops = stop_cells_set.intersection(transit_desert_cells)
            eq_ratio = len(desert_stops) / max(1, len(stop_cells_set))

            # Operations under default fleet size
            ops = compute_route_operations(cand, profile, fleet_size=default_fleet_size)
            route_ivt = compute_route_in_vehicle_matrix(
                route=cand,
                commercial_speed_kmh=ops.commercial_speed_kmh,
                base_transit_tt=base_transit_tt,
                dist_km=dist_km,
                cell_id_to_idx=cell_id_to_idx,
            )
            # Extract 1D active slice for rapid evaluation and Monte Carlo
            cand_slice = extract_route_active_slice(
                route=cand,
                commercial_speed_kmh=ops.commercial_speed_kmh,
                base_transit_tt=base_transit_tt,
                base_mc_tt=base_mc_tt,
                base_car_tt=base_car_tt,
                dist_km=dist_km,
                od_demand=od_demand,
                cell_id_to_idx=cell_id_to_idx,
                mode_choice=mode_choice,
                route_ivt_matrix=route_ivt,
            )

            if cand_slice is not None:
                fast_res = evaluate_active_slice(
                    route_id=cand.route_id,
                    fleet_size=ops.fleet_size,
                    wait_time_minutes=ops.wait_time_minutes,
                    daily_capacity=ops.daily_capacity_passengers,
                    slice_data=cand_slice,
                    mode_choice=mode_choice,
                )
            else:
                fast_res = evaluate_route_fast(
                    route=cand,
                    operations=ops,
                    base_transit_tt=base_transit_tt,
                    base_mc_tt=base_mc_tt,
                    base_car_tt=base_car_tt,
                    dist_km=dist_km,
                    od_demand=od_demand,
                    cell_id_to_idx=cell_id_to_idx,
                    mode_choice=mode_choice,
                    route_ivt_matrix=route_ivt,
                )

            emissions = compute_emission_impacts(fast_res, ops, cand.length_km, config, prof_name)
            impact_score = score_route_impact(
                route=cand,
                eval_result=fast_res,
                emissions=emissions,
                delta_accessibility=fast_res.daily_riders_served * 0.8,
                equity_desert_ratio=eq_ratio,
                config=config,
            )

            # Generate fleet scaling curve (N = 1 to 30)
            fleet_curve = compute_fleet_scaling_curve(
                route=cand,
                profile_name=prof_name,
                base_transit_tt=base_transit_tt,
                base_mc_tt=base_mc_tt,
                base_car_tt=base_car_tt,
                dist_km=dist_km,
                od_demand=od_demand,
                cell_id_to_idx=cell_id_to_idx,
                mode_choice=mode_choice,
                config=config,
                equity_desert_ratio=eq_ratio,
                max_fleet=30,
                route_ivt_matrix=route_ivt,
                slice_data=cand_slice,
            )

            del route_ivt
            gc.collect()

            # Find diminishing returns elbow in fleet curve
            optimal_n = default_fleet_size
            for pt in fleet_curve:
                if pt.capacity_utilization >= 0.70 and pt.headway_minutes <= 10.0:
                    optimal_n = pt.fleet_size
                    break

            evaluated_routes.append(
                EvaluatedRoute(
                    route=cand,
                    profile_name=prof_name,
                    optimal_fleet_size=optimal_n,
                    operations=ops,
                    eval_result=fast_res,
                    emissions=emissions,
                    impact_score=impact_score,
                    fleet_curve=fleet_curve,
                    equity_desert_ratio=eq_ratio,
                    slice_data=cand_slice,
                )
            )

    # Sort candidates by composite score descending
    ranked = sorted(evaluated_routes, key=lambda x: x.impact_score.composite_score, reverse=True)

    mikrotrans_candidates = [r for r in ranked if r.profile_name == "mikrotrans"]
    non_brt_candidates = [r for r in ranked if r.profile_name == "non_brt"]

    best_mikro = mikrotrans_candidates[0] if mikrotrans_candidates else ranked[0]
    best_nonbrt = non_brt_candidates[0] if non_brt_candidates else ranked[0]
    overall_winner = ranked[0]

    logger.info(
        "Optimization complete: Evaluated %d candidate routes. Overall Winner: %s (%s, Score: %.2f)",
        len(ranked),
        overall_winner.route.route_name,
        overall_winner.profile_name,
        overall_winner.impact_score.composite_score,
    )
    return OptimizationSummary(
        ranked_candidates=ranked,
        best_mikrotrans=best_mikro,
        best_non_brt=best_nonbrt,
        overall_winner=overall_winner,
    )
