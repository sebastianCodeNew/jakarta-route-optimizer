import gc
from typing import NamedTuple

import numpy as np

from jro.config import AppConfig
from jro.demand.mode_choice import MultinomialLogitModeChoice
from jro.evaluation.emissions import compute_emission_impacts
from jro.evaluation.fast_eval import (
    RouteActiveSlice,
    evaluate_active_slice,
    evaluate_route_fast,
    extract_route_active_slice,
)
from jro.evaluation.impact import score_route_impact
from jro.evaluation.operations import RouteOperations, compute_route_operations
from jro.logging_setup import logger
from jro.optimize.search import EvaluatedRoute


class RouteRobustnessMetrics(NamedTuple):
    """Statistical stability metrics under parameter uncertainty."""

    route_id: str
    route_name: str
    probability_rank_1: float
    mean_rank: float
    riders_p05: float
    riders_p50: float
    riders_p95: float
    vkt_reduced_p05: float
    vkt_reduced_p50: float
    vkt_reduced_p95: float
    co2_saved_p05: float
    co2_saved_p50: float
    co2_saved_p95: float


def run_monte_carlo_robustness(
    top_candidates: list[EvaluatedRoute],
    base_transit_tt: np.ndarray,
    base_mc_tt: np.ndarray,
    base_car_tt: np.ndarray,
    dist_km: np.ndarray,
    od_demand: np.ndarray,
    cell_ids: list[str],
    config: AppConfig,
    num_samples: int = 100,
    seed: int = 42,
) -> list[RouteRobustnessMetrics]:
    """Evaluates stability of route rankings under parameter uncertainty.

    Args:
        top_candidates: Top candidate routes to evaluate.
        ... matrices and inputs ...
        num_samples: Number of Monte Carlo parameter draws (default 100).
        seed: Random seed for reproducibility.

    Returns:
        List of RouteRobustnessMetrics for each evaluated route.
    """
    logger.info("Running Monte Carlo robustness analysis with %d parameter samples...", num_samples)
    rng = np.random.default_rng(seed)
    cell_id_to_idx = {cid: idx for idx, cid in enumerate(cell_ids)}

    n_routes = len(top_candidates)
    if n_routes == 0:
        return []

    rank_counts = np.zeros(n_routes)
    all_ranks = np.zeros((num_samples, n_routes))
    all_riders = np.zeros((num_samples, n_routes))
    all_vkt = np.zeros((num_samples, n_routes))
    all_co2 = np.zeros((num_samples, n_routes))

    # Precompute 1D active slices and operations once per candidate
    candidate_slices: list[RouteActiveSlice | None] = []
    candidate_ops: list[RouteOperations] = []
    base_mc_model = MultinomialLogitModeChoice(config.model)
    for cand in top_candidates:
        profile = config.vehicles[cand.profile_name]
        ops_c = compute_route_operations(cand.route, profile, fleet_size=cand.optimal_fleet_size)
        candidate_ops.append(ops_c)
        if cand.slice_data is not None:
            candidate_slices.append(cand.slice_data)
        else:
            cand_slice = extract_route_active_slice(
                route=cand.route,
                commercial_speed_kmh=ops_c.commercial_speed_kmh,
                base_transit_tt=base_transit_tt,
                base_mc_tt=base_mc_tt,
                base_car_tt=base_car_tt,
                dist_km=dist_km,
                od_demand=od_demand,
                cell_id_to_idx=cell_id_to_idx,
                mode_choice=base_mc_model,
            )
            candidate_slices.append(cand_slice)

    base_mc = config.model.mode_choice

    for s in range(num_samples):
        # Sample parameter perturbations (+-20%)
        pert_beta_ivt = base_mc.beta_ivt * rng.uniform(0.80, 1.20)
        pert_beta_cost = base_mc.beta_cost * rng.uniform(0.80, 1.20)
        pert_asc_mc = base_mc.asc_motorcycle * rng.uniform(0.85, 1.15)
        pert_asc_car = base_mc.asc_car * rng.uniform(0.85, 1.15)

        # Perturbed mode choice model
        pert_mc_params = base_mc.model_copy(
            update={
                "beta_ivt": pert_beta_ivt,
                "beta_cost": pert_beta_cost,
                "asc_motorcycle": pert_asc_mc,
                "asc_car": pert_asc_car,
            }
        )
        pert_model_params = config.model.model_copy(update={"mode_choice": pert_mc_params})
        pert_mc = MultinomialLogitModeChoice(pert_model_params)

        scores_in_sample = np.zeros(n_routes)

        for r_idx, cand in enumerate(top_candidates):
            ops = candidate_ops[r_idx]
            slice_data = candidate_slices[r_idx]
            if slice_data is not None:
                fast_res = evaluate_active_slice(
                    route_id=cand.route.route_id,
                    fleet_size=ops.fleet_size,
                    wait_time_minutes=ops.wait_time_minutes,
                    daily_capacity=ops.daily_capacity_passengers,
                    slice_data=slice_data,
                    mode_choice=pert_mc,
                    recompute_base_shares=True,
                )
            else:
                fast_res = evaluate_route_fast(
                    route=cand.route,
                    operations=ops,
                    base_transit_tt=base_transit_tt,
                    base_mc_tt=base_mc_tt,
                    base_car_tt=base_car_tt,
                    dist_km=dist_km,
                    od_demand=od_demand,
                    cell_id_to_idx=cell_id_to_idx,
                    mode_choice=pert_mc,
                )
            emis = compute_emission_impacts(
                fast_res, ops, cand.route.length_km, config, cand.profile_name
            )
            score = score_route_impact(
                route=cand.route,
                eval_result=fast_res,
                emissions=emis,
                delta_accessibility=fast_res.daily_riders_served * 0.8,
                equity_desert_ratio=cand.equity_desert_ratio,
                config=config,
            )

            scores_in_sample[r_idx] = score.composite_score
            all_riders[s, r_idx] = fast_res.daily_riders_served
            all_vkt[s, r_idx] = emis.total_private_vkt_reduced
            all_co2[s, r_idx] = emis.net_co2_kg_saved

        # Compute ranks in this sample (highest score = rank 1)
        ranks = np.argsort(np.argsort(-scores_in_sample)) + 1
        all_ranks[s, :] = ranks

        winner_idx = np.argmin(ranks)
        rank_counts[winner_idx] += 1

    del candidate_slices, candidate_ops
    gc.collect()

    results: list[RouteRobustnessMetrics] = []
    for r_idx, cand in enumerate(top_candidates):
        prob_rank_1 = float(rank_counts[r_idx] / num_samples)
        mean_rank = float(np.mean(all_ranks[:, r_idx]))

        results.append(
            RouteRobustnessMetrics(
                route_id=cand.route.route_id,
                route_name=cand.route.route_name,
                probability_rank_1=round(prob_rank_1, 3),
                mean_rank=round(mean_rank, 2),
                riders_p05=round(float(np.percentile(all_riders[:, r_idx], 5)), 1),
                riders_p50=round(float(np.percentile(all_riders[:, r_idx], 50)), 1),
                riders_p95=round(float(np.percentile(all_riders[:, r_idx], 95)), 1),
                vkt_reduced_p05=round(float(np.percentile(all_vkt[:, r_idx], 5)), 1),
                vkt_reduced_p50=round(float(np.percentile(all_vkt[:, r_idx], 50)), 1),
                vkt_reduced_p95=round(float(np.percentile(all_vkt[:, r_idx], 95)), 1),
                co2_saved_p05=round(float(np.percentile(all_co2[:, r_idx], 5)), 1),
                co2_saved_p50=round(float(np.percentile(all_co2[:, r_idx], 50)), 1),
                co2_saved_p95=round(float(np.percentile(all_co2[:, r_idx], 95)), 1),
            )
        )

    logger.info("Monte Carlo robustness analysis complete.")
    return results
