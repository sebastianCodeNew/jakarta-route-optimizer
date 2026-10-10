"""Composite multi-criteria impact scoring."""

from typing import NamedTuple

from jro.candidates.generator import CandidateRoute
from jro.config import AppConfig
from jro.evaluation.emissions import EmissionImpactResult
from jro.evaluation.fast_eval import FastEvaluationResult


class RouteImpactScore(NamedTuple):
    """Normalized multi-criteria score breakdown."""

    composite_score: float
    score_vkt: float
    score_emissions: float
    score_ridership: float
    score_accessibility: float
    score_equity: float
    penalty_deduction: float


def score_route_impact(
    route: CandidateRoute,
    eval_result: FastEvaluationResult,
    emissions: EmissionImpactResult,
    delta_accessibility: float,
    equity_desert_ratio: float,
    config: AppConfig,
    norm_max_vkt: float = 50000.0,
    norm_max_co2_kg: float = 8000.0,
    norm_max_riders: float = 15000.0,
    norm_max_access: float = 10000.0,
) -> RouteImpactScore:
    """Computes weighted multi-criteria score normalized to [0, 100].

    Formula:
        Score = w1 * S_vkt + w2 * S_emis + w3 * S_riders + w4 * S_acc + w5 * S_eq - Penalty

    Args:
        route: CandidateRoute instance.
        eval_result: FastEvaluationResult.
        emissions: EmissionImpactResult.
        delta_accessibility: Net increase in regional jobs reachable.
        equity_desert_ratio: Fraction of route stops situated in Transit Desert cells.
        config: AppConfig with weights and constraint penalties.
        norm_max_...: Normalization reference scales.

    Returns:
        RouteImpactScore instance.
    """
    w = config.weights

    # Normalized component scores [0, 100]
    s_vkt = min(100.0, max(0.0, (emissions.total_private_vkt_reduced / norm_max_vkt) * 100.0))
    s_emis = min(100.0, max(0.0, (emissions.net_co2_kg_saved / norm_max_co2_kg) * 100.0))
    s_riders = min(100.0, max(0.0, (eval_result.daily_riders_served / norm_max_riders) * 100.0))
    s_acc = min(100.0, max(0.0, (delta_accessibility / norm_max_access) * 100.0))
    s_equity = min(100.0, max(0.0, equity_desert_ratio * 100.0))

    # Overlap penalty if above threshold
    penalty = 0.0
    if route.overlap_ratio > config.constraints.max_overlap_ratio:
        excess = route.overlap_ratio - config.constraints.max_overlap_ratio
        penalty += excess * 50.0  # penalty scaled

    composite = (
        w.vkt_reduction * s_vkt
        + w.emission_reduction * s_emis
        + w.ridership * s_riders
        + w.accessibility_gain * s_acc
        + w.equity_transit_desert * s_equity
        - penalty
    )

    return RouteImpactScore(
        composite_score=round(max(0.0, composite), 2),
        score_vkt=round(s_vkt, 2),
        score_emissions=round(s_emis, 2),
        score_ridership=round(s_riders, 2),
        score_accessibility=round(s_acc, 2),
        score_equity=round(s_equity, 2),
        penalty_deduction=round(penalty, 2),
    )
