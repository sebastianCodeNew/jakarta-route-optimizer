"""Accessibility indicators: cumulative opportunities and gravity accessibility."""

import numpy as np


def compute_cumulative_accessibility(
    travel_time_matrix: np.ndarray,
    opportunities: np.ndarray,
    cutoff_minutes: float = 60.0,
) -> np.ndarray:
    """Computes reachable opportunities from each origin within a time threshold.

    Formula:
        A_i = sum_j ( O_j * 1[ T_ij <= cutoff ] )

    Args:
        travel_time_matrix: 2D array of travel times (minutes). Shape: (N, M).
        opportunities: 1D array of opportunities / jobs / attractions. Shape: (M,).
        cutoff_minutes: Travel time threshold (default 60 minutes).

    Returns:
        1D array of accessible opportunities from each origin. Shape: (N,).
    """
    reachable_mask = (travel_time_matrix <= cutoff_minutes) & (travel_time_matrix > 0)
    return np.dot(reachable_mask.astype(float), opportunities)


def compute_population_weighted_accessibility(
    accessibility: np.ndarray,
    population: np.ndarray,
) -> float:
    """Computes regional average accessibility weighted by origin population."""
    total_pop = np.sum(population)
    if total_pop <= 0:
        return 0.0
    return float(np.sum(accessibility * population) / total_pop)
