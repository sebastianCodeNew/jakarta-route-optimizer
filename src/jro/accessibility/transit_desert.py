"""Transit desert identification and scoring for Jabodetabek."""

import numpy as np
import pandas as pd

from jro.logging_setup import logger


def compute_transit_desert_scores(
    cell_ids: list[str],
    population: np.ndarray,
    production: np.ndarray,
    transit_accessibility: np.ndarray,
) -> pd.DataFrame:
    """Computes transit desert index per cell.

    Formula:
        DesertIndex = Norm(Pop) * Norm(Demand) * (1 - Norm(TransitAccess))

    Args:
        cell_ids: List of H3 cell IDs.
        population: 1D array of cell populations.
        production: 1D array of cell commuter production.
        transit_accessibility: 1D array of reachable jobs via transit within 60 min.

    Returns:
        DataFrame with ['cell_id', 'desert_score', 'is_desert'].
    """
    logger.info("Computing transit desert scores for %d cells", len(cell_ids))

    def min_max_norm(arr: np.ndarray) -> np.ndarray:
        p_min, p_max = np.min(arr), np.max(arr)
        if p_max <= p_min:
            return np.zeros_like(arr)
        return (arr - p_min) / (p_max - p_min)

    norm_pop = min_max_norm(population)
    norm_dem = min_max_norm(production)
    norm_acc = min_max_norm(transit_accessibility)

    # Desert index: high demand & population, but low transit access
    desert_score = (0.5 * norm_pop + 0.5 * norm_dem) * (1.0 - norm_acc)

    # Mark top 15% as transit deserts
    threshold = np.percentile(desert_score, 85)
    is_desert = desert_score >= threshold

    df = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "desert_score": desert_score,
            "is_desert": is_desert,
        }
    )

    logger.info(
        "Identified %d transit desert cells (threshold >= %.3f)", int(np.sum(is_desert)), threshold
    )
    return df
