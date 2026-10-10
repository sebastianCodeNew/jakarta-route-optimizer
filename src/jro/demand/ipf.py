"""Iterative Proportional Fitting (IPF) to enforce regional BPS OD margins."""

import numpy as np

from jro.logging_setup import logger


def apply_biproportional_ipf(
    od_matrix: np.ndarray,
    origin_regency_indices: list[int],
    dest_regency_indices: list[int],
    bps_regency_targets: dict[tuple[int, int], float],
    max_iter: int = 50,
    tol: float = 1e-3,
) -> np.ndarray:
    """Adjusts cell-level OD matrix to match aggregate BPS regency-to-regency OD targets.

    Args:
        od_matrix: 2D array of initial cell-level flows T_ij. Shape: (N, M).
        origin_regency_indices: List of regency IDs for each origin cell. Length N.
        dest_regency_indices: List of regency IDs for each destination cell. Length M.
        bps_regency_targets: Dictionary mapping (orig_regency, dest_regency) -> target_commuter_flow.
        max_iter: Max IPF scaling iterations.
        tol: Relative error stopping threshold.

    Returns:
        Calibrated 2D array matching regional targets.
    """
    logger.info(
        "Applying IPF bi-proportional scaling across %d regional OD targets",
        len(bps_regency_targets),
    )

    scaled_matrix = od_matrix.copy()
    orig_regs = np.array(origin_regency_indices)
    dest_regs = np.array(dest_regency_indices)

    for it in range(max_iter):
        max_err = 0.0

        for (r_orig, r_dest), target_flow in bps_regency_targets.items():
            if target_flow <= 0:
                continue

            orig_mask = orig_regs == r_orig
            dest_mask = dest_regs == r_dest

            sub_block = scaled_matrix[np.ix_(orig_mask, dest_mask)]
            current_sum = np.sum(sub_block)

            if current_sum <= 0:
                continue

            factor = target_flow / current_sum
            scaled_matrix[np.ix_(orig_mask, dest_mask)] *= factor

            rel_err = abs(current_sum - target_flow) / target_flow
            max_err = max(max_err, rel_err)

        if max_err < tol:
            logger.debug("IPF converged at iteration %d (max_err=%.4f)", it + 1, max_err)
            break

    logger.info(
        "IPF adjustment complete. Total matrix flow: %s trips/day",
        f"{np.sum(scaled_matrix):,.0f}",
    )
    return scaled_matrix
