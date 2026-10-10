"""Doubly-constrained gravity model for trip distribution."""

import numpy as np

from jro.exceptions import DemandModelError
from jro.logging_setup import logger


def doubly_constrained_gravity(
    origins: np.ndarray,
    destinations: np.ndarray,
    cost_matrix: np.ndarray,
    beta: float = 0.035,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> np.ndarray:
    """Computes doubly-constrained spatial interaction trip distribution matrix.

    Formula:
        T_ij = A_i * B_j * O_i * D_j * exp(-beta * c_ij)
        Subject to:
            sum_j T_ij = O_i
            sum_i T_ij = D_j

    Args:
        origins: 1D array of origin productions (O_i). Shape: (N,)
        destinations: 1D array of destination attractions (D_j). Shape: (M,)
        cost_matrix: 2D array of travel costs / times (c_ij in minutes). Shape: (N, M)
        beta: Exponential distance decay parameter.
        max_iter: Maximum balancing iterations.
        tol: Convergence tolerance for marginal sum errors.

    Returns:
        2D trip flow matrix T_ij. Shape: (N, M)
    """
    logger.info(
        "Running doubly-constrained gravity model (%d origins, %d destinations, beta=%.4f)",
        len(origins),
        len(destinations),
        beta,
    )

    total_o = np.sum(origins)
    total_d = np.sum(destinations)

    if total_o <= 0 or total_d <= 0:
        raise DemandModelError("Origins or destinations have zero sum")

    # Rescale destinations to match total origin production to ensure balance
    dest_scaled = (destinations * (total_o / total_d)).astype(np.float32)
    origins_f32 = origins.astype(np.float32)

    # Friction factor matrix F_ij = exp(-beta * c_ij) in float32
    costs_clipped = np.clip(cost_matrix, 0.0, 180.0).astype(np.float32)
    friction = np.exp(-beta * costs_clipped, dtype=np.float32)
    friction[cost_matrix > 180.0] = 0.0
    del costs_clipped

    b = np.ones_like(dest_scaled, dtype=np.float32)
    a = np.ones_like(origins_f32, dtype=np.float32)

    for it in range(max_iter):
        # Update A_i = 1 / sum_j (B_j * D_j * F_ij)
        denom_a = np.dot(friction, b * dest_scaled)
        denom_a[denom_a <= 0] = 1e-9
        a = 1.0 / denom_a

        # Update B_j = 1 / sum_i (A_i * O_i * F_ij)
        denom_b = np.dot(a * origins_f32, friction)
        denom_b[denom_b <= 0] = 1e-9
        new_b = 1.0 / denom_b

        # Check convergence
        rel_diff = float(np.max(np.abs(new_b - b) / (np.abs(b) + 1e-9)))
        b = new_b
        if rel_diff < tol:
            logger.debug(
                "Gravity model converged at iteration %d (rel_diff=%.2e)", it + 1, rel_diff
            )
            break

    # In-place matrix scaling to conserve memory
    trips = friction
    trips *= (a * origins_f32)[:, np.newaxis]
    trips *= (b * dest_scaled)[np.newaxis, :]

    # Clean tiny values
    trips[trips < 1e-6] = 0.0

    # Ensure exact origin conservation
    row_sums = np.sum(trips, axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    trips *= origins_f32[:, np.newaxis] / row_sums

    logger.info(
        "Gravity distribution complete. Total distributed trips: %s", f"{np.sum(trips):,.0f}"
    )
    return trips
