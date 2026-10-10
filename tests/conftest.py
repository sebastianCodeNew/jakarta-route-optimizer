"""Shared pytest fixtures for Jakarta Route Optimizer (JRO)."""

from pathlib import Path

import numpy as np
import pytest

from jro.config import AppConfig, load_config
from jro.demand.mode_choice import MultinomialLogitModeChoice


@pytest.fixture
def config_dir() -> Path:
    """Path to repository config directory."""
    return Path("config")


@pytest.fixture
def app_config(config_dir: Path) -> AppConfig:
    """Loaded and validated application configuration."""
    return load_config(config_dir)


@pytest.fixture
def mode_choice_model(app_config: AppConfig) -> MultinomialLogitModeChoice:
    """Configured MultinomialLogitModeChoice instance."""
    return MultinomialLogitModeChoice(app_config.model)


@pytest.fixture
def sample_od_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Small 5x5 test matrices for fast vector testing."""
    n = 5
    rng = np.random.default_rng(42)

    dist_km = np.array(
        [
            [0.5, 4.2, 7.5, 12.0, 15.5],
            [4.2, 0.5, 3.8, 8.5, 11.2],
            [7.5, 3.8, 0.5, 5.0, 8.0],
            [12.0, 8.5, 5.0, 0.5, 4.0],
            [15.5, 11.2, 8.0, 4.0, 0.5],
        ]
    )

    base_transit_tt = dist_km * 3.5 + 15.0  # slower with wait/walk
    np.fill_diagonal(base_transit_tt, 5.0)

    base_car_tt = dist_km * 2.2 + 5.0
    np.fill_diagonal(base_car_tt, 2.0)

    base_mc_tt = dist_km * 1.8 + 3.0
    np.fill_diagonal(base_mc_tt, 2.0)

    od_demand = rng.uniform(50, 500, size=(n, n))
    np.fill_diagonal(od_demand, 20.0)

    return base_transit_tt, base_mc_tt, base_car_tt, dist_km, od_demand
