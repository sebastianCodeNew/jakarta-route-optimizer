"""End-to-end integration test for optimization, robustness, backtest, and reporting."""

import numpy as np
import pandas as pd

from jro.optimize.robustness import run_monte_carlo_robustness
from jro.optimize.search import run_route_optimization
from jro.report.render import render_report_bundle
from jro.validation.backtest import run_leave_one_route_out_backtest


def test_e2e_optimization_and_reporting_pipeline(app_config, tmp_path) -> None:
    """Verifies complete end-to-end flow from matrices to final HTML/GTFS bundle."""
    # Collect candidate routes to obtain their actual stop cell IDs
    from jro.candidates.generator import generate_candidate_routes

    cand_sample = generate_candidate_routes(app_config.vehicles["mikrotrans"])
    sample_cells = []
    for c in cand_sample:
        for s in c.stops:
            if s.cell_id not in sample_cells:
                sample_cells.append(s.cell_id)

    cell_ids = sample_cells[:15] if len(sample_cells) >= 15 else sample_cells
    n = len(cell_ids)
    rng = np.random.default_rng(42)

    # Synthetic coordinates and distance matrix
    dist_km = rng.uniform(2.0, 20.0, size=(n, n))
    dist_km = (dist_km + dist_km.T) / 2.0
    np.fill_diagonal(dist_km, 0.5)

    base_transit_tt = dist_km * 3.5 + 15.0
    np.fill_diagonal(base_transit_tt, 5.0)

    base_car_tt = dist_km * 2.2 + 5.0
    np.fill_diagonal(base_car_tt, 2.0)

    base_mc_tt = dist_km * 1.8 + 3.0
    np.fill_diagonal(base_mc_tt, 2.0)

    od_demand = rng.uniform(200, 1500, size=(n, n))
    np.fill_diagonal(od_demand, 50.0)

    transit_deserts = set(cell_ids[:3])

    # 1. Run optimization search
    summary = run_route_optimization(
        base_transit_tt=base_transit_tt,
        base_mc_tt=base_mc_tt,
        base_car_tt=base_car_tt,
        dist_km=dist_km,
        od_demand=od_demand,
        cell_ids=cell_ids,
        transit_desert_cells=transit_deserts,
        config=app_config,
        default_fleet_size=10,
    )

    assert len(summary.ranked_candidates) > 0
    winner = summary.overall_winner
    assert winner.impact_score.composite_score > 0.0
    assert len(winner.fleet_curve) == 30

    # 2. Run Monte Carlo robustness
    rob_results = run_monte_carlo_robustness(
        top_candidates=summary.ranked_candidates[:3],
        base_transit_tt=base_transit_tt,
        base_mc_tt=base_mc_tt,
        base_car_tt=base_car_tt,
        dist_km=dist_km,
        od_demand=od_demand,
        cell_ids=cell_ids,
        config=app_config,
        num_samples=10,
    )
    assert len(rob_results) == min(3, len(summary.ranked_candidates))

    # 3. Run backtest
    val_df = pd.DataFrame(
        [
            {
                "route_id": "R1",
                "route_name": "Test Corridor 1",
                "service_type": "BRT",
                "daily_ridership_reported": 50000.0,
            },
            {
                "route_id": "R2",
                "route_name": "Test Corridor 2",
                "service_type": "Non-BRT",
                "daily_ridership_reported": 12000.0,
            },
        ]
    )
    backtest_res = run_leave_one_route_out_backtest(val_df, {})
    assert backtest_res.spearman_rho >= 0.5

    # 4. Render report bundle
    out_dir = render_report_bundle(
        opt_summary=summary,
        robustness_list=rob_results,
        backtest=backtest_res,
        output_dir=tmp_path / "outputs",
        run_id="test_run",
    )

    assert (out_dir / "report.html").exists()
    assert (out_dir / "recommended_route_map.html").exists()
    assert (out_dir / "recommended_route.geojson").exists()
    assert (out_dir / "recommended_route_gtfs.zip").exists()
    assert (out_dir / "run_metadata.json").exists()
