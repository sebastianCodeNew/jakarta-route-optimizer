"""Typer CLI command interface for Jakarta Route Optimizer (JRO)."""

import os

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from jro.accessibility.baseline import compute_baseline_matrices
from jro.accessibility.metrics import compute_cumulative_accessibility
from jro.accessibility.transit_desert import compute_transit_desert_scores
from jro.config import load_config
from jro.congestion.collector import TomTomCongestionCollector
from jro.congestion.profile import build_congestion_profile
from jro.congestion.sampling_plan import get_sampling_probe_points
from jro.demand.attractors import ActivityAttractorModel
from jro.demand.generation import compute_trip_production
from jro.demand.gravity import doubly_constrained_gravity
from jro.demand.ipf import apply_biproportional_ipf
from jro.exceptions import DataSourceError, JroError
from jro.ingest.bps import BpsDataLoader
from jro.ingest.gtfs_tj import TransJakartaGtfsIngestor
from jro.ingest.ridership import RidershipDataLoader
from jro.ingest.worldpop import WorldPopIngestor
from jro.io.manifest import ManifestManager
from jro.io.paths import DataPaths
from jro.io.storage import (
    load_numpy,
    load_parquet,
    save_geodataframe,
    save_numpy,
    save_parquet,
)
from jro.network.gtfs_merge import merge_gtfs_feeds
from jro.network.gtfs_validate import validate_gtfs_feed
from jro.network.rail_gtfs_builder import build_synthetic_rail_gtfs
from jro.optimize.robustness import run_monte_carlo_robustness
from jro.optimize.search import run_route_optimization
from jro.report.render import render_report_bundle
from jro.validation.backtest import (
    predict_validation_corridors_from_model,
    run_leave_one_route_out_backtest,
)
from jro.zoning.h3_grid import assign_regency_by_proximity, generate_study_area_grid

app = typer.Typer(
    name="jro",
    help="Jakarta Route Optimizer: Data-driven public transit route recommendation system for Greater Jakarta.",
    add_completion=False,
)
console = Console()


@app.command()
def ingest(
    target: str = typer.Argument(
        "all", help="Target dataset to ingest ('all', 'gtfs', 'bps', 'rail')"
    ),
    config_dir: str = typer.Option("config", "--config", help="Configuration directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download even if cached"),
) -> None:
    """Ingests public datasets (TransJakarta GTFS, BPS statistics, and synthetic Rail GTFS)."""
    cfg = load_config(config_dir)
    paths = DataPaths.from_root(cfg.settings.data_dir)
    manifest = ManifestManager(paths.base / "manifest.json")

    console.print(f"[bold green]Starting ingestion for target:[/bold green] {target}")

    if target in ("all", "bps"):
        loader = BpsDataLoader(paths.raw)
        labor_df = loader.load_labor_data()
        od_df = loader.load_od_matrix()
        console.print(f"Loaded BPS Labor: {len(labor_df)} zones, OD pairs: {len(od_df)}")

    if target in ("all", "rail"):
        rail_out = paths.raw / "gtfs_rail" / "gtfs_rail.zip"
        build_synthetic_rail_gtfs(cfg.rail, rail_out)
        manifest.record("raw_rail_gtfs", rail_out, license_str="Synthetic ODbL/KAI")
        console.print(f"Synthesized Rail GTFS at: {rail_out}")

    if target in ("all", "gtfs"):
        tj_ingestor = TransJakartaGtfsIngestor(paths.raw, manifest)
        try:
            zip_path = tj_ingestor.ingest(force_download=force)
            console.print(f"TransJakarta GTFS verified at: {zip_path}")
        except (DataSourceError, JroError, OSError) as e:
            console.print(
                f"[yellow]Note: Remote TransJakarta GTFS fetch warning ({e}). Ensuring fallback synthetic feed.[/yellow]"
            )


@app.command(name="build-network")
def build_network(
    config_dir: str = typer.Option("config", "--config", help="Configuration directory"),
) -> None:
    """Merges and validates TransJakarta and Rail GTFS feeds."""
    cfg = load_config(config_dir)
    paths = DataPaths.from_root(cfg.settings.data_dir)

    tj_zip = paths.raw / "gtfs_tj" / "transjakarta_gtfs.zip"
    rail_zip = paths.raw / "gtfs_rail" / "gtfs_rail.zip"

    if not rail_zip.exists():
        build_synthetic_rail_gtfs(cfg.rail, rail_zip)

    # If TJ zip is not present, use rail zip as base feed
    base_zip = tj_zip if tj_zip.exists() else rail_zip
    merged_zip = paths.interim / "gtfs_merged.zip"

    merge_gtfs_feeds(base_zip, rail_zip, merged_zip)
    report = validate_gtfs_feed(merged_zip)

    console.print(f"[bold green]Network feed created:[/bold green] {merged_zip}")
    console.print(
        f"Validation summary: Valid={report['is_valid']}, Stops={report['stats'].get('stops_count')}, Routes={report['stats'].get('routes_count')}"
    )


@app.command(name="demand")
def demand_pipeline(
    calibrate: bool = typer.Option(
        True, "--calibrate/--no-calibrate", help="Calibrate IPF against BPS OD matrix"
    ),
    config_dir: str = typer.Option("config", "--config", help="Configuration directory"),
) -> None:
    """Builds H3 spatial grid, commuter trip production, gravity distribution, and IPF calibration."""
    cfg = load_config(config_dir)
    paths = DataPaths.from_root(cfg.settings.data_dir)

    console.print("[bold blue]1. Generating H3 Hexagonal Grid...[/bold blue]")
    grid_gdf = generate_study_area_grid(
        cfg.settings.study_area, resolution=cfg.settings.h3_resolution
    )

    bps_loader = BpsDataLoader(paths.raw)
    labor_df = bps_loader.load_labor_data()
    od_df = bps_loader.load_od_matrix()

    # Assign regencies
    reg_centers = labor_df[["kabkota_code", "kabkota_name"]].copy()
    from jro.zoning.h3_grid import REGENCY_CENTERS_DATA

    reg_centers = reg_centers.merge(
        REGENCY_CENTERS_DATA[["kabkota_code", "latitude", "longitude"]], on="kabkota_code"
    )
    grid_gdf = assign_regency_by_proximity(grid_gdf, reg_centers)

    console.print("[bold blue]2. Distributing Population and Commuter Production...[/bold blue]")
    wp = WorldPopIngestor(paths.raw, ManifestManager(paths.base / "manifest.json"))
    pop_df = wp.ingest_or_generate_h3_population(grid_gdf, labor_df)
    grid_with_pop = grid_gdf.merge(pop_df, on="cell_id")

    prod_df = compute_trip_production(grid_with_pop, labor_df)
    grid_with_pop = grid_with_pop.merge(prod_df, on="cell_id")

    console.print("[bold blue]3. Modeling Employment and Activity Attractors...[/bold blue]")
    attractor_model = ActivityAttractorModel()
    attr_df = attractor_model.compute_attraction_scores(grid_with_pop, od_df)
    grid_final = grid_with_pop.merge(attr_df, on="cell_id")

    # Retain active settlement and commuter activity cells
    active_mask = (grid_final["production"] > 25.0) | (grid_final["attraction_score"] > 25.0)
    grid_final = grid_final[active_mask].reset_index(drop=True)
    save_geodataframe(grid_final, paths.processed / "h3_cells_demand.parquet")

    console.print(
        f"[bold blue]4. Running Gravity Model & IPF Calibration for {len(grid_final)} active cells...[/bold blue]"
    )
    n_cells = len(grid_final)
    cell_coords = list(zip(grid_final["latitude"], grid_final["longitude"]))
    lats = np.array([c[0] for c in cell_coords], dtype=np.float32)
    lons = np.array([c[1] for c in cell_coords], dtype=np.float32)
    lat_diff = (lats[:, np.newaxis] - lats[np.newaxis, :]) * 111.0
    lon_diff = (lons[:, np.newaxis] - lons[np.newaxis, :]) * 110.5
    dist_km = np.sqrt(lat_diff**2 + lon_diff**2, dtype=np.float32)
    del lat_diff, lon_diff
    np.fill_diagonal(dist_km, 0.5)

    travel_time_approx = (dist_km / 25.0) * 60.0
    origins = grid_final["production"].to_numpy(dtype=np.float32)
    destinations = grid_final["attraction_score"].to_numpy(dtype=np.float32)

    gravity_od = doubly_constrained_gravity(
        origins=origins,
        destinations=destinations,
        cost_matrix=travel_time_approx,
        beta=cfg.model.gravity.beta,
    )

    if calibrate:
        bps_targets = {
            (int(row["origin_code"]), int(row["destination_code"])): float(
                row["commuter_flow_daily"]
            )
            for _, row in od_df.iterrows()
        }
        orig_regs = list(grid_final["kabkota_code"])
        dest_regs = list(grid_final["kabkota_code"])
        calibrated_od = apply_biproportional_ipf(
            od_matrix=gravity_od,
            origin_regency_indices=orig_regs,
            dest_regency_indices=dest_regs,
            bps_regency_targets=bps_targets,
        )
    else:
        calibrated_od = gravity_od

    save_numpy(calibrated_od, paths.processed / "od_matrix.npy")
    console.print(
        f"[bold green]Demand pipeline complete:[/bold green] {n_cells} cells, total flows: {np.sum(calibrated_od):,.0f} trips/day"
    )


@app.command(name="congestion")
def congestion_cli(
    action: str = typer.Argument(
        "profile", help="'run' (probe TomTom API) or 'profile' (build speed ratio profile)"
    ),
    config_dir: str = typer.Option("config", "--config", help="Configuration directory"),
) -> None:
    """Manages TomTom traffic flow observation daemon and speed profiling."""
    cfg = load_config(config_dir)
    paths = DataPaths.from_root(cfg.settings.data_dir)
    db_path = paths.processed / "congestion_budget.db"

    if action == "run":
        api_key = os.getenv("TOMTOM_API_KEY", "")
        collector = TomTomCongestionCollector(api_key=api_key, db_path=db_path)
        probes = get_sampling_probe_points()
        console.print(
            f"Sampling {len(probes)} corridor probe points with TomTom API (Budget used today: {collector.tracker.get_today_used()})..."
        )

        for _, p in probes.iterrows():
            if not collector.running:
                break
            collector.fetch_flow_segment(
                point_id=str(p["point_id"]),
                lat=float(p["latitude"]),
                lon=float(p["longitude"]),
                highway_class=str(p["highway_class"]),
            )
        console.print(
            f"Probing cycle complete. Requests used today: {collector.tracker.get_today_used()}"
        )

    elif action == "profile":
        profile = build_congestion_profile(db_path)
        console.print(
            f"[bold green]Congestion profile ready (Source: {profile.source}):[/bold green]"
        )
        for hw, ratio in profile.speed_ratio_by_class.items():
            console.print(f"  {hw:12s}: speed ratio = {ratio:.2f}")


@app.command(name="baseline")
def baseline_cli(
    force: bool = typer.Option(
        False, "--force", "-f", help="Force recomputation of baseline matrices"
    ),
    config_dir: str = typer.Option("config", "--config", help="Configuration directory"),
) -> None:
    """Computes baseline transit and private travel times and identifies transit deserts."""
    cfg = load_config(config_dir)
    paths = DataPaths.from_root(cfg.settings.data_dir)

    cells_file = paths.processed / "h3_cells_demand.parquet"
    if not cells_file.exists():
        console.print("[yellow]Demand file not found. Running demand pipeline first...[/yellow]")
        demand_pipeline(calibrate=True, config_dir=config_dir)

    import zipfile

    import geopandas as gpd
    import pandas as pd

    cells_gdf = gpd.read_parquet(cells_file)
    cell_coords = list(zip(cells_gdf["latitude"], cells_gdf["longitude"]))

    # Load stops from merged GTFS if available, otherwise fallback to station anchors
    gtfs_zip = paths.interim / "gtfs_merged.zip"
    transit_stops: list[tuple[float, float, str]] = []
    if gtfs_zip.exists():
        with zipfile.ZipFile(gtfs_zip, "r") as zf:
            if "stops.txt" in zf.namelist():
                with zf.open("stops.txt") as f:
                    stops_df = pd.read_csv(f)
                    for _, row in stops_df.iterrows():
                        transit_stops.append(
                            (float(row["stop_lat"]), float(row["stop_lon"]), str(row["stop_id"]))
                        )

    if not transit_stops:
        from jro.candidates.anchors import TRANSIT_HUBS

        transit_stops = [(h.latitude, h.longitude, h.anchor_id) for h in TRANSIT_HUBS]

    profile = build_congestion_profile(paths.processed / "congestion_budget.db")
    matrices = compute_baseline_matrices(
        cell_coords=cell_coords,
        transit_stops=transit_stops,
        congestion_profile=profile,
        cache_dir=paths.processed,
        force_recompute=force,
    )

    console.print("[bold blue]Computing accessibility and transit desert indices...[/bold blue]")
    opportunities = cells_gdf["attraction_score"].to_numpy()
    transit_acc = compute_cumulative_accessibility(
        matrices.transit_travel_time, opportunities, cutoff_minutes=60.0
    )

    desert_df = compute_transit_desert_scores(
        cell_ids=list(cells_gdf["cell_id"]),
        population=cells_gdf["population"].to_numpy(),
        production=cells_gdf["production"].to_numpy(),
        transit_accessibility=transit_acc,
    )
    save_parquet(desert_df, paths.processed / "transit_deserts.parquet")
    console.print(
        f"[bold green]Baseline matrices and transit desert index saved to {paths.processed}[/bold green]"
    )


@app.command(name="optimize")
def optimize_cli(
    fleet_size: int = typer.Option(
        12, "--fleet-size", "-n", help="Default fleet size for initial ranking"
    ),
    samples: int = typer.Option(
        100, "--samples", "-s", help="Monte Carlo parameter perturbation samples"
    ),
    run_id: str = typer.Option("latest", "--run-id", help="Identifier for output directory"),
    config_dir: str = typer.Option("config", "--config", help="Configuration directory"),
) -> None:
    """Evaluates all candidate routes across Mikrotrans and Non-BRT, runs Monte Carlo robustness, and generates output report."""
    cfg = load_config(config_dir)
    paths = DataPaths.from_root(cfg.settings.data_dir)

    # Ensure baseline and demand matrices exist
    base_tr_file = paths.processed / "base_transit_tt.npy"
    if not base_tr_file.exists():
        baseline_cli(config_dir=config_dir)

    import geopandas as gpd

    cells_gdf = gpd.read_parquet(paths.processed / "h3_cells_demand.parquet")
    desert_df = load_parquet(paths.processed / "transit_deserts.parquet")
    od_demand = load_numpy(paths.processed / "od_matrix.npy")

    base_transit_tt = load_numpy(paths.processed / "base_transit_tt.npy")
    base_car_tt = load_numpy(paths.processed / "base_car_tt.npy")
    base_mc_tt = load_numpy(paths.processed / "base_mc_tt.npy")
    dist_km = load_numpy(paths.processed / "base_dist_km.npy")

    cell_ids = list(cells_gdf["cell_id"])
    desert_cells = set(desert_df[desert_df["is_desert"]]["cell_id"])

    console.print(
        "[bold cyan]1. Executing route search and multi-profile evaluation...[/bold cyan]"
    )
    summary = run_route_optimization(
        base_transit_tt=base_transit_tt,
        base_mc_tt=base_mc_tt,
        base_car_tt=base_car_tt,
        dist_km=dist_km,
        od_demand=od_demand,
        cell_ids=cell_ids,
        transit_desert_cells=desert_cells,
        config=cfg,
        default_fleet_size=fleet_size,
    )

    console.print(
        f"[bold green]Top Candidates Evaluated: {len(summary.ranked_candidates)}[/bold green]"
    )
    console.print(
        f"Overall Winner: [bold yellow]{summary.overall_winner.route.route_name}[/bold yellow] ({summary.overall_winner.profile_name.upper()})"
    )

    console.print(
        f"[bold cyan]2. Running Monte Carlo robustness analysis ({samples} iterations)...[/bold cyan]"
    )
    robustness_records = run_monte_carlo_robustness(
        top_candidates=summary.ranked_candidates[:5],
        base_transit_tt=base_transit_tt,
        base_mc_tt=base_mc_tt,
        base_car_tt=base_car_tt,
        dist_km=dist_km,
        od_demand=od_demand,
        cell_ids=cell_ids,
        config=cfg,
        num_samples=samples,
    )

    console.print("[bold cyan]3. Executing empirical validation backtest...[/bold cyan]")
    bps_loader = RidershipDataLoader(paths.raw)
    val_df = bps_loader.load_validation_ridership()
    pred_map = predict_validation_corridors_from_model(
        validation_df=val_df,
        od_matrix=od_demand,
        cells_df=cells_gdf,
        gtfs_merged_zip=paths.interim / "gtfs_merged.zip",
    )
    backtest_res = run_leave_one_route_out_backtest(val_df, pred_map)

    console.print(
        "[bold cyan]4. Rendering final report, GTFS, GeoJSON, and metadata bundle...[/bold cyan]"
    )
    out_dir = render_report_bundle(
        opt_summary=summary,
        robustness_list=robustness_records,
        backtest=backtest_res,
        output_dir=paths.outputs,
        run_id=run_id,
    )

    console.print(f"\n[bold green]SUCCESS! Report bundle generated at:[/bold green] {out_dir}")
    console.print(f"Interactive Report: [underline]{out_dir / 'report.html'}[/underline]")
    console.print(
        f"Standard GTFS feed: [underline]{out_dir / 'recommended_route_gtfs.zip'}[/underline]"
    )
    console.print(
        f"GeoJSON Trase:     [underline]{out_dir / 'recommended_route.geojson'}[/underline]"
    )


@app.command(name="validate")
def validate_cli(
    target: str = typer.Argument("backtest", help="'backtest' or 'gtfs'"),
    config_dir: str = typer.Option("config", "--config", help="Configuration directory"),
) -> None:
    """Runs empirical backtesting or GTFS validation checks."""
    cfg = load_config(config_dir)
    paths = DataPaths.from_root(cfg.settings.data_dir)

    if target == "backtest":
        loader = RidershipDataLoader(paths.raw)
        val_df = loader.load_validation_ridership()
        cells_file = paths.processed / "h3_cells_demand.parquet"
        od_file = paths.processed / "od_matrix.npy"
        pred_map: dict[str, float] = {}
        if cells_file.exists() and od_file.exists():
            import geopandas as gpd

            cells_gdf = gpd.read_parquet(cells_file)
            od_demand = load_numpy(od_file)
            pred_map = predict_validation_corridors_from_model(
                validation_df=val_df,
                od_matrix=od_demand,
                cells_df=cells_gdf,
                gtfs_merged_zip=paths.interim / "gtfs_merged.zip",
            )
        res = run_leave_one_route_out_backtest(val_df, pred_map)

        table = Table(title="Leave-One-Route-Out Backtest Validation")
        table.add_column("Route ID")
        table.add_column("Corridor Name")
        table.add_column("Reported Ridership", justify="right")
        table.add_column("Predicted Ridership", justify="right")
        table.add_column("Error %", justify="right")

        for _, r in res.comparison_table.iterrows():
            table.add_row(
                str(r["route_id"]),
                str(r["route_name"]),
                f"{r['reported_ridership']:,.0f}",
                f"{r['predicted_ridership']:,.0f}",
                f"{r['percentage_error']:.1f}%",
            )
        console.print(table)
        console.print(
            f"[bold]Spearman Rho:[/bold] {res.spearman_rho} | [bold]Pearson r:[/bold] {res.pearson_r} | [bold]MAPE:[/bold] {res.mape_percentage}%"
        )


@app.command(name="report")
def report_cli(
    run_id: str = typer.Option("latest", "--run-id", help="Run identifier to inspect"),
    config_dir: str = typer.Option("config", "--config", help="Configuration directory"),
) -> None:
    """Inspects generated evaluation artifacts for a specific run."""
    cfg = load_config(config_dir)
    paths = DataPaths.from_root(cfg.settings.data_dir)
    target_dir = paths.outputs / run_id
    if not target_dir.exists():
        console.print(f"[red]Error: Run output directory not found at {target_dir}[/red]")
        raise typer.Exit(code=1)

    meta_file = target_dir / "run_metadata.json"
    if meta_file.exists():
        import json

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            winner = meta.get("overall_winner", {})
            console.print(f"[bold green]Run ID:[/bold green] {run_id}")
            console.print(
                f"[bold]Winning Route:[/bold] {winner.get('route_name')} ({winner.get('vehicle_profile')})"
            )
            console.print(f"[bold]Composite Score:[/bold] {winner.get('composite_score')}")
            console.print(
                f"[bold]Daily Riders:[/bold] {winner.get('metrics', {}).get('daily_riders_served'):,.0f} passengers/day"
            )
            console.print(
                f"[bold]VKT Reduced:[/bold] {winner.get('metrics', {}).get('total_private_vkt_reduced'):,.0f} km/day"
            )
            console.print(
                f"[bold]Net CO2 Saved:[/bold] {winner.get('metrics', {}).get('net_co2_kg_saved'):,.1f} kg/day"
            )
