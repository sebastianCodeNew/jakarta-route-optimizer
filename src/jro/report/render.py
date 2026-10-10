"""Renders final evaluation report, GeoJSON exports, and metadata bundle."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import jinja2
from shapely.geometry import LineString, Point

from jro.io.storage import save_geodataframe, save_json
from jro.logging_setup import logger
from jro.optimize.robustness import RouteRobustnessMetrics
from jro.optimize.search import OptimizationSummary
from jro.report.export_gtfs import export_route_to_gtfs
from jro.report.maps import generate_interactive_route_map
from jro.validation.backtest import BacktestResult


def render_report_bundle(
    opt_summary: OptimizationSummary,
    robustness_list: list[RouteRobustnessMetrics],
    backtest: BacktestResult,
    output_dir: Path,
    run_id: str = "run_latest",
) -> Path:
    """Generates complete report directory with HTML, maps, GeoJSON, and GTFS.

    Args:
        opt_summary: OptimizationSummary containing ranked routes and winners.
        robustness_list: List of Monte Carlo robustness records.
        backtest: Leave-one-route-out validation results.
        output_dir: Base directory where output run folder will be created.
        run_id: Unique identifier for this evaluation run.

    Returns:
        Path to output run directory.
    """
    logger.info("Rendering report bundle for run_id=%s into %s", run_id, output_dir)
    run_dir = (output_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    winner = opt_summary.overall_winner

    # 1. Generate interactive Folium map
    map_path = run_dir / "recommended_route_map.html"
    generate_interactive_route_map(
        route=winner.route,
        output_html_path=map_path,
        title=f"JRO #{run_id} - {winner.route.route_name}",
    )

    # 2. Export GTFS archive
    gtfs_path = run_dir / "recommended_route_gtfs.zip"
    export_route_to_gtfs(
        route=winner.route,
        operations=winner.operations,
        output_zip_path=gtfs_path,
    )

    # 3. Export GeoJSON for recommended route and top candidates
    # Winner route GeoJSON
    line_geom = LineString([(wp[1], wp[0]) for wp in winner.route.waypoints])
    winner_gdf = gpd.GeoDataFrame(
        [
            {
                "route_id": winner.route.route_id,
                "route_name": winner.route.route_name,
                "vehicle_profile": winner.profile_name,
                "length_km": winner.route.length_km,
                "optimal_fleet_size": winner.optimal_fleet_size,
                "daily_riders_served": winner.eval_result.daily_riders_served,
                "total_private_vkt_reduced": winner.emissions.total_private_vkt_reduced,
                "net_co2_kg_saved": winner.emissions.net_co2_kg_saved,
                "composite_score": winner.impact_score.composite_score,
            }
        ],
        geometry=[line_geom],
        crs="EPSG:4326",
    )
    save_geodataframe(winner_gdf, run_dir / "recommended_route.geojson", driver="GeoJSON")

    # Stops GeoJSON
    stops_records = []
    stops_geoms = []
    for s in winner.route.stops:
        stops_records.append(
            {
                "stop_id": s.stop_id,
                "stop_name": s.stop_name,
                "sequence": s.sequence,
                "cell_id": s.cell_id,
            }
        )
        stops_geoms.append(Point(s.longitude, s.latitude))
    stops_gdf = gpd.GeoDataFrame(stops_records, geometry=stops_geoms, crs="EPSG:4326")
    save_geodataframe(stops_gdf, run_dir / "recommended_stops.geojson", driver="GeoJSON")

    # Top-K candidate routes GeoJSON
    top_k_records = []
    top_k_geoms = []
    for r in opt_summary.ranked_candidates[:10]:
        top_k_records.append(
            {
                "route_id": r.route.route_id,
                "route_name": r.route.route_name,
                "profile": r.profile_name,
                "length_km": r.route.length_km,
                "composite_score": r.impact_score.composite_score,
                "daily_riders": r.eval_result.daily_riders_served,
                "net_co2_kg": r.emissions.net_co2_kg_saved,
            }
        )
        top_k_geoms.append(LineString([(wp[1], wp[0]) for wp in r.route.waypoints]))
    top_k_gdf = gpd.GeoDataFrame(top_k_records, geometry=top_k_geoms, crs="EPSG:4326")
    save_geodataframe(top_k_gdf, run_dir / "top_k_routes.geojson", driver="GeoJSON")

    # 4. Render HTML Report with Jinja2
    template_dir = Path(__file__).parent / "templates"
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )
    template = jinja_env.get_template("report.html.j2")

    waypoints_json = json.dumps([{"lat": wp[0], "lng": wp[1]} for wp in winner.route.waypoints])
    stops_json = json.dumps(
        [
            {
                "name": s.stop_name,
                "seq": s.sequence,
                "lat": s.latitude,
                "lng": s.longitude,
            }
            for s in winner.route.stops
        ]
    )

    html_content = template.render(
        run_id=run_id,
        timestamp=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        winner=winner,
        best_mikro=opt_summary.best_mikrotrans,
        best_nonbrt=opt_summary.best_non_brt,
        total_candidates_evaluated=len(opt_summary.ranked_candidates),
        robustness_list=robustness_list,
        backtest=backtest,
        waypoints_json=waypoints_json,
        stops_json=stops_json,
    )

    report_html_path = run_dir / "report.html"
    with open(report_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # 5. Metadata JSON
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "overall_winner": {
            "route_id": winner.route.route_id,
            "route_name": winner.route.route_name,
            "vehicle_profile": winner.profile_name,
            "optimal_fleet_size": winner.optimal_fleet_size,
            "length_km": winner.route.length_km,
            "stops_count": len(winner.route.stops),
            "composite_score": winner.impact_score.composite_score,
            "metrics": {
                "daily_riders_served": winner.eval_result.daily_riders_served,
                "total_private_vkt_reduced": winner.emissions.total_private_vkt_reduced,
                "net_co2_kg_saved": winner.emissions.net_co2_kg_saved,
                "net_pm25_kg_saved": winner.emissions.net_pm25_kg_saved,
                "headway_minutes": winner.operations.headway_minutes,
                "daily_operating_cost_idr": winner.operations.daily_operating_cost_idr,
            },
        },
        "best_mikrotrans": {
            "route_id": opt_summary.best_mikrotrans.route.route_id,
            "route_name": opt_summary.best_mikrotrans.route.route_name,
            "composite_score": opt_summary.best_mikrotrans.impact_score.composite_score,
        },
        "best_non_brt": {
            "route_id": opt_summary.best_non_brt.route.route_id,
            "route_name": opt_summary.best_non_brt.route.route_name,
            "composite_score": opt_summary.best_non_brt.impact_score.composite_score,
        },
        "backtest_spearman_rho": backtest.spearman_rho,
        "backtest_mape_percentage": backtest.mape_percentage,
    }
    save_json(metadata, run_dir / "run_metadata.json")

    logger.info("Report bundle successfully generated at %s", run_dir)
    return run_dir
