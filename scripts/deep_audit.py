import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

print("=" * 65)
print("     JAKARTA ROUTE OPTIMIZER (JRO) - COMPREHENSIVE DATA & CODE AUDIT")
print("=" * 65)

# 1. RAW DATASETS AUDIT
print("\n[1] RAW INPUT DATASETS AUDIT:")
raw_checks = [
    ("BPS Commuter OD 2023", "data/raw/bps/bps_commuter_od_2023.csv"),
    ("BPS Sakernas Labor 2023", "data/raw/bps/bps_kabkota_labor.csv"),
    ("PPID / KAI Validation", "data/raw/bps/ridership_validation.csv"),
    ("TransJakarta Official GTFS", "data/raw/gtfs_tj/transjakarta_gtfs.zip"),
    ("Rail Network GTFS", "data/raw/gtfs_rail/gtfs_rail.zip"),
]

for label, p_str in raw_checks:
    p = Path(p_str)
    if p.exists():
        sz = p.stat().st_size
        print(f"  ✓ {label:28s} : {sz:10,d} bytes | Exists: True")
    else:
        print(f"  ✗ {label:28s} : MISSING!")

# 2. PROCESSED MATRICES AUDIT
print("\n[2] PROCESSED MATRICES INTEGRITY & REAL NUMBERS:")
proc_files = [
    ("H3 Cells Demand", "data/processed/h3_cells_demand.parquet"),
    ("Transit Deserts", "data/processed/transit_deserts.parquet"),
    ("OD Flow Matrix (trips)", "data/processed/od_matrix.npy"),
    ("Distance Matrix (km)", "data/processed/base_dist_km.npy"),
    ("Transit Travel Time (min)", "data/processed/base_transit_tt.npy"),
    ("Car Travel Time (min)", "data/processed/base_car_tt.npy"),
    ("Motorcycle Travel Time (min)", "data/processed/base_mc_tt.npy"),
]

for label, p_str in proc_files:
    p = Path(p_str)
    if not p.exists():
        print(f"  ✗ {label:28s} : MISSING!")
        continue
    sz = p.stat().st_size
    if p_str.endswith(".npy"):
        arr = np.load(p)
        nan_cnt = np.isnan(arr).sum()
        inf_cnt = np.isinf(arr).sum()
        print(
            f"  ✓ {label:28s} : {sz:10,d} bytes | shape={arr.shape} | "
            f"min={arr.min():.2f}, mean={arr.mean():.2f}, max={arr.max():.2f}, sum={arr.sum():,.0f} | "
            f"NaN={nan_cnt}, Inf={inf_cnt}"
        )
    elif p_str.endswith(".parquet"):
        df = pd.read_parquet(p)
        print(
            f"  ✓ {label:28s} : {sz:10,d} bytes | rows={len(df):,d}, cols={len(df.columns)} | "
            f"columns={list(df.columns)[:5]}..."
        )

# 3. TOMTOM CONGESTION DATABASE AUDIT
print("\n[3] TOMTOM CONGESTION PROBE DATABASE AUDIT:")
db_path = Path("data/processed/congestion_budget.db")
if db_path.exists():
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print(f"  ✓ SQLite database found at {db_path} with tables: {tables}")
    for t in tables:
        cnt = cur.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        print(f"     * Table '{t}': {cnt} records")
        if cnt > 0 and t == "probe_observations":
            sample = cur.execute(f"SELECT * FROM {t} LIMIT 3").fetchall()
            print(f"       Sample row: {sample[0]}")
    conn.close()
else:
    print("  ✗ Congestion database does not exist!")

# 4. ROAD NETWORK & GEOMETRIES AUDIT
print("\n[4] ROAD NETWORK (OPENSTREETMAP) GEOMETRIES AUDIT:")
geom_path = Path("config/corridor_road_geometries.json")
if geom_path.exists():
    with open(geom_path, "r", encoding="utf-8") as f:
        geoms = json.load(f)
    print(f"  ✓ Found real road geometry entries for {len(geoms)} corridors:")
    for cid, d in geoms.items():
        wps = d.get("waypoints", [])
        dist = d.get("distance_km", 0.0)
        print(f"     * {cid:32s} : {len(wps):4d} road nodes, {dist:6.2f} km")
else:
    print("  ✗ Corridor road geometries file missing!")

# 5. OFFICIAL RUN ARTIFACTS AUDIT
print("\n[5] OFFICIAL RUN ARTIFACTS VERIFICATION:")
official_dir = Path("data/outputs/official_run")
if official_dir.exists():
    meta_file = official_dir / "run_metadata.json"
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        winner = meta.get("overall_winner", {})
        print(f"  ✓ Metadata present for run_id: {meta.get('run_id')}")
        print(f"  ✓ Winning Corridor : {winner.get('route_name')} ({winner.get('vehicle_profile')})")
        print(f"  ✓ Optimal Fleet Size : {winner.get('optimal_fleet_size')} vehicles")
        print(f"  ✓ Composite Score  : {winner.get('composite_score')}")
        print(f"  ✓ Ridership Served : {winner.get('metrics', {}).get('daily_riders_served')} pax/day")
        print(f"  ✓ Net CO2 Saved    : {winner.get('metrics', {}).get('net_co2_kg_saved')} kg/day")
        print(f"  ✓ Backtest Spearman: rho = {meta.get('backtest_spearman_rho')}")
        print(f"  ✓ Backtest MAPE    : {meta.get('backtest_mape_percentage')}%")
    for fname in ["report.html", "recommended_route_gtfs.zip", "recommended_route.geojson", "recommended_route_map.html"]:
        f_p = official_dir / fname
        if f_p.exists():
            print(f"  ✓ Artifact file '{fname}': {f_p.stat().st_size:,} bytes")
        else:
            print(f"  ✗ Artifact file '{fname}' missing!")
else:
    print("  ✗ Official run output directory not found!")
