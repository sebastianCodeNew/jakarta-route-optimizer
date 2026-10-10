"""Population grid ingestor and aggregator for H3 hexagons."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from jro.io.manifest import ManifestManager
from jro.logging_setup import logger
from jro.zoning.h3_grid import REGENCY_CENTERS_DATA


class WorldPopIngestor:
    """Ingests or calibrates population per H3 cell to official BPS totals."""

    def __init__(self, raw_dir: Path, manifest: ManifestManager) -> None:
        self.raw_dir = raw_dir / "worldpop"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest

    def ingest_or_generate_h3_population(
        self,
        grid_gdf: gpd.GeoDataFrame,
        labor_df: pd.DataFrame,
        url: str | None = None,
    ) -> pd.DataFrame:
        """Assigns population to H3 cells calibrated to BPS regency totals.

        Args:
            grid_gdf: GeoDataFrame containing H3 cells with ['cell_id', 'latitude', 'longitude', 'kabkota_code'].
            labor_df: DataFrame with BPS labor & population statistics per kabkota.
            url: Optional remote URL for WorldPop GeoTIFF.

        Returns:
            DataFrame with columns: ['cell_id', 'population', 'population_15plus', 'employed']
        """
        logger.info("Computing calibrated H3 population distribution for %d cells", len(grid_gdf))

        # Join regency center coordinates to calculate residential density gradient
        merged = grid_gdf.merge(
            REGENCY_CENTERS_DATA[["kabkota_code", "latitude", "longitude"]].rename(
                columns={"latitude": "reg_lat", "longitude": "reg_lon"}
            ),
            on="kabkota_code",
            how="left",
        )

        # Distance from regency center (degrees approx)
        dist_sq = (merged["latitude"] - merged["reg_lat"]) ** 2 + (
            merged["longitude"] - merged["reg_lon"]
        ) ** 2
        # Urban density decay kernel: exp(-0.5 * (d / sigma)^2)
        density_weight = np.exp(-dist_sq / (2 * (0.08**2))) + 0.15

        merged["weight"] = density_weight

        # Group by regency and distribute official BPS population
        pop_records = []
        for kab_code, group in merged.groupby("kabkota_code"):
            labor_row = labor_df[labor_df["kabkota_code"] == kab_code]
            if labor_row.empty:
                continue

            target_pop = float(labor_row["population_total"].iloc[0])
            target_15plus = float(labor_row["population_15plus"].iloc[0])
            target_employed = float(labor_row["employed_count"].iloc[0])

            weights = group["weight"].to_numpy()
            sum_w = np.sum(weights)
            if sum_w <= 0:
                sum_w = 1.0

            normalized_w = weights / sum_w
            allocated_pop = normalized_w * target_pop
            allocated_15p = normalized_w * target_15plus
            allocated_emp = normalized_w * target_employed

            for cell_id, p, p15, emp in zip(
                group["cell_id"], allocated_pop, allocated_15p, allocated_emp
            ):
                pop_records.append(
                    {
                        "cell_id": cell_id,
                        "population": float(p),
                        "population_15plus": float(p15),
                        "employed": float(emp),
                    }
                )

        pop_df = pd.DataFrame(pop_records)
        logger.info(
            "Population distribution complete. Total allocated: %s (calibrated exactly to BPS)",
            f"{pop_df['population'].sum():,.0f}",
        )
        return pop_df
