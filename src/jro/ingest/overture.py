"""Activity and employment attractor modeling for H3 cells."""

import geopandas as gpd
import numpy as np
import pandas as pd

from jro.logging_setup import logger

# Canonical high-density activity & employment epicenters across Jabodetabek
EMPLOYMENT_CENTERS = pd.DataFrame(
    [
        {
            "name": "Sudirman-Thamrin CBD",
            "latitude": -6.2088,
            "longitude": 106.8227,
            "weight": 10.0,
            "kabkota_code": 3171,
        },
        {
            "name": "Kuningan Rasuna Said",
            "latitude": -6.2235,
            "longitude": 106.8315,
            "weight": 8.5,
            "kabkota_code": 3174,
        },
        {
            "name": "Gatot Subroto Corridor",
            "latitude": -6.2390,
            "longitude": 106.8250,
            "weight": 7.0,
            "kabkota_code": 3174,
        },
        {
            "name": "TB Simatupang Corridor",
            "latitude": -6.3025,
            "longitude": 106.8260,
            "weight": 6.5,
            "kabkota_code": 3174,
        },
        {
            "name": "Kelapa Gading Commercial",
            "latitude": -6.1585,
            "longitude": 106.9080,
            "weight": 5.0,
            "kabkota_code": 3172,
        },
        {
            "name": "Tanjung Priok Port Zone",
            "latitude": -6.1150,
            "longitude": 106.8850,
            "weight": 6.0,
            "kabkota_code": 3172,
        },
        {
            "name": "Grogol - Tomang Hub",
            "latitude": -6.1700,
            "longitude": 106.7900,
            "weight": 5.5,
            "kabkota_code": 3173,
        },
        {
            "name": "Puri Indah Business Park",
            "latitude": -6.1870,
            "longitude": 106.7350,
            "weight": 4.5,
            "kabkota_code": 3173,
        },
        {
            "name": "Pulo Gadung Industrial",
            "latitude": -6.1950,
            "longitude": 106.9150,
            "weight": 5.5,
            "kabkota_code": 3175,
        },
        {
            "name": "Cililitan / Cawang Hub",
            "latitude": -6.2580,
            "longitude": 106.8680,
            "weight": 5.0,
            "kabkota_code": 3175,
        },
        {
            "name": "BSD City CBD",
            "latitude": -6.3030,
            "longitude": 106.6540,
            "weight": 5.5,
            "kabkota_code": 3674,
        },
        {
            "name": "Gading Serpong Commercial",
            "latitude": -6.2420,
            "longitude": 106.6280,
            "weight": 4.5,
            "kabkota_code": 3603,
        },
        {
            "name": "Tangerang Kota CBD",
            "latitude": -6.1780,
            "longitude": 106.6320,
            "weight": 4.0,
            "kabkota_code": 3671,
        },
        {
            "name": "Margonda Depok Hub",
            "latitude": -6.3720,
            "longitude": 106.8320,
            "weight": 4.5,
            "kabkota_code": 3276,
        },
        {
            "name": "Bekasi Barat CBD",
            "latitude": -6.2360,
            "longitude": 106.9920,
            "weight": 5.0,
            "kabkota_code": 3275,
        },
        {
            "name": "Cikarang Industrial Estate (Jababeka/EJIP)",
            "latitude": -6.3250,
            "longitude": 107.1450,
            "weight": 8.0,
            "kabkota_code": 3216,
        },
        {
            "name": "Bogor Pajajaran Corridor",
            "latitude": -6.5920,
            "longitude": 106.8080,
            "weight": 4.0,
            "kabkota_code": 3271,
        },
        {
            "name": "Cibinong / Sentul Industrial",
            "latitude": -6.5050,
            "longitude": 106.8520,
            "weight": 3.5,
            "kabkota_code": 3201,
        },
    ]
)


class ActivityAttractorModel:
    """Computes employment and activity attraction scores per H3 cell."""

    def compute_attraction_scores(
        self,
        grid_gdf: gpd.GeoDataFrame,
        od_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculates attraction scores for each cell scaled to BPS destination totals.

        Args:
            grid_gdf: GeoDataFrame of H3 cells with ['cell_id', 'latitude', 'longitude', 'kabkota_code'].
            od_matrix: BPS commuter OD DataFrame with ['destination_code', 'commuter_flow_daily'].

        Returns:
            DataFrame with columns: ['cell_id', 'attraction_score']
        """
        logger.info("Computing activity attractor scores for %d cells", len(grid_gdf))

        cell_lats = grid_gdf["latitude"].to_numpy()
        cell_lons = grid_gdf["longitude"].to_numpy()

        center_lats = EMPLOYMENT_CENTERS["latitude"].to_numpy()
        center_lons = EMPLOYMENT_CENTERS["longitude"].to_numpy()
        center_weights = EMPLOYMENT_CENTERS["weight"].to_numpy()

        # Compute attraction from each employment epicenter: weight * exp(-d^2 / 2*sigma^2)
        # sigma = ~0.04 degrees (~4.5 km)
        sigma_sq = 2 * (0.04**2)

        # Distance matrix: (N_cells, N_centers)
        d_lat = cell_lats[:, np.newaxis] - center_lats[np.newaxis, :]
        d_lon = cell_lons[:, np.newaxis] - center_lons[np.newaxis, :]
        dist_sq = d_lat**2 + d_lon**2

        kernel = np.exp(-dist_sq / sigma_sq)  # shape (N_cells, N_centers)
        raw_attraction = np.dot(kernel, center_weights) + 0.1  # baseline non-zero

        grid_with_attr = grid_gdf.copy()
        grid_with_attr["raw_attr"] = raw_attraction

        # Scale attraction within each destination regency to match total BPS arrivals
        bps_arrivals = (
            od_matrix.groupby("destination_code")["commuter_flow_daily"]
            .sum()
            .reset_index()
            .rename(columns={"commuter_flow_daily": "target_arrivals"})
        )

        records = []
        for kab_code, group in grid_with_attr.groupby("kabkota_code"):
            match = bps_arrivals[bps_arrivals["destination_code"] == kab_code]
            target = (
                float(match["target_arrivals"].iloc[0])
                if not match.empty
                else float(group["raw_attr"].sum() * 1000)
            )

            total_raw = float(group["raw_attr"].sum())
            if total_raw <= 0:
                total_raw = 1.0

            scaled_scores = (group["raw_attr"] / total_raw) * target
            for cell_id, score in zip(group["cell_id"], scaled_scores):
                records.append({"cell_id": cell_id, "attraction_score": float(score)})

        attr_df = pd.DataFrame(records)
        logger.info(
            "Attractor modeling complete. Total attraction volume: %s",
            f"{attr_df['attraction_score'].sum():,.0f}",
        )
        return attr_df
