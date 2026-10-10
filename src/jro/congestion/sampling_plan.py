"""Stratified sampling plan for corridor congestion probes."""

from typing import NamedTuple

import pandas as pd


class CongestionProbePoint(NamedTuple):
    """Observation probe coordinate on a specific road corridor."""

    point_id: str
    corridor_name: str
    highway_class: str
    latitude: float
    longitude: float
    kabkota_name: str


STRATIFIED_CORRIDOR_PROBES = [
    # Jakarta CBD & Radial Arterials
    CongestionProbePoint("P01", "Sudirman CBD", "primary", -6.2150, 106.8220, "Jakarta Pusat"),
    CongestionProbePoint("P02", "Thamrin", "primary", -6.1910, 106.8230, "Jakarta Pusat"),
    CongestionProbePoint(
        "P03", "Gatot Subroto Kuningan", "primary", -6.2390, 106.8270, "Jakarta Selatan"
    ),
    CongestionProbePoint("P04", "Rasuna Said", "primary", -6.2210, 106.8315, "Jakarta Selatan"),
    CongestionProbePoint(
        "P05", "TB Simatupang Fatmawati", "primary", -6.2940, 106.7960, "Jakarta Selatan"
    ),
    CongestionProbePoint(
        "P06", "TB Simatupang Cilandak", "primary", -6.3010, 106.8180, "Jakarta Selatan"
    ),
    CongestionProbePoint("P07", "S. Parman Slipi", "primary", -6.1930, 106.7980, "Jakarta Barat"),
    CongestionProbePoint(
        "P08", "Daan Mogot Cengkareng", "primary", -6.1550, 106.7200, "Jakarta Barat"
    ),
    CongestionProbePoint("P09", "MT Haryono Cawang", "primary", -6.2430, 106.8640, "Jakarta Timur"),
    CongestionProbePoint("P10", "Pemuda Rawamangun", "primary", -6.1950, 106.8850, "Jakarta Timur"),
    CongestionProbePoint(
        "P11", "Yos Sudarso Sunter", "primary", -6.1420, 106.8870, "Jakarta Utara"
    ),
    # Suburb-to-Core Cross-Border Corridors
    CongestionProbePoint(
        "P12", "Ciledug Raya Kreo", "secondary", -6.2330, 106.7450, "Kota Tangerang"
    ),
    CongestionProbePoint(
        "P13", "Daan Mogot Batu Ceper", "primary", -6.1660, 106.6650, "Kota Tangerang"
    ),
    CongestionProbePoint(
        "P14", "Serpong Raya BSD", "primary", -6.2750, 106.6620, "Kota Tangerang Selatan"
    ),
    CongestionProbePoint(
        "P15", "Ciputat Raya Lebak Bulus", "secondary", -6.3100, 106.7650, "Kota Tangerang Selatan"
    ),
    CongestionProbePoint("P16", "Margonda Raya Depok", "primary", -6.3750, 106.8320, "Kota Depok"),
    CongestionProbePoint("P17", "Cinere Raya", "secondary", -6.3260, 106.7820, "Kota Depok"),
    CongestionProbePoint(
        "P18", "Raya Bogor Cibinong", "primary", -6.4850, 106.8520, "Kabupaten Bogor"
    ),
    CongestionProbePoint("P19", "Pajajaran Bogor Kota", "primary", -6.5920, 106.8080, "Kota Bogor"),
    CongestionProbePoint(
        "P20", "Kalimalang Bekasi Barat", "secondary", -6.2480, 106.9850, "Kota Bekasi"
    ),
    CongestionProbePoint(
        "P21", "Juanda Bekasi Timur", "secondary", -6.2420, 107.0120, "Kota Bekasi"
    ),
    CongestionProbePoint(
        "P22", "Imam Bonjol Cikarang Barat", "primary", -6.2710, 107.0850, "Kabupaten Bekasi"
    ),
    CongestionProbePoint(
        "P23", "Urip Sumoharjo Cikarang", "primary", -6.2620, 107.1550, "Kabupaten Bekasi"
    ),
]


def get_sampling_probe_points() -> pd.DataFrame:
    """Returns DataFrame of standard congestion observation probe points."""
    return pd.DataFrame([p._asdict() for p in STRATIFIED_CORRIDOR_PROBES])
