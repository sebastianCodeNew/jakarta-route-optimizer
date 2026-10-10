"""Transit hub anchors and high-demand passenger centroids."""

from typing import NamedTuple

import pandas as pd


class TransitAnchor(NamedTuple):
    """High-capacity transit network connection hub."""

    anchor_id: str
    name: str
    hub_type: str  # "rail", "brt", "multimodal"
    latitude: float
    longitude: float
    kabkota_name: str


class DemandAnchor(NamedTuple):
    """High-density passenger origin/activity cluster."""

    anchor_id: str
    name: str
    demand_type: str  # "residential_desert", "mixed_corridor", "commercial"
    latitude: float
    longitude: float
    kabkota_name: str


TRANSIT_HUBS = [
    TransitAnchor(
        "HUB_DUKUH_ATAS", "Dukuh Atas Multimodal", "multimodal", -6.2008, 106.8228, "Jakarta Pusat"
    ),
    TransitAnchor(
        "HUB_MANGGARAI", "Manggarai Station", "rail", -6.2099, 106.8501, "Jakarta Selatan"
    ),
    TransitAnchor(
        "HUB_LEBAK_BULUS", "Lebak Bulus Grab", "multimodal", -6.2891, 106.7745, "Jakarta Selatan"
    ),
    TransitAnchor("HUB_BLOK_M", "Blok M Hub", "multimodal", -6.2443, 106.7981, "Jakarta Selatan"),
    TransitAnchor(
        "HUB_TANAH_ABANG", "Tanah Abang Station", "rail", -6.1856, 106.8111, "Jakarta Pusat"
    ),
    TransitAnchor(
        "HUB_KP_RAMBUTAN",
        "Kampung Rambutan Terminal",
        "multimodal",
        -6.3131,
        106.8761,
        "Jakarta Timur",
    ),
    TransitAnchor(
        "HUB_KALIDERES", "Kalideres Terminal", "multimodal", -6.1528, 106.7028, "Jakarta Barat"
    ),
    TransitAnchor(
        "HUB_PULO_GEBANG",
        "Pulo Gebang Integrated Terminal",
        "multimodal",
        -6.2128,
        106.9536,
        "Jakarta Timur",
    ),
    TransitAnchor(
        "HUB_CISAUK", "Cisauk Intermodal Hub", "rail", -6.3236, 106.6436, "Kabupaten Tangerang"
    ),
    TransitAnchor(
        "HUB_JURANGMANGU",
        "Jurangmangu Bintaro",
        "rail",
        -6.2897,
        106.7269,
        "Kota Tangerang Selatan",
    ),
    TransitAnchor("HUB_DEPOK_BARU", "Depok Baru Station", "rail", -6.3917, 106.8214, "Kota Depok"),
    TransitAnchor(
        "HUB_HARJAMUKTI", "Harjamukti LRT Cibubur", "rail", -6.3714, 106.8925, "Kota Depok"
    ),
    TransitAnchor(
        "HUB_BEKASI_BARAT",
        "Bekasi Barat LRT / Revo",
        "multimodal",
        -6.2480,
        106.9920,
        "Kota Bekasi",
    ),
    TransitAnchor("HUB_KRANJI", "Kranji Station", "rail", -6.2242, 106.9792, "Kota Bekasi"),
    TransitAnchor(
        "HUB_TANGERANG", "Tangerang Kota Station", "rail", -6.1767, 106.6347, "Kota Tangerang"
    ),
]

DEMAND_CLUSTERS = [
    # Tangerang & Tangsel Deserts
    DemandAnchor(
        "DEM_CILEDUG",
        "Ciledug Raya - Puri Beta",
        "residential_desert",
        -6.2310,
        106.7260,
        "Kota Tangerang",
    ),
    DemandAnchor(
        "DEM_PAMULANG",
        "Pamulang Square Center",
        "residential_desert",
        -6.3420,
        106.7380,
        "Kota Tangerang Selatan",
    ),
    DemandAnchor(
        "DEM_CIPUTAT",
        "Ciputat Pasar / UIN Hub",
        "residential_desert",
        -6.3150,
        106.7520,
        "Kota Tangerang Selatan",
    ),
    DemandAnchor(
        "DEM_GRAHA_RAYA",
        "Graha Raya Bintaro",
        "residential_desert",
        -6.2550,
        106.6950,
        "Kota Tangerang Selatan",
    ),
    DemandAnchor(
        "DEM_KOTABUMI",
        "Kotabumi Pasar Kemis",
        "residential_desert",
        -6.1550,
        106.5820,
        "Kabupaten Tangerang",
    ),
    # Depok & Bogor Deserts
    DemandAnchor(
        "DEM_SAWANGAN", "Sawangan Permai", "residential_desert", -6.4020, 106.7620, "Kota Depok"
    ),
    DemandAnchor(
        "DEM_CINERE",
        "Cinere Gandul Corridor",
        "residential_desert",
        -6.3350,
        106.7910,
        "Kota Depok",
    ),
    DemandAnchor(
        "DEM_CIKEAS", "Cikeas - Nagrak", "residential_desert", -6.3680, 106.9320, "Kabupaten Bogor"
    ),
    DemandAnchor(
        "DEM_CIBUBUR_JUNCTION",
        "Cibubur Transyogi",
        "mixed_corridor",
        -6.3750,
        106.9050,
        "Jakarta Timur",
    ),
    # Bekasi Deserts
    DemandAnchor(
        "DEM_HARAPAN_INDAH",
        "Harapan Indah Boulevard",
        "residential_desert",
        -6.1820,
        106.9750,
        "Kota Bekasi",
    ),
    DemandAnchor(
        "DEM_TAMBUN",
        "Tambun Selatan Mangunjaya",
        "residential_desert",
        -6.2510,
        107.0580,
        "Kabupaten Bekasi",
    ),
    DemandAnchor(
        "DEM_JATI_ASIH",
        "Jati Asih Komsen Hub",
        "residential_desert",
        -6.2980,
        106.9620,
        "Kota Bekasi",
    ),
    # Jakarta Underserved Feeder Corridors
    DemandAnchor(
        "DEM_JOGLO",
        "Joglo Meruya Intercon",
        "residential_desert",
        -6.2120,
        106.7380,
        "Jakarta Barat",
    ),
    DemandAnchor(
        "DEM_PETUKANGAN",
        "Petukangan Selatan",
        "residential_desert",
        -6.2450,
        106.7550,
        "Jakarta Selatan",
    ),
    DemandAnchor(
        "DEM_PONDOK_BAMBU",
        "Pondok Bambu Duren Sawit",
        "residential_desert",
        -6.2320,
        106.9050,
        "Jakarta Timur",
    ),
]


def get_transit_hubs() -> pd.DataFrame:
    """Returns transit anchors DataFrame."""
    return pd.DataFrame([h._asdict() for h in TRANSIT_HUBS])


def get_demand_anchors() -> pd.DataFrame:
    """Returns demand anchors DataFrame."""
    return pd.DataFrame([d._asdict() for d in DEMAND_CLUSTERS])
