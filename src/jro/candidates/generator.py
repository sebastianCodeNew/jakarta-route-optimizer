"""Candidate bus route corridor generator."""

import json
from pathlib import Path
from typing import Any, NamedTuple

from jro.candidates.stop_placement import RouteStop, place_stops_along_path
from jro.config import VehicleProfile
from jro.logging_setup import logger
from jro.network.road_graph import haversine_distance_km


def load_snapped_road_geometries() -> dict[str, Any]:
    """Loads precomputed real road network geometries for corridors if available."""
    for candidate_path in [
        Path("config/corridor_road_geometries.json"),
        Path(__file__).parent.parent.parent.parent / "config" / "corridor_road_geometries.json",
    ]:
        if candidate_path.exists():
            try:
                with open(candidate_path, "r", encoding="utf-8") as f:
                    res: dict[str, Any] = json.load(f)
                    return res
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Failed loading road geometries from %s: %s", candidate_path, e)
    return {}


class CandidateRoute(NamedTuple):
    """Candidate bus route specification."""

    route_id: str
    route_name: str
    vehicle_profile: str
    length_km: float
    origin_name: str
    destination_name: str
    stops: list[RouteStop]
    waypoints: list[tuple[float, float]]
    stop_cell_ids: list[str]
    overlap_ratio: float


class CorridorTemplate(NamedTuple):
    """Template specification for high-potential transit corridor."""

    id: str
    name: str
    waypoints: list[tuple[float, float]]
    profiles: list[str]
    overlap_ratio: float


# Curated high-potential corridors linking transit hubs with unserved demand centers
RAW_CORRIDOR_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "CORR_PAMULANG_LEBAK_BULUS",
        "name": "Pamulang Square - UIN Ciputat - Lebak Bulus MRT",
        "waypoints": [
            (-6.3420, 106.7380),  # Pamulang
            (-6.3150, 106.7520),  # Ciputat UIN
            (-6.2990, 106.7680),  # Pasar Jumat
            (-6.2891, 106.7745),  # Lebak Bulus Grab MRT
        ],
        "profiles": ["mikrotrans", "non_brt"],
        "overlap_ratio": 0.15,
    },
    {
        "id": "CORR_CILEDUG_BLOK_M",
        "name": "Ciledug Puri Beta - Joglo - Gandaria - Blok M Hub",
        "waypoints": [
            (-6.2310, 106.7260),  # Ciledug
            (-6.2220, 106.7450),  # Petukangan Utara
            (-6.2350, 106.7680),  # Kebayoran Lama
            (-6.2443, 106.7981),  # Blok M Hub
        ],
        "profiles": ["mikrotrans", "non_brt"],
        "overlap_ratio": 0.22,
    },
    {
        "id": "CORR_SAWANGAN_DEPOK_BARU",
        "name": "Sawangan Permai - Muchtar Raya - Depok Baru Station",
        "waypoints": [
            (-6.4020, 106.7620),  # Sawangan
            (-6.3980, 106.7880),  # Mampang Depok
            (-6.3950, 106.8120),  # Margonda Barat
            (-6.3917, 106.8214),  # Stasiun Depok Baru KRL
        ],
        "profiles": ["mikrotrans", "non_brt"],
        "overlap_ratio": 0.10,
    },
    {
        "id": "CORR_CINERE_FATMAWATI",
        "name": "Cinere Gandul - Pondok Labu - Fatmawati MRT",
        "waypoints": [
            (-6.3350, 106.7910),  # Cinere
            (-6.3120, 106.7940),  # Pondok Labu
            (-6.2925, 106.7937),  # Fatmawati MRT
        ],
        "profiles": ["mikrotrans"],
        "overlap_ratio": 0.12,
    },
    {
        "id": "CORR_CIKEAS_HARJAMUKTI",
        "name": "Cikeas Nagrak - Transyogi - Harjamukti LRT Cibubur",
        "waypoints": [
            (-6.3680, 106.9320),  # Cikeas
            (-6.3750, 106.9050),  # Cibubur Junction
            (-6.3714, 106.8925),  # Harjamukti LRT Station
        ],
        "profiles": ["mikrotrans", "non_brt"],
        "overlap_ratio": 0.18,
    },
    {
        "id": "CORR_HARAPAN_INDAH_PULO_GEBANG",
        "name": "Harapan Indah - Cakung - Pulo Gebang Terminal",
        "waypoints": [
            (-6.1820, 106.9750),  # Harapan Indah
            (-6.1950, 106.9620),  # Raya Bekasi / Cakung
            (-6.2128, 106.9536),  # Pulo Gebang Terminal
        ],
        "profiles": ["mikrotrans", "non_brt"],
        "overlap_ratio": 0.14,
    },
    {
        "id": "CORR_TAMBUN_BEKASI_BARAT",
        "name": "Tambun Mangunjaya - Juanda - Bekasi Barat LRT",
        "waypoints": [
            (-6.2510, 107.0580),  # Tambun
            (-6.2492, 107.0189),  # Bekasi Timur
            (-6.2480, 106.9920),  # Bekasi Barat LRT / Revo Hub
        ],
        "profiles": ["mikrotrans", "non_brt"],
        "overlap_ratio": 0.20,
    },
    {
        "id": "CORR_JATI_ASIH_BEKASI_BARAT",
        "name": "Jati Asih Komsen - Pekayon - Bekasi Barat LRT",
        "waypoints": [
            (-6.2980, 106.9620),  # Jati Asih
            (-6.2650, 106.9820),  # Pekayon Raya
            (-6.2480, 106.9920),  # Bekasi Barat LRT
        ],
        "profiles": ["mikrotrans", "non_brt"],
        "overlap_ratio": 0.12,
    },
    {
        "id": "CORR_GRAHA_RAYA_JURANGMANGU",
        "name": "Graha Raya Bintaro - Bintaro Jaya - Jurangmangu KRL",
        "waypoints": [
            (-6.2550, 106.6950),  # Graha Raya
            (-6.2750, 106.7120),  # Bintaro Sektor 7/9
            (-6.2897, 106.7269),  # Jurangmangu KRL
        ],
        "profiles": ["mikrotrans"],
        "overlap_ratio": 0.11,
    },
    {
        "id": "CORR_KOTABUMI_TANGERANG",
        "name": "Kotabumi Pasar Kemis - Sangego - Tangerang Kota Station",
        "waypoints": [
            (-6.1550, 106.5820),  # Kotabumi
            (-6.1680, 106.6120),  # Sangego
            (-6.1767, 106.6347),  # Stasiun Tangerang KRL
        ],
        "profiles": ["mikrotrans", "non_brt"],
        "overlap_ratio": 0.15,
    },
    {
        "id": "CORR_CISAUK_BSD_LEBAK_BULUS",
        "name": "Cisauk Intermodal - BSD City - Lebak Bulus MRT",
        "waypoints": [
            (-6.3236, 106.6436),  # Cisauk
            (-6.3030, 106.6850),  # BSD Boulevard
            (-6.2950, 106.7320),  # Pondok Ranji Outer
            (-6.2891, 106.7745),  # Lebak Bulus Grab MRT
        ],
        "profiles": ["non_brt"],
        "overlap_ratio": 0.25,
    },
    {
        "id": "CORR_CILEDUG_DUKUH_ATAS",
        "name": "Ciledug Puri Beta - Palmerah - Dukuh Atas Multimodal",
        "waypoints": [
            (-6.2310, 106.7260),  # Ciledug
            (-6.2180, 106.7720),  # Kebayoran Baru
            (-6.2069, 106.7978),  # Palmerah
            (-6.2008, 106.8228),  # Dukuh Atas Multimodal
        ],
        "profiles": ["non_brt"],
        "overlap_ratio": 0.28,
    },
]

STRATEGIC_CORRIDOR_TEMPLATES: list[CorridorTemplate] = [
    CorridorTemplate(
        id=str(t["id"]),
        name=str(t["name"]),
        waypoints=list(t["waypoints"]),
        profiles=list(t["profiles"]),
        overlap_ratio=float(t["overlap_ratio"]),
    )
    for t in RAW_CORRIDOR_TEMPLATES
]


def generate_candidate_routes(
    profile: VehicleProfile,
    max_overlap_ratio: float = 0.40,
    h3_resolution: int = 8,
) -> list[CandidateRoute]:
    """Generates and validates candidate routes tailored to vehicle profile.

    Args:
        profile: Target VehicleProfile (mikrotrans or non_brt).
        max_overlap_ratio: Maximum allowable overlap with existing routes.
        h3_resolution: Hexagonal resolution for spatial mapping.

    Returns:
        List of physically viable CandidateRoute instances.
    """
    logger.info("Generating candidate routes for profile: %s", profile.name)
    candidates: list[CandidateRoute] = []
    snapped_geoms = load_snapped_road_geometries()

    for tmpl in STRATEGIC_CORRIDOR_TEMPLATES:
        if profile.name not in tmpl.profiles:
            continue

        overlap = tmpl.overlap_ratio
        if overlap > max_overlap_ratio:
            continue

        if tmpl.id in snapped_geoms:
            geom_data = snapped_geoms[tmpl.id]
            wps = [tuple(pt) for pt in geom_data["waypoints"]]
            length_km = float(geom_data["distance_km"])
        else:
            wps = tmpl.waypoints
            length_km = sum(
                haversine_distance_km(wps[i][0], wps[i][1], wps[i + 1][0], wps[i + 1][1])
                for i in range(len(wps) - 1)
            )

        if length_km < profile.min_length_km or length_km > profile.max_length_km:
            logger.debug(
                "Skipping corridor %s (length %.1f km outside [%.1f, %.1f])",
                tmpl.id,
                length_km,
                profile.min_length_km,
                profile.max_length_km,
            )
            continue

        target_spacing = 0.40 if profile.name == "mikrotrans" else 0.65
        stops = place_stops_along_path(
            waypoints=wps,
            route_code=tmpl.id,
            target_spacing_km=target_spacing,
            h3_resolution=h3_resolution,
        )

        stop_cells = [s.cell_id for s in stops]

        cand = CandidateRoute(
            route_id=tmpl.id,
            route_name=tmpl.name,
            vehicle_profile=profile.name,
            length_km=round(length_km, 2),
            origin_name=stops[0].stop_name,
            destination_name=stops[-1].stop_name,
            stops=stops,
            waypoints=wps,
            stop_cell_ids=stop_cells,
            overlap_ratio=overlap,
        )
        candidates.append(cand)

    logger.info("Generated %d valid candidate routes for %s", len(candidates), profile.name)
    return candidates
