"""Synthetic GTFS feed builder for KRL Commuterline, MRT, and LRT networks."""

import csv
import io
import zipfile
from pathlib import Path

from jro.config import RailService
from jro.logging_setup import logger

# Essential station sequence corridors with precise WGS-84 coordinates
RAIL_CORRIDORS: dict[str, list[tuple[str, str, float, float]]] = {
    "mrt_jakarta": [
        ("MRT_LB", "Lebak Bulus Grab", -6.2891, 106.7745),
        ("MRT_FT", "Fatmawati", -6.2925, 106.7937),
        ("MRT_CP", "Cipete Raya", -6.2785, 106.7972),
        ("MRT_HN", "Haji Nawi", -6.2667, 106.7974),
        ("MRT_BA", "Blok A", -6.2553, 106.7972),
        ("MRT_BM", "Blok M BCA", -6.2443, 106.7981),
        ("MRT_AS", "ASEAN", -6.2384, 106.7990),
        ("MRT_SN", "Senayan Mastercard", -6.2252, 106.8027),
        ("MRT_IS", "Istora Mandiri", -6.2195, 106.8062),
        ("MRT_BH", "Bendungan Hilir", -6.2132, 106.8174),
        ("MRT_ST", "Setiabudi Astra", -6.2088, 106.8209),
        ("MRT_DA", "Dukuh Atas BNI", -6.2008, 106.8228),
        ("MRT_HI", "Bundaran HI", -6.1928, 106.8230),
    ],
    "krl_bogor": [
        ("KRL_BOO", "Bogor", -6.5971, 106.7891),
        ("KRL_CLT", "Cilebut", -6.5283, 106.7997),
        ("KRL_BJD", "Bojong Gede", -6.4932, 106.7988),
        ("KRL_CTA", "Citayam", -6.4489, 106.8010),
        ("KRL_DP", "Depok", -6.4058, 106.8188),
        ("KRL_DPB", "Depok Baru", -6.3917, 106.8214),
        ("KRL_POC", "Pondok Cina", -6.3686, 106.8317),
        ("KRL_UI", "Universitas Indonesia", -6.3606, 106.8317),
        ("KRL_UP", "Universitas Pancasila", -6.3392, 106.8344),
        ("KRL_LNA", "Lenteng Agung", -6.3303, 106.8349),
        ("KRL_TNT", "Tanjung Barat", -6.3078, 106.8389),
        ("KRL_PSM", "Pasar Minggu", -6.2842, 106.8443),
        ("KRL_PSMB", "Pasar Minggu Baru", -6.2631, 106.8536),
        ("KRL_DRN", "Duren Kalibata", -6.2553, 106.8550),
        ("KRL_CWG", "Cawang", -6.2427, 106.8583),
        ("KRL_TEB", "Tebet", -6.2266, 106.8584),
        ("KRL_MRI", "Manggarai", -6.2099, 106.8501),
        ("KRL_CKI", "Cikini", -6.1983, 106.8433),
        ("KRL_GDD", "Gondangdia", -6.1864, 106.8329),
        ("KRL_JUA", "Juanda", -6.1667, 106.8300),
        ("KRL_SW", "Sawah Besar", -6.1603, 106.8286),
        ("KRL_MB", "Mangga Besar", -6.1497, 106.8272),
        ("KRL_JAY", "Jayakarta", -6.1417, 106.8228),
        ("KRL_JAK", "Jakarta Kota", -6.1376, 106.8144),
    ],
    "krl_cikarang": [
        ("KRL_CKR", "Cikarang", -6.2550, 107.1511),
        ("KRL_CBT", "Cibitung", -6.2625, 107.0989),
        ("KRL_TBN", "Tambun", -6.2611, 107.0600),
        ("KRL_BKT", "Bekasi Timur", -6.2492, 107.0189),
        ("KRL_BKS", "Bekasi", -6.2361, 106.9997),
        ("KRL_KRJ", "Kranji", -6.2242, 106.9792),
        ("KRL_CKN", "Cakung", -6.2189, 106.9531),
        ("KRL_KLDB", "Klender Baru", -6.2167, 106.9389),
        ("KRL_BRN", "Buaran", -6.2153, 106.9231),
        ("KRL_KLD", "Klender", -6.2133, 106.8997),
        ("KRL_JNG", "Jatinegara", -6.2150, 106.8681),
        ("KRL_MRI2", "Manggarai", -6.2099, 106.8501),
        ("KRL_SUD", "Sudirman", -6.2025, 106.8236),
        ("KRL_KRT", "Karet", -6.2008, 106.8164),
        ("KRL_THB", "Tanah Abang", -6.1856, 106.8111),
        ("KRL_DUR", "Duri", -6.1558, 106.8019),
        ("KRL_KPB", "Kampung Bandan", -6.1328, 106.8267),
        ("KRL_PSE", "Pasar Senen", -6.1744, 106.8447),
    ],
    "krl_tangerang": [
        ("KRL_TNG", "Tangerang", -6.1767, 106.6347),
        ("KRL_TTI", "Tanah Tinggi", -6.1728, 106.6508),
        ("KRL_BPR", "Batu Ceper", -6.1706, 106.6631),
        ("KRL_PRS", "Poris", -6.1667, 106.6742),
        ("KRL_KDR", "Kalideres", -6.1603, 106.6975),
        ("KRL_RWB", "Rawa Buaya", -6.1589, 106.7231),
        ("KRL_BJI", "Bojong Indah", -6.1561, 106.7364),
        ("KRL_TMK", "Taman Kota", -6.1550, 106.7561),
        ("KRL_PSG", "Pesing", -6.1583, 106.7725),
        ("KRL_GGL", "Grogol", -6.1600, 106.7889),
        ("KRL_DUR2", "Duri", -6.1558, 106.8019),
    ],
    "krl_rangkasbitung": [
        ("KRL_RK", "Rangkasbitung", -6.3606, 106.2506),
        ("KRL_MJA", "Maja", -6.3317, 106.3986),
        ("KRL_TGS", "Tigaraksa", -6.3150, 106.4678),
        ("KRL_PRP", "Parung Panjang", -6.3444, 106.5686),
        ("KRL_CSK", "Cisauk", -6.3236, 106.6436),
        ("KRL_SRP", "Serpong", -6.3183, 106.6669),
        ("KRL_RU", "Rawa Buntu", -6.3094, 106.6872),
        ("KRL_SDM", "Sudimara", -6.2975, 106.7114),
        ("KRL_JMU", "Jurangmangu", -6.2897, 106.7269),
        ("KRL_PDJ", "Pondok Ranji", -6.2750, 106.7444),
        ("KRL_KBY", "Kebayoran", -6.2383, 106.7828),
        ("KRL_PLM", "Palmerah", -6.2069, 106.7978),
        ("KRL_THB2", "Tanah Abang", -6.1856, 106.8111),
    ],
    "lrt_jabodebek_cibubur": [
        ("LRT_DA", "Dukuh Atas", -6.2028, 106.8242),
        ("LRT_STB", "Setiabudi", -6.2114, 106.8286),
        ("LRT_RS", "Rasuna Said", -6.2208, 106.8322),
        ("LRT_KNG", "Kuningan", -6.2347, 106.8336),
        ("LRT_PCR", "Pancoran", -6.2425, 106.8456),
        ("LRT_CKK", "Cikoko", -6.2431, 106.8589),
        ("LRT_CWG", "Cawang", -6.2492, 106.8719),
        ("LRT_TMII", "TMII", -6.2936, 106.8839),
        ("LRT_KMR", "Kampung Rambutan", -6.3131, 106.8761),
        ("LRT_CRC", "Ciracas", -6.3353, 106.8789),
        ("LRT_HJM", "Harjamukti Cibubur", -6.3714, 106.8925),
    ],
}


def build_synthetic_rail_gtfs(
    rail_configs: dict[str, RailService],
    output_path: Path,
) -> Path:
    """Compiles synthetic GTFS zip archive for rail lines."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Building synthetic Rail GTFS feed at %s", output_path)

    agency_rows = [
        ["agency_id", "agency_name", "agency_url", "agency_timezone", "agency_lang"],
        [
            "KAI_COMMUTER",
            "PT Kereta Commuter Indonesia",
            "https://commuterline.id",
            "Asia/Jakarta",
            "id",
        ],
        ["MRT_JAKARTA", "PT MRT Jakarta", "https://jakartamrt.co.id", "Asia/Jakarta", "id"],
        [
            "LRT_JABODEBEK",
            "KAI Divisi LRT Jabodebek",
            "https://lrtjabodebek.kai.id",
            "Asia/Jakarta",
            "id",
        ],
    ]

    calendar_rows = [
        [
            "service_id",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "start_date",
            "end_date",
        ],
        ["ALL_DAYS", "1", "1", "1", "1", "1", "1", "1", "20240101", "20301231"],
    ]

    stops_seen: set[str] = set()
    stops_rows = [["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type"]]
    routes_rows = [["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"]]
    trips_rows = [["route_id", "service_id", "trip_id", "trip_headsign", "direction_id"]]
    stop_times_rows = [["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]]
    frequencies_rows = [["trip_id", "start_time", "end_time", "headway_secs"]]

    for corr_id, station_list in RAIL_CORRIDORS.items():
        conf = rail_configs.get(corr_id)
        agency_id = (
            "MRT_JAKARTA"
            if "mrt" in corr_id
            else ("LRT_JABODEBEK" if "lrt" in corr_id else "KAI_COMMUTER")
        )
        route_name = conf.name if conf else corr_id.upper()
        route_type = str(conf.route_type if conf else (1 if "mrt" in corr_id else 2))

        routes_rows.append([corr_id, agency_id, route_name, route_name, route_type])

        # Forward trip
        trip_fwd = f"TRIP_{corr_id}_FWD"
        trips_rows.append([corr_id, "ALL_DAYS", trip_fwd, station_list[-1][1], "0"])

        # Backward trip
        trip_rev = f"TRIP_{corr_id}_REV"
        trips_rows.append([corr_id, "ALL_DAYS", trip_rev, station_list[0][1], "1"])

        # Headway frequencies (peak vs off-peak)
        peak_headway = (conf.headway_minutes.get("peak", 6.0) if conf else 6.0) * 60
        offpeak_headway = (conf.headway_minutes.get("off_peak", 12.0) if conf else 12.0) * 60

        frequencies_rows.append([trip_fwd, "06:00:00", "09:00:00", str(int(peak_headway))])
        frequencies_rows.append([trip_fwd, "09:00:00", "16:00:00", str(int(offpeak_headway))])
        frequencies_rows.append([trip_fwd, "16:00:00", "20:00:00", str(int(peak_headway))])

        frequencies_rows.append([trip_rev, "06:00:00", "09:00:00", str(int(peak_headway))])
        frequencies_rows.append([trip_rev, "09:00:00", "16:00:00", str(int(offpeak_headway))])
        frequencies_rows.append([trip_rev, "16:00:00", "20:00:00", str(int(peak_headway))])

        # Add stops and stop times
        curr_min = 360  # 06:00 AM base
        for seq, (sid, sname, slat, slon) in enumerate(station_list, start=1):
            if sid not in stops_seen:
                stops_rows.append([sid, sname, f"{slat:.6f}", f"{slon:.6f}", "0"])
                stops_seen.add(sid)

            arr_str = f"{curr_min // 60:02d}:{curr_min % 60:02d}:00"
            dep_str = f"{(curr_min + 1) // 60:02d}:{(curr_min + 1) % 60:02d}:00"
            stop_times_rows.append([trip_fwd, arr_str, dep_str, sid, str(seq)])
            curr_min += 3

        # Reverse stop times
        curr_min = 360
        for seq, (sid, sname, slat, slon) in enumerate(reversed(station_list), start=1):
            arr_str = f"{curr_min // 60:02d}:{curr_min % 60:02d}:00"
            dep_str = f"{(curr_min + 1) // 60:02d}:{(curr_min + 1) % 60:02d}:00"
            stop_times_rows.append([trip_rev, arr_str, dep_str, sid, str(seq)])
            curr_min += 3

    # Pack into ZIP archive
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, rows in [
            ("agency.txt", agency_rows),
            ("calendar.txt", calendar_rows),
            ("stops.txt", stops_rows),
            ("routes.txt", routes_rows),
            ("trips.txt", trips_rows),
            ("stop_times.txt", stop_times_rows),
            ("frequencies.txt", frequencies_rows),
        ]:
            out = io.StringIO()
            writer = csv.writer(out)
            writer.writerows(rows)
            zf.writestr(fname, out.getvalue())

    logger.info(
        "Built synthetic Rail GTFS feed (%d stops, %d corridors)",
        len(stops_seen),
        len(RAIL_CORRIDORS),
    )
    return output_path
