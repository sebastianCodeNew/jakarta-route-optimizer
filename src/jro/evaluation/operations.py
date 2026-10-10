"""Bus fleet operations, cycle time, headway, and capacity modeling."""

from typing import NamedTuple

from jro.candidates.generator import CandidateRoute
from jro.config import VehicleProfile


class RouteOperations(NamedTuple):
    """Operational metrics for a candidate bus route under a given fleet size."""

    fleet_size: int
    cycle_time_minutes: float
    headway_minutes: float
    wait_time_minutes: float
    hourly_capacity_passengers: float
    daily_capacity_passengers: float
    commercial_speed_kmh: float
    daily_bus_vkt: float
    daily_operating_cost_idr: float


def compute_route_operations(
    route: CandidateRoute,
    profile: VehicleProfile,
    fleet_size: int = 10,
    congested_road_speed_kmh: float = 20.0,
) -> RouteOperations:
    """Calculates cycle time, headway, capacity, and operational cost.

    Formulas:
        C = (2 * L / v_bus) * 60 + 2 * t_layover + 2 * n_stops * t_dwell
        h = C / N
        HourlyCap = Capacity * (60 / h)

    Args:
        route: CandidateRoute instance.
        profile: VehicleProfile (mikrotrans or non_brt).
        fleet_size: Number of operating buses allocated to route (N >= 1).
        congested_road_speed_kmh: Effective traffic speed along corridor.

    Returns:
        RouteOperations instance.
    """
    n_stops = len(route.stops)
    # Effective bus operating speed accounting for acceleration dynamics
    bus_speed = max(10.0, min(35.0, congested_road_speed_kmh * 0.85))

    running_time_min = (2.0 * route.length_km / bus_speed) * 60.0
    dwell_time_min = 2.0 * n_stops * (profile.dwell_seconds / 60.0)
    layover_time_min = 2.0 * profile.layover_minutes

    cycle_time = running_time_min + dwell_time_min + layover_time_min
    headway = cycle_time / max(1, fleet_size)

    # Effective wait time is headway / 2, capped by passenger scheduling limit (e.g. 20 min)
    wait_time = min(20.0, headway / 2.0)

    # Hourly capacity in peak direction
    hourly_capacity = profile.capacity * (60.0 / max(0.5, headway))
    # Daily capacity across 16-hour operating window with directional peak concentration
    daily_capacity = hourly_capacity * 10.5

    # Daily bus vehicle kilometers travelled (VKT)
    # Total round-trips per day per bus across 16-hour window with off-peak tapering (~12 equivalent peak hours)
    trips_per_bus_day = (12.0 * 60.0) / cycle_time
    total_trips_day = trips_per_bus_day * fleet_size
    daily_vkt = total_trips_day * (2.0 * route.length_km)

    daily_cost = daily_vkt * profile.operating_cost_per_km_idr

    return RouteOperations(
        fleet_size=fleet_size,
        cycle_time_minutes=round(cycle_time, 2),
        headway_minutes=round(headway, 2),
        wait_time_minutes=round(wait_time, 2),
        hourly_capacity_passengers=round(hourly_capacity, 1),
        daily_capacity_passengers=round(daily_capacity, 1),
        commercial_speed_kmh=round(bus_speed, 1),
        daily_bus_vkt=round(daily_vkt, 1),
        daily_operating_cost_idr=round(daily_cost, 0),
    )
