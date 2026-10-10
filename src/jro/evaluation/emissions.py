"""VKT reduction and multi-pollutant emission impacts evaluation."""

from typing import NamedTuple

from jro.config import AppConfig
from jro.evaluation.fast_eval import FastEvaluationResult
from jro.evaluation.operations import RouteOperations


class EmissionImpactResult(NamedTuple):
    """Environmental impact metrics (daily VKT reduction and net emission savings)."""

    shifted_motorcycle_vkt: float
    shifted_car_vkt: float
    total_private_vkt_reduced: float
    added_bus_vkt: float
    net_vkt_reduced: float
    net_co2_kg_saved: float
    net_pm25_kg_saved: float
    net_nox_kg_saved: float
    net_co_kg_saved: float


def compute_emission_impacts(
    eval_result: FastEvaluationResult,
    operations: RouteOperations,
    route_length_km: float,
    config: AppConfig,
    vehicle_profile_name: str,
) -> EmissionImpactResult:
    """Computes daily VKT reduction and net pollutant emission savings.

    Args:
        eval_result: FastEvaluationResult containing shifted trips.
        operations: RouteOperations containing bus VKT.
        route_length_km: One-way route distance in km.
        config: System AppConfig with emission factors and occupancies.
        vehicle_profile_name: "mikrotrans" or "non_brt".

    Returns:
        EmissionImpactResult instance.
    """
    occ = config.model.occupancy
    ef = config.emissions.factors

    # Average private vehicle commute trip length displaced across Jabodetabek (BPS commuter survey: 15.5-18.5 km)
    # Feeder routes connect commuters to trunk rail/BRT lines, eliminating their entire door-to-door private drive.
    avg_trip_dist_km = max(route_length_km * 0.85, 14.5)

    # Private vehicle kilometers displaced
    # Occupancy: 1.15 for motorcycle, 1.45 for car
    mc_vkt_avoided = (eval_result.shifted_motorcycle_trips * avg_trip_dist_km) / occ.motorcycle
    car_vkt_avoided = (eval_result.shifted_car_trips * avg_trip_dist_km) / occ.car
    total_private_vkt = mc_vkt_avoided + car_vkt_avoided

    added_bus_vkt = operations.daily_bus_vkt
    net_vkt = total_private_vkt - added_bus_vkt

    # Emission factors in grams / km
    ef_mc = ef["motorcycle"]
    ef_car = ef["car"]
    bus_ef_key = "mikrotrans" if vehicle_profile_name == "mikrotrans" else "bus_medium_large"
    ef_bus = ef[bus_ef_key]

    # Gross emissions avoided from private trips (grams -> kilograms: / 1000)
    co2_avoided_kg = (mc_vkt_avoided * ef_mc.co2 + car_vkt_avoided * ef_car.co2) / 1000.0
    pm25_avoided_kg = (mc_vkt_avoided * ef_mc.pm25 + car_vkt_avoided * ef_car.pm25) / 1000.0
    nox_avoided_kg = (mc_vkt_avoided * ef_mc.nox + car_vkt_avoided * ef_car.nox) / 1000.0
    co_avoided_kg = (mc_vkt_avoided * ef_mc.co + car_vkt_avoided * ef_car.co) / 1000.0

    # Gross emissions emitted by added buses (kilograms)
    co2_bus_kg = (added_bus_vkt * ef_bus.co2) / 1000.0
    pm25_bus_kg = (added_bus_vkt * ef_bus.pm25) / 1000.0
    nox_bus_kg = (added_bus_vkt * ef_bus.nox) / 1000.0
    co_bus_kg = (added_bus_vkt * ef_bus.co) / 1000.0

    # Net savings: positive means pollution reduction!
    net_co2 = co2_avoided_kg - co2_bus_kg
    net_pm25 = pm25_avoided_kg - pm25_bus_kg
    net_nox = nox_avoided_kg - nox_bus_kg
    net_co = co_avoided_kg - co_bus_kg

    return EmissionImpactResult(
        shifted_motorcycle_vkt=round(mc_vkt_avoided, 1),
        shifted_car_vkt=round(car_vkt_avoided, 1),
        total_private_vkt_reduced=round(total_private_vkt, 1),
        added_bus_vkt=round(added_bus_vkt, 1),
        net_vkt_reduced=round(net_vkt, 1),
        net_co2_kg_saved=round(net_co2, 2),
        net_pm25_kg_saved=round(net_pm25, 4),
        net_nox_kg_saved=round(net_nox, 3),
        net_co_kg_saved=round(net_co, 2),
    )
