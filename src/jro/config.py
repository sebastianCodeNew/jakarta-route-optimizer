"""Validated configuration models using Pydantic v2."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from jro.exceptions import ConfigurationError


class BBox(BaseModel, frozen=True):
    """Geographic bounding box with coordinate validation."""

    min_lon: float = Field(ge=105.0, le=108.5)
    min_lat: float = Field(ge=-7.5, le=-5.0)
    max_lon: float = Field(ge=105.0, le=108.5)
    max_lat: float = Field(ge=-7.5, le=-5.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "BBox":
        if self.min_lon >= self.max_lon:
            raise ValueError("min_lon must be strictly less than max_lon")
        if self.min_lat >= self.max_lat:
            raise ValueError("min_lat must be strictly less than max_lat")
        return self


class VehicleProfile(BaseModel, frozen=True):
    """Operational and physical specifications for transit vehicle types."""

    name: Literal["mikrotrans", "non_brt"]
    display_name: str
    capacity: int = Field(gt=0)
    dwell_seconds: float = Field(gt=0)
    layover_minutes: float = Field(ge=0)
    allowed_highway: tuple[str, ...]
    min_length_km: float = Field(gt=0)
    max_length_km: float = Field(gt=0)
    operating_cost_per_km_idr: float = Field(gt=0)
    emission_factor_key: str

    @model_validator(mode="after")
    def validate_lengths(self) -> "VehicleProfile":
        if self.min_length_km >= self.max_length_km:
            raise ValueError("min_length_km must be less than max_length_km")
        return self


class RailService(BaseModel, frozen=True):
    """Rail corridor parameters and service headway."""

    name: str
    agency: str
    route_type: int
    headway_minutes: dict[str, float]
    commercial_speed_kmh: float = Field(gt=0)


class GravityParams(BaseModel, frozen=True):
    """Parameters for trip distribution."""

    beta: float = Field(gt=0)
    max_travel_time_minutes: float = Field(gt=0)
    min_flow_threshold: float = Field(ge=0)


class ModeChoiceParams(BaseModel, frozen=True):
    """Utility coefficients for Multinomial Logit mode choice."""

    asc_motorcycle: float
    asc_car: float
    asc_transit: float = 0.0
    beta_ivt: float = Field(lt=0)
    beta_walk: float = Field(lt=0)
    beta_wait: float = Field(lt=0)
    beta_xfer: float = Field(lt=0)
    beta_cost: float = Field(lt=0)
    walk_speed_kmh: float = Field(gt=0)
    max_walk_distance_km: float = Field(gt=0)
    max_wait_minutes: float = Field(gt=0)


class CostParams(BaseModel, frozen=True):
    """Fares, vehicle operating costs, and parking fees."""

    fare_mikrotrans_idr: float = Field(ge=0)
    fare_non_brt_idr: float = Field(ge=0)
    fare_krl_base_idr: float = Field(ge=0)
    fare_krl_rate_per_km_idr: float = Field(ge=0)
    motorcycle_cost_per_km_idr: float = Field(gt=0)
    car_cost_per_km_idr: float = Field(gt=0)
    motorcycle_parking_idr: float = Field(ge=0)
    car_parking_idr: float = Field(ge=0)


class OccupancyParams(BaseModel, frozen=True):
    """Vehicle occupancy factors."""

    motorcycle: float = Field(ge=1.0)
    car: float = Field(ge=1.0)


class ModelParams(BaseModel, frozen=True):
    """Consolidated demand model parameters."""

    gravity: GravityParams
    mode_choice: ModeChoiceParams
    costs: CostParams
    occupancy: OccupancyParams


class EmissionMode(BaseModel, frozen=True):
    """Emission factors per kilometer (g/km) for a vehicle class."""

    co2: float = Field(ge=0)
    pm25: float = Field(ge=0)
    nox: float = Field(ge=0)
    co: float = Field(ge=0)


class EmissionFactors(BaseModel, frozen=True):
    """Emission factor library."""

    factors: dict[str, EmissionMode]


class ObjectiveWeights(BaseModel, frozen=True):
    """Multi-criteria objective weights."""

    vkt_reduction: float = Field(ge=0)
    emission_reduction: float = Field(ge=0)
    ridership: float = Field(ge=0)
    accessibility_gain: float = Field(ge=0)
    equity_transit_desert: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_sum(self) -> "ObjectiveWeights":
        total = (
            self.vkt_reduction
            + self.emission_reduction
            + self.ridership
            + self.accessibility_gain
            + self.equity_transit_desert
        )
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Objective weights must sum to 1.0, got {total:.4f}")
        return self


class ObjectiveConstraints(BaseModel, frozen=True):
    """Hard physical and service constraints for candidate bus routes."""

    max_overlap_ratio: float = Field(gt=0, le=1.0)
    min_stop_spacing_meters: float = Field(gt=0)
    max_stop_spacing_meters: float = Field(gt=0)
    min_stops: int = Field(gt=2)
    max_stops: int = Field(gt=2)
    require_transit_anchor: bool = True


class Settings(BaseModel, frozen=True):
    """Global system runtime and spatial settings."""

    study_area: BBox
    h3_resolution: int = Field(ge=6, le=10, default=8)
    peak_window: tuple[str, str] = ("06:00", "09:00")
    off_peak_window: tuple[str, str] = ("10:00", "15:00")
    evening_peak_window: tuple[str, str] = ("17:00", "20:00")
    data_dir: Path = Path("data")
    seed: int = 42
    max_cpu_workers: int = Field(ge=1, default=4)


class AppConfig(BaseModel, frozen=True):
    """Unified application configuration."""

    settings: Settings
    vehicles: dict[str, VehicleProfile]
    rail: dict[str, RailService]
    model: ModelParams
    emissions: EmissionFactors
    weights: ObjectiveWeights
    constraints: ObjectiveConstraints


def _read_yaml(file_path: Path) -> dict[str, Any]:
    if not file_path.exists():
        raise ConfigurationError(f"Configuration file not found: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
            return content if isinstance(content, dict) else {}
    except Exception as e:
        raise ConfigurationError(f"Error parsing YAML from {file_path}: {e}") from e


def load_config(config_dir: str | Path = "config") -> AppConfig:
    """Loads and validates all system YAML configurations.

    Args:
        config_dir: Directory containing YAML configuration files.

    Returns:
        Fully validated AppConfig instance.
    """
    base_dir = Path(config_dir)

    raw_settings = _read_yaml(base_dir / "settings.yaml")
    settings = Settings(
        study_area=BBox(**raw_settings["study_area"]),
        h3_resolution=raw_settings.get("h3_resolution", 8),
        peak_window=tuple(raw_settings.get("peak_window", ["06:00", "09:00"])),
        off_peak_window=tuple(raw_settings.get("off_peak_window", ["10:00", "15:00"])),
        evening_peak_window=tuple(raw_settings.get("evening_peak_window", ["17:00", "20:00"])),
        data_dir=Path(raw_settings.get("data_dir", "data")),
        seed=raw_settings.get("seed", 42),
        max_cpu_workers=raw_settings.get("max_cpu_workers", 4),
    )

    raw_vehicles = _read_yaml(base_dir / "vehicle_profiles.yaml").get("vehicle_profiles", {})
    vehicles = {
        k: VehicleProfile(
            name=v["name"],
            display_name=v["display_name"],
            capacity=v["capacity"],
            dwell_seconds=v["dwell_seconds"],
            layover_minutes=v["layover_minutes"],
            allowed_highway=tuple(v["allowed_highway"]),
            min_length_km=v["min_length_km"],
            max_length_km=v["max_length_km"],
            operating_cost_per_km_idr=v["operating_cost_per_km_idr"],
            emission_factor_key=v["emission_factor_key"],
        )
        for k, v in raw_vehicles.items()
    }

    raw_rail = _read_yaml(base_dir / "rail_services.yaml").get("rail_services", {})
    rail = {
        k: RailService(
            name=v["name"],
            agency=v["agency"],
            route_type=v["route_type"],
            headway_minutes=v["headway_minutes"],
            commercial_speed_kmh=v["commercial_speed_kmh"],
        )
        for k, v in raw_rail.items()
    }

    raw_model = _read_yaml(base_dir / "model_params.yaml")
    model = ModelParams(
        gravity=GravityParams(**raw_model["gravity"]),
        mode_choice=ModeChoiceParams(**raw_model["mode_choice"]),
        costs=CostParams(**raw_model["costs"]),
        occupancy=OccupancyParams(**raw_model["occupancy"]),
    )

    raw_emissions = _read_yaml(base_dir / "emission_factors.yaml").get("emission_factors", {})
    emissions = EmissionFactors(factors={k: EmissionMode(**v) for k, v in raw_emissions.items()})

    raw_obj = _read_yaml(base_dir / "objective.yaml")
    weights = ObjectiveWeights(**raw_obj["weights"])
    constraints = ObjectiveConstraints(**raw_obj["constraints"])

    return AppConfig(
        settings=settings,
        vehicles=vehicles,
        rail=rail,
        model=model,
        emissions=emissions,
        weights=weights,
        constraints=constraints,
    )
