"""Custom domain exceptions for Jakarta Route Optimizer (JRO)."""


class JroError(Exception):
    """Base exception for all JRO domain errors."""

    def __init__(self, message: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(JroError):
    """Raised when configuration validation or file parsing fails."""


class DataSourceError(JroError):
    """Raised when remote data ingestion or checksum verification fails."""


class NetworkBuildError(JroError):
    """Raised when transportation network assembly or validation fails."""


class DemandModelError(JroError):
    """Raised when trip generation, gravity distribution, or IPF fails to converge."""


class CongestionBudgetExceededError(JroError):
    """Raised when API daily rate limit or budget threshold is reached."""


class OptimizationError(JroError):
    """Raised when candidate generation or search optimization encounters an error."""


class ValidationError(JroError):
    """Raised when backtesting or GTFS validation criteria fail."""
