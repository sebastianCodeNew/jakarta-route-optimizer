"""Directory and path resolution for JRO data pipeline."""

from pathlib import Path

from pydantic import BaseModel


class DataPaths(BaseModel, frozen=True):
    """Immutable path container for data pipeline stages."""

    base: Path
    raw: Path
    interim: Path
    processed: Path
    outputs: Path

    @classmethod
    def from_root(cls, base_dir: Path | str = "data") -> "DataPaths":
        """Instantiates DataPaths and creates directories if they don't exist."""
        base = Path(base_dir).resolve()
        raw = base / "raw"
        interim = base / "interim"
        processed = base / "processed"
        outputs = base / "outputs"

        for p in (raw, interim, processed, outputs):
            p.mkdir(parents=True, exist_ok=True)

        return cls(base=base, raw=raw, interim=interim, processed=processed, outputs=outputs)
