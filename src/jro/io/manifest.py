"""Data manifest tracking and checksum validation."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from jro.exceptions import DataSourceError
from jro.logging_setup import logger


class ManifestEntry(BaseModel):
    """Metadata record for an ingested or processed data asset."""

    key: str
    file_path: str
    sha256: str
    size_bytes: int
    source_url: str | None = None
    license: str | None = None
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hex digest of a local file."""
    if not file_path.is_file():
        raise DataSourceError(f"Target file for hashing does not exist: {file_path}")

    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


class ManifestManager:
    """Manages the manifest.json file in a data directory."""

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path
        self._entries: dict[str, ManifestEntry] = {}
        self.load()

    def load(self) -> None:
        """Loads existing entries from disk if manifest exists."""
        if not self.manifest_path.exists():
            return
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data.get("entries", []):
                    entry = ManifestEntry(**item)
                    self._entries[entry.key] = entry
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load existing manifest at %s: %s", self.manifest_path, e)

    def save(self) -> None:
        """Persists current entries to disk atomically."""
        temp_file = self.manifest_path.with_suffix(".tmp")
        data = {
            "version": "1.0",
            "last_updated": datetime.now(UTC).isoformat(),
            "entries": [e.model_dump() for e in self._entries.values()],
        }
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(self.manifest_path)

    def record(
        self,
        key: str,
        file_path: Path,
        source_url: str | None = None,
        license_str: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ManifestEntry:
        """Records or updates a file in the manifest."""
        sha = compute_sha256(file_path)
        size = file_path.stat().st_size
        entry = ManifestEntry(
            key=key,
            file_path=str(file_path.resolve()),
            sha256=sha,
            size_bytes=size,
            source_url=source_url,
            license=license_str,
            metadata=metadata or {},
        )
        self._entries[key] = entry
        self.save()
        logger.info("Manifest updated for %s (%d bytes, sha256=%s)", key, size, sha[:10])
        return entry

    def get(self, key: str) -> ManifestEntry | None:
        """Retrieves entry by key."""
        return self._entries.get(key)

    def is_valid(self, key: str) -> bool:
        """Checks if file on disk matches its registered sha256 hash."""
        entry = self.get(key)
        if not entry:
            return False
        p = Path(entry.file_path)
        if not p.is_file():
            return False
        return compute_sha256(p) == entry.sha256
