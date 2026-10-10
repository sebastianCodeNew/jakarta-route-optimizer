"""Atomic storage utilities for tabular, geospatial, and array data."""

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from jro.exceptions import JroError
from jro.logging_setup import logger


def save_parquet(df: pd.DataFrame, file_path: Path) -> Path:
    """Saves DataFrame as Parquet atomically."""
    file_path = file_path.resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    try:
        df.to_parquet(tmp_path, index=False, engine="pyarrow")
        tmp_path.replace(file_path)
        logger.debug("Saved Parquet to %s (%d rows)", file_path, len(df))
        return file_path
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise JroError(f"Failed to save Parquet to {file_path}: {e}") from e


def load_parquet(file_path: Path) -> pd.DataFrame:
    """Loads DataFrame from Parquet file."""
    if not file_path.exists():
        raise JroError(f"Parquet file does not exist: {file_path}")
    return pd.read_parquet(file_path, engine="pyarrow")


def save_geodataframe(gdf: gpd.GeoDataFrame, file_path: Path, driver: str = "Parquet") -> Path:
    """Saves GeoDataFrame to GeoParquet or GeoJSON atomically."""
    file_path = file_path.resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    try:
        if driver.lower() in ("parquet", "geoparquet") or file_path.suffix == ".parquet":
            gdf.to_parquet(tmp_path)
        else:
            gdf.to_file(tmp_path, driver="GeoJSON")
        tmp_path.replace(file_path)
        logger.debug("Saved GeoDataFrame to %s (%d geometries)", file_path, len(gdf))
        return file_path
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise JroError(f"Failed to save GeoDataFrame to {file_path}: {e}") from e


def load_geodataframe(file_path: Path) -> gpd.GeoDataFrame:
    """Loads GeoDataFrame from GeoParquet or GeoJSON."""
    if not file_path.exists():
        raise JroError(f"Geospatial file does not exist: {file_path}")
    if file_path.suffix == ".parquet":
        return gpd.read_parquet(file_path)
    return gpd.read_file(file_path)


def save_numpy(array: np.ndarray, file_path: Path) -> Path:
    """Saves NumPy array atomically."""
    file_path = file_path.resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    try:
        np.save(tmp_path, array)
        # np.save appends .npy if not present
        actual_tmp = tmp_path if tmp_path.exists() else tmp_path.with_suffix(".tmp.npy")
        actual_target = file_path if file_path.suffix == ".npy" else file_path.with_suffix(".npy")
        actual_tmp.replace(actual_target)
        logger.debug("Saved NumPy array shape %s to %s", array.shape, actual_target)
        return actual_target
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise JroError(f"Failed to save NumPy array to {file_path}: {e}") from e


def load_numpy(file_path: Path) -> np.ndarray:
    """Loads NumPy array from disk."""
    p = file_path if file_path.suffix == ".npy" else file_path.with_suffix(".npy")
    if not p.exists():
        raise JroError(f"NumPy file does not exist: {p}")
    return np.load(p)


def save_json(data: Any, file_path: Path) -> Path:
    """Saves arbitrary JSON-serializable structure atomically."""
    file_path = file_path.resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_path.replace(file_path)
        return file_path
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise JroError(f"Failed to save JSON to {file_path}: {e}") from e
