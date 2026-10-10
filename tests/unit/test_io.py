"""Unit tests for I/O operations, checksums, and manifests."""

import numpy as np
import pandas as pd

from jro.io.manifest import ManifestManager, compute_sha256
from jro.io.paths import DataPaths
from jro.io.storage import load_numpy, load_parquet, save_numpy, save_parquet


def test_paths_initialization(tmp_path) -> None:
    """Verifies DataPaths creates required directories."""
    paths = DataPaths.from_root(tmp_path / "data_test")
    assert paths.raw.is_dir()
    assert paths.interim.is_dir()
    assert paths.processed.is_dir()
    assert paths.outputs.is_dir()


def test_sha256_and_manifest(tmp_path) -> None:
    """Tests file hashing and manifest validation."""
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Jakarta Route Optimizer Test Data", encoding="utf-8")

    sha = compute_sha256(test_file)
    assert len(sha) == 64

    manifest_file = tmp_path / "manifest.json"
    mgr = ManifestManager(manifest_file)
    mgr.record("test_key", test_file, source_url="https://example.com", license_str="MIT")

    assert mgr.is_valid("test_key")

    # Tamper with file
    test_file.write_text("Modified data", encoding="utf-8")
    assert not mgr.is_valid("test_key")


def test_atomic_storage_helpers(tmp_path) -> None:
    """Tests Parquet and NumPy storage helpers."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    pq_path = tmp_path / "test.parquet"
    save_parquet(df, pq_path)
    loaded_df = load_parquet(pq_path)
    assert len(loaded_df) == 3
    assert list(loaded_df["a"]) == [1, 2, 3]

    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    npy_path = tmp_path / "matrix.npy"
    save_numpy(arr, npy_path)
    loaded_arr = load_numpy(npy_path)
    assert np.allclose(arr, loaded_arr)
