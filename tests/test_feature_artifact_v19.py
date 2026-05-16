from __future__ import annotations

import builtins

import pandas as pd
import pytest

from modules import feature_artifact_v19 as artifact


def test_pickle_roundtrip(tmp_path):
    path = tmp_path / "features.pkl"
    df = pd.DataFrame({"x": [1, 2], "y": [3.5, 4.5]})

    artifact.write_table(df, str(path))
    out = artifact.read_table(str(path))

    pd.testing.assert_frame_equal(out, df)


def test_parquet_roundtrip_when_engine_available(tmp_path):
    try:
        artifact._require_parquet_engine()
    except ImportError:
        pytest.skip("parquet engine is not installed in this environment")

    path = tmp_path / "features.parquet"
    df = pd.DataFrame({"x": [1, 2], "y": [3.5, 4.5]})

    artifact.write_table(df, str(path))
    out = artifact.read_table(str(path))

    pd.testing.assert_frame_equal(out, df)


def test_parquet_without_engine_fails_without_pickle_fallback(tmp_path, monkeypatch):
    path = tmp_path / "features.parquet"
    df = pd.DataFrame({"x": [1]})
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"pyarrow", "fastparquet"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="Parquet support requires"):
        artifact.write_table(df, str(path))

    assert not path.exists()


def test_unsupported_extension_fails(tmp_path):
    with pytest.raises(ValueError, match="Unsupported table extension"):
        artifact.write_table(pd.DataFrame({"x": [1]}), str(tmp_path / "features.bin"))
