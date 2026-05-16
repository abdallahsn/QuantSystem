from __future__ import annotations

import json

import pytest

from modules.feature_factory_v19 import (
    EXPECTED_SCHEMA_VERSION,
    V19FeatureFactory,
    resolve_meta_feature_names,
)


def _write_schema(tmp_path, stat_features):
    schema = {
        "version": EXPECTED_SCHEMA_VERSION,
        "stat_features": list(stat_features),
        "meta_features": resolve_meta_feature_names(include_xgboost=True),
        "visual_features": [],
        "passthrough_cols": [],
        "timestamp_cols": ["ts_event", "label_end_ts"],
        "input_dim": len(stat_features) + len(resolve_meta_feature_names(include_xgboost=True)),
        "seq_len": 50,
    }
    (tmp_path / "feature_schema_v19.json").write_text(json.dumps(schema), encoding="utf-8")
    (tmp_path / "scaler_params.json").write_text("{}", encoding="utf-8")


@pytest.mark.parametrize("feature", ["bid_wall_delta_fwd_k", "ask_wall_delta_fwd_k"])
def test_forward_wall_delta_targets_are_forbidden(tmp_path, feature):
    _write_schema(tmp_path, ["cvd", feature])

    with pytest.raises(ValueError, match="forbidden leakage-prone"):
        V19FeatureFactory(str(tmp_path))


def test_raw_forward_wall_delta_targets_are_forbidden(tmp_path):
    _write_schema(tmp_path, ["cvd", "raw__bid_wall_delta_fwd_k"])

    with pytest.raises(ValueError, match="forbidden leakage-prone"):
        V19FeatureFactory(str(tmp_path))


def test_allowed_historical_features_pass_schema_validation(tmp_path):
    _write_schema(tmp_path, ["cvd", "obi", "micro_atr"])

    factory = V19FeatureFactory(str(tmp_path))

    assert factory.stat_features == ["cvd", "obi", "micro_atr"]
