from __future__ import annotations

import pandas as pd

from modules.feature_factory_v19 import (
    ROBUST_IQR_MIN,
    apply_scaler_params_to_frame,
    fit_numeric_scaler_param,
)


def test_fit_numeric_scaler_param_falls_back_to_minmax_for_tiny_iqr():
    series = pd.Series(
        [
            1.00000,
            1.00001,
            1.00002,
            1.00003,
            1.00004,
            1.00005,
        ],
        dtype="float32",
    )

    param = fit_numeric_scaler_param(series, robust_iqr_min=ROBUST_IQR_MIN)

    assert param["type"] == "minmax"
    assert param["fallback_from"] == "low_iqr"

    scaled = apply_scaler_params_to_frame(
        pd.DataFrame({"feature": series}),
        {"feature": param},
        clip_range=(-10.0, 10.0),
    )["feature"]

    assert float(scaled.min()) >= -1.000001
    assert float(scaled.max()) <= 1.000001


def test_apply_scaler_params_uses_iqr_floor_for_legacy_robust_scalers():
    frame = pd.DataFrame({"feature": [1.0, 1.00005]}, dtype="float32")
    scaled = apply_scaler_params_to_frame(
        frame,
        {
            "feature": {
                "type": "robust",
                "median": 1.0,
                "iqr": 1e-6,
            }
        },
        clip_range=(-10.0, 10.0),
    )["feature"]

    assert float(scaled.iloc[0]) == 0.0
    assert 0.0 < float(scaled.iloc[1]) < 1.0
