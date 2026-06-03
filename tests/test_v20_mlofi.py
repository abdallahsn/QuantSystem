import numpy as np
import pandas as pd

from feature_engineering.ofi import compute_mlofi


def _frame():
    return pd.DataFrame(
        {
            "ts_event": pd.date_range("2026-01-01", periods=5, freq="s"),
            "bid_px_00": [1.2500, 1.2500, 1.2501, 1.2501, 1.2500],
            "ask_px_00": [1.2501, 1.2501, 1.2502, 1.2502, 1.2501],
            "bid_sz_00": [10, 14, 8, 9, 7],
            "ask_sz_00": [11, 9, 9, 7, 12],
            "bid_px_01": [1.2499, 1.2499, 1.2500, 1.2500, 1.2499],
            "ask_px_01": [1.2502, 1.2502, 1.2503, 1.2503, 1.2502],
            "bid_sz_01": [8, 8, 12, 10, 9],
            "ask_sz_01": [7, 10, 8, 8, 10],
        }
    )


def test_mlofi_first_row_is_zero_and_uses_previous_snapshot():
    features = compute_mlofi(_frame(), levels=2)

    assert np.isclose(float(features["mlofi_00"].iloc[0]), 0.0)
    assert np.isclose(float(features["mlofi_00"].iloc[1]), 6.0)
    assert "mlofi_sum" in features.columns
    assert "mlofi_top3" in features.columns


def test_mlofi_prefix_values_do_not_change_when_future_rows_change():
    base = _frame()
    changed_future = base.copy()
    changed_future.loc[3:, "bid_sz_00"] = 999
    changed_future.loc[3:, "ask_sz_00"] = 999

    original = compute_mlofi(base, levels=2)
    mutated = compute_mlofi(changed_future, levels=2)

    pd.testing.assert_series_equal(original["mlofi_sum"].iloc[:3], mutated["mlofi_sum"].iloc[:3])
