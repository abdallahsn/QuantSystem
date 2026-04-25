import numpy as np
import pandas as pd

from backtest_v19 import _filter_backtest_window, _load_meta_features, _load_visual_embeddings


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                [
                    "2025-01-15 09:00:00",
                    "2025-01-15 09:00:01",
                    "2025-01-15 09:00:02",
                    "2025-01-15 09:00:03",
                ]
            ),
            "train_event_flag": [0, 1, 1, 0],
            "event_flag": [0, 1, 1, 0],
            "bias_label": [2, 0, 1, 2],
        }
    )


def test_visual_embeddings_expand_from_directional_event_rows(tmp_path):
    df = _sample_df()
    vis = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    path = tmp_path / "visual.npy"
    np.save(path, vis)

    out = _load_visual_embeddings(
        df,
        explicit_path=str(path),
        default_path=None,
        expected_dim=2,
        models_dir=str(tmp_path),
    )

    assert out.shape == (len(df), 2)
    assert np.allclose(out[0], [0.0, 0.0])
    assert np.allclose(out[1], [1.0, 2.0])
    assert np.allclose(out[2], [3.0, 4.0])
    assert np.allclose(out[3], [0.0, 0.0])


def test_visual_embeddings_expand_compact_rows_via_coverage_sidecar(tmp_path):
    df = _sample_df()
    vis = np.array([[7.0, 8.0]], dtype=np.float32)
    path = tmp_path / "visual_embeddings_v19.npy"
    cov_path = tmp_path / "visual_coverage_v19.npy"
    np.save(path, vis)
    np.save(cov_path, np.array([1, 0], dtype=np.uint8))

    out = _load_visual_embeddings(
        df,
        explicit_path=str(path),
        default_path=None,
        expected_dim=2,
        models_dir=str(tmp_path),
    )

    assert out.shape == (len(df), 2)
    assert np.allclose(out[0], [0.0, 0.0])
    assert np.allclose(out[1], [7.0, 8.0])
    assert np.allclose(out[2], [0.0, 0.0])
    assert np.allclose(out[3], [0.0, 0.0])


def test_meta_features_expand_from_directional_event_rows(tmp_path):
    df = _sample_df()
    meta = np.array(
        [
            [0.9, 0.1, 1.0, 0.0, 0.0, 0.0],
            [0.2, 0.8, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    path = tmp_path / "meta.npy"
    np.save(path, meta)

    out = _load_meta_features(
        df,
        explicit_path=str(path),
        expected_dim=6,
        allow_in_sample_live_override=True,
    )

    assert out.shape == (len(df), 6)
    assert np.allclose(out[0], np.zeros(6, dtype=np.float32))
    assert np.allclose(out[1], meta[0])
    assert np.allclose(out[2], meta[1])
    assert np.allclose(out[3], np.zeros(6, dtype=np.float32))


def test_filter_backtest_window_respects_bounds():
    df = _sample_df()
    out = _filter_backtest_window(
        df,
        start_ts="2025-01-15 09:00:01",
        end_ts="2025-01-15 09:00:03",
    )
    assert len(out) == 2
    assert out["ts_event"].iloc[0] == pd.Timestamp("2025-01-15 09:00:01")
    assert out["ts_event"].iloc[-1] == pd.Timestamp("2025-01-15 09:00:02")
