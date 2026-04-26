import numpy as np
import pandas as pd

from backtest_v19 import (
    _build_visual_diagnostics,
    _filter_backtest_window,
    _load_meta_features,
    _load_visual_embeddings,
)
from modules.html_reporter import generate_backtest_report


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


def test_visual_diagnostics_explain_training_vs_backtest_denominator_gap(tmp_path):
    df = _sample_df()
    visual = np.zeros((len(df), 2), dtype=np.float32)
    visual[1] = [1.0, 0.5]
    results_df = pd.DataFrame(
        {
            "tradeable": [0, 1, 1, 0],
            "executed": [0, 1, 0, 0],
        }
    )
    metrics_path = tmp_path / "visual_metrics_v19.json"
    metrics_path.write_text(
        """
{
  "n_rows": 2,
  "rows_with_tensor": 2,
  "coverage_ratio": 0.8,
  "n_tensors": 2
}
        """.strip()
    )

    diag = _build_visual_diagnostics(
        df,
        visual,
        models_dir=str(tmp_path),
        results_df=results_df,
        eval_visual_diagnostics={
            "source": "cnn_eval",
            "reason": "ok",
            "rows_with_tensor": 1,
            "used_tensor_count": 1,
        },
    )

    assert diag["coverage_ratio"] == 0.25
    assert diag["tradeable_coverage_ratio"] == 0.5
    assert diag["coverage_gap_vs_train"] == -0.55
    assert any("event rows" in note for note in diag["diagnosis_notes"])


def test_backtest_html_report_includes_visual_diagnostics(tmp_path):
    trades = [
        {
            "result": "WIN",
            "pips": 12.5,
            "pnl": 125.0,
            "tp": 15,
            "sl": 10,
            "dur_min": 4,
            "dir": "LONG",
            "ep": 1.1010,
            "xp": 1.10225,
            "regime": "Trending",
            "conf": 0.72,
        },
        {
            "result": "LOSE",
            "pips": -8.0,
            "pnl": -80.0,
            "tp": 15,
            "sl": 10,
            "dur_min": 7,
            "dir": "SHORT",
            "ep": 1.1030,
            "xp": 1.1038,
            "regime": "Ranging",
            "conf": 0.63,
        },
    ]
    equity = [100000.0, 100125.0, 100045.0]
    path = generate_backtest_report(
        output_dir=str(tmp_path),
        trades=trades,
        equity=equity,
        n_test_bars=12,
        model_acc=0.41,
        n_features=45,
        n_dataset=20,
        backtest_summary={
            "predictions": 12,
            "trades": 2,
            "win_rate": 0.5,
            "profit_factor": 1.56,
            "trade_sharpe": 0.22,
            "directional_f1_macro": 0.41,
            "visual_coverage": 0.25,
        },
        visual_diagnostics={
            "source": "cnn_eval",
            "reason": "ok",
            "rows_total": 20,
            "rows_with_visual": 5,
            "coverage_ratio": 0.25,
            "event_rows": 6,
            "event_rows_with_visual": 4,
            "event_coverage_ratio": 0.6667,
            "train_event_rows": 4,
            "train_event_rows_with_visual": 3,
            "train_event_coverage_ratio": 0.75,
            "directional_train_event_rows": 4,
            "directional_train_event_rows_with_visual": 3,
            "directional_train_event_coverage_ratio": 0.75,
            "tradeable_rows": 2,
            "tradeable_rows_with_visual": 1,
            "tradeable_coverage_ratio": 0.5,
            "executed_rows": 1,
            "executed_rows_with_visual": 1,
            "executed_coverage_ratio": 1.0,
            "rows_with_tensor": 5,
            "used_tensor_count": 3,
            "diagnosis_notes": ["Training coverage uses event rows only."],
        },
    )

    assert path is not None
    html = (tmp_path / "V19_Backtest_Report.html").read_text(encoding="utf-8")
    assert "Visual Coverage Diagnostics" in html
    assert "Training coverage uses event rows only." in html
