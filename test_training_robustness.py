import pandas as pd

from prepare_training_data import _select_event_rich_lob_emit_positions
from train_v19 import _resolve_meta_learner_profile


def test_meta_learner_profile_compacts_on_small_data():
    profile = _resolve_meta_learner_profile(train_sequences=1837, visual_seq_coverage=0.80)

    assert profile["name"] == "compact"
    assert profile["lstm_units_1"] == 48
    assert profile["lstm_units_2"] == 24
    assert profile["force_rebuild"] is True
    assert profile["visual_dropout_rate"] >= 0.35


def test_emit_positions_expand_neighbors_when_events_collapse_to_few_snapshots():
    df_labeled = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                [
                    "2025-01-01 00:00:31",
                    "2025-01-01 00:00:32",
                    "2025-01-01 00:00:33",
                    "2025-01-01 00:00:34",
                    "2025-01-01 00:00:35",
                    "2025-01-01 00:00:36",
                ]
            ),
            "event_flag": [1] * 6,
            "train_event_flag": [1] * 6,
            "bias_label": [0] * 6,
            "signal_quality": [2] * 6,
        }
    )
    lob_mbp_src = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                [
                    "2025-01-01 00:00:00",
                    "2025-01-01 00:00:10",
                    "2025-01-01 00:00:20",
                    "2025-01-01 00:00:30",
                    "2025-01-01 00:00:40",
                    "2025-01-01 00:00:50",
                    "2025-01-01 00:01:00",
                ]
            )
        }
    )

    emit_positions, meta = _select_event_rich_lob_emit_positions(
        df_labeled,
        lob_mbp_src,
        max_events=6,
        max_positions=3,
    )

    assert meta["base_emit_positions"] == 1
    assert meta["emit_neighbor_radius"] == 2
    assert meta["emit_neighbor_positions_capped"] is True
    assert meta["selected_emit_positions"] == 3
    assert len(emit_positions) == 3
