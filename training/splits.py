"""Purged walk-forward split generation for QuantSystem v20."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedWalkForwardConfig:
    n_folds: int = 5
    initial_train_frac: float = 0.50
    test_frac: float = 0.10
    embargo_rows: int = 0
    min_train_rows: int = 100
    min_test_rows: int = 20

    def to_dict(self) -> dict:
        return asdict(self)


def _ts(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)
    if out.isna().any():
        raise ValueError(f"Invalid split timestamp rows: {int(out.isna().sum())}")
    return out.reset_index(drop=True)


def build_purged_walkforward_splits(
    ts_event: pd.Series,
    label_end_ts: pd.Series,
    config: PurgedWalkForwardConfig | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    cfg = config or PurgedWalkForwardConfig()
    t0 = _ts(ts_event)
    t1 = _ts(label_end_ts)
    n = len(t0)
    if n != len(t1):
        raise ValueError("ts_event and label_end_ts length mismatch")
    order = np.argsort(t0.to_numpy(dtype="datetime64[ns]"), kind="mergesort")
    t0 = t0.iloc[order].reset_index(drop=True)
    t1 = t1.iloc[order].reset_index(drop=True)

    base_train = max(int(n * float(cfg.initial_train_frac)), int(cfg.min_train_rows))
    test_rows = max(int(n * float(cfg.test_frac)), int(cfg.min_test_rows))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    fold_reports: list[dict] = []

    for fold in range(int(cfg.n_folds)):
        test_start_pos = base_train + fold * test_rows
        test_end_pos = min(test_start_pos + test_rows, n)
        if test_start_pos >= n or (test_end_pos - test_start_pos) < int(cfg.min_test_rows):
            break
        test_start_ts = t0.iloc[test_start_pos]
        test_end_ts = t0.iloc[test_end_pos - 1]
        train_mask = (t0 < test_start_ts) & (t1 < test_start_ts)
        if int(cfg.embargo_rows) > 0:
            embargo_start = max(0, test_start_pos - int(cfg.embargo_rows))
            train_mask.iloc[embargo_start:test_start_pos] = False
        train_idx_sorted = np.flatnonzero(train_mask.to_numpy(dtype=bool))
        test_idx_sorted = np.arange(test_start_pos, test_end_pos, dtype=np.int64)
        train_idx = order[train_idx_sorted].astype(np.int64)
        test_idx = order[test_idx_sorted].astype(np.int64)
        if len(train_idx) < int(cfg.min_train_rows):
            continue
        splits.append((train_idx, test_idx))
        fold_reports.append(
            {
                "fold": int(len(splits)),
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "test_start_ts": str(test_start_ts),
                "test_end_ts": str(test_end_ts),
                "max_train_label_end_ts": str(t1.iloc[train_idx_sorted].max()) if len(train_idx_sorted) else None,
                "embargo_rows": int(cfg.embargo_rows),
            }
        )

    report = {
        "config": cfg.to_dict(),
        "rows": int(n),
        "fold_count": int(len(splits)),
        "folds": fold_reports,
        "passed": bool(len(splits) > 0),
    }
    return splits, report
