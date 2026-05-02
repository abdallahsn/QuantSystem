"""
oof_stacking.py - Shared helpers for leakage-safe stacked models
"""

from __future__ import annotations

import numpy as np


def align_probability_columns(
    probs: np.ndarray,
    n_outputs: int,
    classes: np.ndarray | list | tuple | None = None,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Expand/crop probability columns to a fixed output width.

    This is useful when a fold-trained classifier sees only a subset of the
    global classes and therefore returns fewer columns from ``predict_proba``.
    When ``classes`` is provided, each input column is mapped back to its
    global class index.
    """
    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)

    if probs.ndim != 2:
        raise ValueError(f"Expected 2D probabilities, got shape {probs.shape}")

    if probs.shape[1] == n_outputs and classes is None:
        return probs.astype(np.float32, copy=False)

    out = np.full((probs.shape[0], n_outputs), fill_value, dtype=np.float32)
    if probs.shape[1] == 0 or n_outputs == 0:
        return out

    if classes is None:
        width = min(probs.shape[1], n_outputs)
        out[:, :width] = probs[:, :width]
    else:
        cls_arr = np.asarray(classes).reshape(-1)
        if len(cls_arr) != probs.shape[1]:
            raise ValueError(
                "Probability columns do not match class labels: "
                f"{probs.shape[1]} cols vs {len(cls_arr)} classes"
            )
        for src_col, cls in enumerate(cls_arr):
            try:
                cls_idx = int(cls)
            except (TypeError, ValueError):
                continue
            if 0 <= cls_idx < n_outputs:
                out[:, cls_idx] = probs[:, src_col]

    row_sums = out.sum(axis=1, keepdims=True)
    valid = row_sums[:, 0] > 0
    if np.any(valid):
        out[valid] = out[valid] / row_sums[valid]
    return out


def run_sequential_oof(
    n_samples: int,
    n_outputs: int,
    splits,
    predictor_fn,
    fill_value: float = np.nan,
):
    """
    Run sequential out-of-fold predictions over precomputed time splits.

    predictor_fn signature:
        predictor_fn(train_idx, test_idx, fold_number) -> (predictions, fold_info)
    """
    out = np.full((n_samples, n_outputs), fill_value, dtype=np.float32)
    covered = np.zeros(n_samples, dtype=bool)
    reports = []

    for fold_no, (train_idx, test_idx) in enumerate(splits, start=1):
        test_idx = np.asarray(test_idx, dtype=np.int64)
        if test_idx.ndim != 1:
            raise ValueError(f"OOF test_idx must be 1D, got shape {test_idx.shape}")
        if len(np.unique(test_idx)) != len(test_idx):
            raise ValueError(f"OOF split {fold_no} contains duplicate test rows")
        if np.any(test_idx < 0) or np.any(test_idx >= n_samples):
            raise ValueError(f"OOF split {fold_no} has out-of-range test rows")
        if np.any(covered[test_idx]):
            overlap_rows = test_idx[covered[test_idx]].tolist()[:10]
            raise ValueError(
                f"OOF split {fold_no} overlaps previously covered rows: sample={overlap_rows}"
            )
        preds, info = predictor_fn(train_idx, test_idx, fold_no)
        preds = np.asarray(preds, dtype=np.float32)
        if preds.shape != (len(test_idx), n_outputs):
            raise ValueError(
                f"OOF predictor returned {preds.shape}, expected {(len(test_idx), n_outputs)}"
            )
        out[test_idx] = preds
        covered[test_idx] = True
        info = info or {}
        info['fold'] = fold_no
        info['train_size'] = int(len(train_idx))
        info['test_size'] = int(len(test_idx))
        reports.append(info)

    return out, covered, reports


def fill_uncovered_probabilities(
    probs: np.ndarray,
    covered: np.ndarray,
    priors: np.ndarray | None = None,
) -> np.ndarray:
    out = np.asarray(probs, dtype=np.float32).copy()
    if priors is None:
        priors = np.ones(out.shape[1], dtype=np.float32) / max(out.shape[1], 1)
    priors = np.asarray(priors, dtype=np.float32).reshape(1, -1)
    out[~covered] = priors
    return out


def fill_uncovered_one_hot(
    one_hot: np.ndarray,
    covered: np.ndarray,
    default_class: int = 0,
) -> np.ndarray:
    out = np.asarray(one_hot, dtype=np.float32).copy()
    if out.shape[1] == 0:
        return out
    default_class = int(np.clip(default_class, 0, out.shape[1] - 1))
    out[~covered] = 0.0
    out[~covered, default_class] = 1.0
    return out
