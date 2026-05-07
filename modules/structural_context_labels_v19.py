"""
structural_context_labels_v19.py
================================
Causal structural context derived at label time (V19).

- Range-style context uses **only information available at row i** (Kalman trend
  label + Kalman trend strength). This is *not* a crystal-ball regime classifier;
  it flags “weak / non-directional” tape suitable for range-aware policies.

- Wall forward deltas are **supervised targets** (they use future wall strength
  within a fixed horizon). Safe for training auxiliary heads; do **not** feed
  them as live features without shifting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def append_kalman_range_context(
    df: pd.DataFrame,
    trend_lbl: np.ndarray,
    trend_strength: np.ndarray,
    *,
    strength_max: float = 0.10,
    require_kalman_neutral: bool = True,
    trend_neutral_code: int,
) -> pd.DataFrame:
    """
    Add range-context columns (all causal at time i).

    Columns
    -------
    range_kalman_weak      int8  : 1 if kalman_trend_strength <= strength_max
    range_kalman_neutral   int8  : 1 if trend label == NEUTRAL
    range_ctx              int8  : combined “ranging-like tape” flag
    range_run_len          int32 : bars so far in current True run of range_ctx
    range_start            int8  : first bar of a new range_ctx run
    range_end              int8  : first bar after a range_ctx run (exit)
    """
    out = df
    n = len(out)
    ts = np.asarray(trend_strength, dtype=np.float32).reshape(-1)[:n]
    tl = np.asarray(trend_lbl, dtype=np.int8).reshape(-1)[:n]

    weak = (ts <= float(max(strength_max, 0.0))).astype(np.int8)
    neutral = (tl == int(trend_neutral_code)).astype(np.int8)
    if require_kalman_neutral:
        ctx = (weak & neutral).astype(np.int8)
    else:
        ctx = (weak | neutral).astype(np.int8)

    out = out.copy()
    out["range_kalman_weak"] = weak
    out["range_kalman_neutral"] = neutral
    out["range_ctx"] = ctx

    m = ctx.astype(bool)
    run_len = np.zeros(n, dtype=np.int32)
    c = 0
    for i in range(n):
        if m[i]:
            c += 1
            run_len[i] = c
        else:
            c = 0
    out["range_run_len"] = run_len

    prev = np.zeros(n, dtype=bool)
    prev[1:] = m[:-1]
    start = m & ~prev
    end = (~m) & prev
    out["range_start"] = start.astype(np.int8)
    out["range_end"] = end.astype(np.int8)
    return out


def append_wall_forward_deltas(
    df: pd.DataFrame,
    *,
    fwd_steps: int,
    bid_col: str = "bid_wall_strength",
    ask_col: str = "ask_wall_strength",
) -> pd.DataFrame:
    """
    Forward wall deltas over a **fixed** horizon k (supervision targets).

    For row i uses strength at i+k - strength at i (last k rows are NaN).
    """
    if fwd_steps <= 0:
        return df
    out = df.copy()
    n = len(out)
    k = int(min(max(fwd_steps, 0), max(n - 1, 0)))
    if k <= 0 or n == 0:
        return out

    def _col_delta(name: str, out_name: str) -> None:
        if name not in out.columns:
            return
        s = pd.to_numeric(out[name], errors="coerce").astype(np.float64).to_numpy()
        delta = np.full(n, np.nan, dtype=np.float64)
        delta[: n - k] = s[k:] - s[: n - k]
        out[out_name] = delta.astype(np.float32)

    _col_delta(bid_col, "bid_wall_delta_fwd_k")
    _col_delta(ask_col, "ask_wall_delta_fwd_k")
    return out
