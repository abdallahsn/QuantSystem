"""Order-flow imbalance features, including true multi-level OFI."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _arr(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)


def _level_ofi(bid_px: np.ndarray, bid_sz: np.ndarray, ask_px: np.ndarray, ask_sz: np.ndarray) -> np.ndarray:
    prev_bid_px = np.r_[bid_px[0], bid_px[:-1]]
    prev_bid_sz = np.r_[bid_sz[0], bid_sz[:-1]]
    prev_ask_px = np.r_[ask_px[0], ask_px[:-1]]
    prev_ask_sz = np.r_[ask_sz[0], ask_sz[:-1]]

    bid_flow = np.where(
        bid_px > prev_bid_px,
        bid_sz,
        np.where(bid_px == prev_bid_px, bid_sz - prev_bid_sz, -prev_bid_sz),
    )
    ask_flow = np.where(
        ask_px < prev_ask_px,
        ask_sz,
        np.where(ask_px == prev_ask_px, ask_sz - prev_ask_sz, -prev_ask_sz),
    )
    return bid_flow - ask_flow


def compute_mlofi(df: pd.DataFrame, *, levels: int = 10, prefix: str = "mlofi") -> pd.DataFrame:
    """Compute causal MLOFI vectors from current and previous MBP snapshots only."""
    out = pd.DataFrame(index=df.index)
    total = np.zeros(len(df), dtype=np.float64)
    for level in range(int(levels)):
        ofi = _level_ofi(
            _arr(df, f"bid_px_{level:02d}"),
            _arr(df, f"bid_sz_{level:02d}"),
            _arr(df, f"ask_px_{level:02d}"),
            _arr(df, f"ask_sz_{level:02d}"),
        )
        out[f"{prefix}_{level:02d}"] = ofi.astype(np.float32)
        total += ofi
    out[f"{prefix}_sum"] = total.astype(np.float32)
    out[f"{prefix}_top3"] = out[[f"{prefix}_{i:02d}" for i in range(min(3, int(levels)))]].sum(axis=1).astype(np.float32)
    return out
