"""Core causal LOB features for MBP snapshots."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(np.float64)


def compute_lob_features(df: pd.DataFrame, *, levels: int = 10) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    bid0 = _num(df, "bid_px_00")
    ask0 = _num(df, "ask_px_00")
    bid_sz0 = _num(df, "bid_sz_00")
    ask_sz0 = _num(df, "ask_sz_00")
    valid_bbo = (bid0 > 0) & (ask0 > bid0)

    out["mid_price"] = np.where(valid_bbo, (bid0 + ask0) / 2.0, np.nan)
    out["spread"] = np.where(valid_bbo, ask0 - bid0, np.nan)
    out["microprice"] = np.where(
        valid_bbo & ((bid_sz0 + ask_sz0) > 0),
        (ask0 * bid_sz0 + bid0 * ask_sz0) / np.maximum(bid_sz0 + ask_sz0, 1e-12),
        np.nan,
    )

    bid_depth = np.zeros(len(df), dtype=np.float64)
    ask_depth = np.zeros(len(df), dtype=np.float64)
    weighted_bid = np.zeros(len(df), dtype=np.float64)
    weighted_ask = np.zeros(len(df), dtype=np.float64)
    for level in range(int(levels)):
        weight = 1.0 / float(level + 1)
        bsz = _num(df, f"bid_sz_{level:02d}")
        asz = _num(df, f"ask_sz_{level:02d}")
        bid_depth += bsz
        ask_depth += asz
        weighted_bid += bsz * weight
        weighted_ask += asz * weight

    total_depth = np.maximum(bid_depth + ask_depth, 1e-12)
    weighted_total = np.maximum(weighted_bid + weighted_ask, 1e-12)
    out["bid_depth"] = bid_depth
    out["ask_depth"] = ask_depth
    out["depth_imbalance"] = (bid_depth - ask_depth) / total_depth
    out["order_book_imbalance"] = (weighted_bid - weighted_ask) / weighted_total
    out["liquidity_pressure"] = out["order_book_imbalance"] / np.maximum(out["spread"].fillna(np.inf), 1e-12)
    return out.astype(np.float32, errors="ignore")
