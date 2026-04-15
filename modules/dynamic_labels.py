"""
dynamic_labels.py — V2: Unified Pipeline (OrderBook → Features → Labels)
========================================================================
This module upgrades the old dynamic labeling path into a unified pipeline:

  1. OrderWallScanner returns a dense feature vector, not just TP/SL anchors
  2. Labels are multi-layered: direction + quality + regime
  3. Event-based filtering suppresses quiet/noisy rows before window building
  4. Feature engineering happens on the full frame before slicing windows
  5. TP/SL fallbacks adapt to current liquidity context rather than fixed levels
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


LABEL_WIN = 1
LABEL_LOSE = 0
LABEL_CANCEL = -1

DIR_LONG = 0
DIR_SHORT = 1
DIR_NEUTRAL = 2

QUALITY_STRONG = 2
QUALITY_WEAK = 1
QUALITY_NONE = 0

REGIME_TRENDING = 1
REGIME_RANGING = 0


@dataclass
class MarketFeatureVector:
    """
    Compact order-book feature vector for each tick.
    """

    obi: float = 0.0
    cvd: float = 0.0
    volume: float = 0.0
    micro_price: float = 0.0
    bid_wall_strength: float = 0.0
    ask_wall_strength: float = 0.0
    distance_to_wall: float = 0.0
    gap_size: float = 0.0
    liquidity_density: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.obi,
                self.cvd,
                self.volume,
                self.micro_price,
                self.bid_wall_strength,
                self.ask_wall_strength,
                self.distance_to_wall,
                self.gap_size,
                self.liquidity_density,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "obi",
            "cvd",
            "volume",
            "micro_price",
            "bid_wall_strength",
            "ask_wall_strength",
            "distance_to_wall",
            "gap_size",
            "liquidity_density",
        ]


class OrderWallScanner:
    """
    Scan MBP10 snapshots and return:
      - walls/gaps used for dynamic TP/SL
      - a unified feature vector usable by training
    """

    def __init__(
        self,
        wall_mult: float = 3.0,
        gap_mult: float = 2.0,
        levels: int = 10,
        hist_size: int = 200,
    ):
        self.wall_mult = wall_mult
        self.gap_mult = gap_mult
        self.levels = levels
        self._size_hist = deque(maxlen=hist_size)
        self._vol_hist = deque(maxlen=hist_size)

    def scan(
        self,
        row: dict,
        tick_size: float = 0.0001,
        cvd: float = 0.0,
        volume: float = 0.0,
    ) -> dict:
        bid_px, bid_sz = [], []
        ask_px, ask_sz = [], []

        for i in range(self.levels):
            bp = float(row.get(f"bid_px_{i:02d}", 0) or 0)
            bs = float(row.get(f"bid_sz_{i:02d}", 0) or 0)
            ap = float(row.get(f"ask_px_{i:02d}", 0) or 0)
            az = float(row.get(f"ask_sz_{i:02d}", 0) or 0)
            if bp > 0 and bs > 0:
                bid_px.append(bp)
                bid_sz.append(bs)
            if ap > 0 and az > 0:
                ask_px.append(ap)
                ask_sz.append(az)

        if not bid_sz or not ask_sz:
            return self._empty_result()

        tick = max(float(tick_size or 0.0), 1e-9)
        all_sz = bid_sz + ask_sz
        self._size_hist.extend(all_sz)
        self._vol_hist.append(float(volume))

        mean_sz = float(np.mean(self._size_hist)) if self._size_hist else float(np.mean(all_sz))
        mean_sz = max(mean_sz, 1e-9)
        wall_thr = mean_sz * self.wall_mult

        best_bid = float(bid_px[0])
        best_ask = float(ask_px[0])
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2.0
        micro_px = (best_bid * bid_sz[0] + best_ask * ask_sz[0]) / max(bid_sz[0] + ask_sz[0], 1e-9)

        top_n = min(5, len(bid_sz), len(ask_sz))
        bid_top = sum(bid_sz[:top_n])
        ask_top = sum(ask_sz[:top_n])
        obi = (bid_top - ask_top) / max(bid_top + ask_top, 1e-9)

        bid_wall_px = None
        bid_wall_str = 0.0
        for px, sz in zip(bid_px, bid_sz):
            if sz >= wall_thr:
                bid_wall_px = float(px)
                bid_wall_str = float(sz) / mean_sz
                break

        ask_wall_px = None
        ask_wall_str = 0.0
        for px, sz in zip(ask_px, ask_sz):
            if sz >= wall_thr:
                ask_wall_px = float(px)
                ask_wall_str = float(sz) / mean_sz
                break

        bid_gap_px = None
        bid_gap_size = 0.0
        for i in range(len(bid_px) - 1):
            gap = abs(float(bid_px[i]) - float(bid_px[i + 1]))
            if gap > tick * self.gap_mult:
                bid_gap_px = float(bid_px[i + 1])
                bid_gap_size = gap / tick
                break

        ask_gap_px = None
        ask_gap_size = 0.0
        for i in range(len(ask_px) - 1):
            gap = abs(float(ask_px[i]) - float(ask_px[i + 1]))
            if gap > tick * self.gap_mult:
                ask_gap_px = float(ask_px[i + 1])
                ask_gap_size = gap / tick
                break

        dist_bid = (mid - bid_wall_px) / tick if bid_wall_px else 0.0
        dist_ask = (ask_wall_px - mid) / tick if ask_wall_px else 0.0
        valid_dists = [d for d in (dist_bid, dist_ask) if d > 0]
        distance_to_wall = min(valid_dists) if valid_dists else 0.0
        gap_size = max(bid_gap_size, ask_gap_size)

        top_bid_idx = min(4, len(bid_px) - 1)
        top_ask_idx = min(4, len(ask_px) - 1)
        price_range_pips = max(
            (float(ask_px[top_ask_idx]) - float(ask_px[0])) / tick
            + (float(bid_px[0]) - float(bid_px[top_bid_idx])) / tick,
            1.0,
        )
        near_sz = float(sum(bid_sz[:5]) + sum(ask_sz[:5]))
        liquidity_density = near_sz / price_range_pips

        fv = MarketFeatureVector(
            obi=round(float(obi), 6),
            cvd=float(cvd),
            volume=float(volume),
            micro_price=round(float(micro_px), 6),
            bid_wall_strength=round(float(bid_wall_str), 4),
            ask_wall_strength=round(float(ask_wall_str), 4),
            distance_to_wall=round(float(distance_to_wall), 2),
            gap_size=round(float(gap_size), 2),
            liquidity_density=round(float(liquidity_density), 4),
        )

        return {
            "mid_price": round(float(mid), 6),
            "micro_price": round(float(micro_px), 6),
            "obi": round(float(obi), 6),
            "spread": round(float(spread), 6),
            "bid_wall_px": bid_wall_px,
            "ask_wall_px": ask_wall_px,
            "bid_wall_str": round(float(bid_wall_str), 4),
            "ask_wall_str": round(float(ask_wall_str), 4),
            "bid_wall_strength": round(float(bid_wall_str), 4),
            "ask_wall_strength": round(float(ask_wall_str), 4),
            "bid_gap_px": bid_gap_px,
            "ask_gap_px": ask_gap_px,
            "bid_gap_size": round(float(bid_gap_size), 2),
            "ask_gap_size": round(float(ask_gap_size), 2),
            "distance_to_wall": round(float(distance_to_wall), 2),
            "gap_size": round(float(gap_size), 2),
            "liquidity_density": round(float(liquidity_density), 4),
            "feature_vector": fv,
        }

    def _empty_result(self) -> dict:
        return {
            "mid_price": 0.0,
            "micro_price": 0.0,
            "obi": 0.0,
            "spread": 0.0,
            "bid_wall_px": None,
            "ask_wall_px": None,
            "bid_wall_str": 0.0,
            "ask_wall_str": 0.0,
            "bid_wall_strength": 0.0,
            "ask_wall_strength": 0.0,
            "bid_gap_px": None,
            "ask_gap_px": None,
            "bid_gap_size": 0.0,
            "ask_gap_size": 0.0,
            "distance_to_wall": 0.0,
            "gap_size": 0.0,
            "liquidity_density": 0.0,
            "feature_vector": MarketFeatureVector(),
        }


def engineer_features(df: pd.DataFrame, roll_window: int = 20) -> pd.DataFrame:
    """
    Apply feature engineering on the full frame before slicing windows.
    """

    df = df.copy()
    base_cols = MarketFeatureVector.feature_names()

    for col in base_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df[f"{col}_diff"] = series.diff().fillna(0.0)

        roll_mean = series.rolling(roll_window, min_periods=1).mean()
        roll_std = series.rolling(roll_window, min_periods=1).std().replace(0, 1e-9)
        df[f"{col}_zscore"] = ((series - roll_mean) / roll_std).fillna(0.0)
        df[f"{col}_rmean"] = roll_mean.fillna(0.0)

    if "obi" in df.columns:
        obi_std = pd.to_numeric(df["obi"], errors="coerce").fillna(0.0).rolling(roll_window, min_periods=1).std().fillna(0.0)
        df["regime"] = (obi_std > float(obi_std.median())).astype(np.int8)
    else:
        df["regime"] = REGIME_RANGING

    return df


def build_event_filter(
    df: pd.DataFrame,
    vol_mult: float = 1.5,
    obi_thr: float = 0.3,
    wall_str_thr: float = 1.0,
) -> pd.Series:
    """
    Important-event mask driven by current activity, imbalance, and wall strength.
    """

    if "volume" in df.columns:
        volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    elif "size" in df.columns:
        volume = pd.to_numeric(df["size"], errors="coerce").fillna(0.0)
    else:
        volume = pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)

    roll_vol = volume.rolling(50, min_periods=1).mean()
    cond_vol = volume > (roll_vol * float(vol_mult))

    if "obi" in df.columns:
        cond_obi = pd.to_numeric(df["obi"], errors="coerce").fillna(0.0).abs() > float(obi_thr)
    else:
        cond_obi = pd.Series(False, index=df.index)

    if "bid_wall_strength" in df.columns and "ask_wall_strength" in df.columns:
        bid_wall = pd.to_numeric(df["bid_wall_strength"], errors="coerce").fillna(0.0)
        ask_wall = pd.to_numeric(df["ask_wall_strength"], errors="coerce").fillna(0.0)
        cond_wall = (bid_wall > float(wall_str_thr)) | (ask_wall > float(wall_str_thr))
    else:
        cond_wall = pd.Series(False, index=df.index)

    zscore_cols = [
        col
        for col in ("obi_zscore", "cvd_zscore", "micro_price_zscore", "liquidity_density_zscore")
        if col in df.columns
    ]
    if zscore_cols:
        cond_shift = pd.Series(False, index=df.index)
        for col in zscore_cols:
            cond_shift = cond_shift | (pd.to_numeric(df[col], errors="coerce").fillna(0.0).abs() > 0.75)
    else:
        cond_shift = pd.Series(False, index=df.index)

    return (cond_vol | cond_obi | cond_wall | cond_shift).astype(bool)


def compute_dynamic_levels(
    scan_result: dict,
    current_price: float,
    tick_size: float = 0.0001,
    min_tp_pips: float = 15.0,
    max_sl_pips: float = 20.0,
) -> dict:
    """
    Dynamic TP/SL anchored to walls, gaps, and current liquidity.
    """

    mid = float(current_price or 0.0)
    pip = max(float(tick_size or 0.0), 1e-9)

    bid_wall_str = float(scan_result.get("bid_wall_strength", scan_result.get("bid_wall_str", 0.0)) or 0.0)
    ask_wall_str = float(scan_result.get("ask_wall_strength", scan_result.get("ask_wall_str", 0.0)) or 0.0)

    if scan_result.get("ask_gap_px") and float(scan_result["ask_gap_px"]) > mid:
        long_tp_raw = float(scan_result["ask_gap_px"]) - mid
        gap_bonus = float(scan_result.get("ask_gap_size", 0.0) or 0.0) * pip * 0.1
        long_tp = long_tp_raw + gap_bonus
    else:
        long_tp = pip * float(min_tp_pips)

    if scan_result.get("bid_wall_px") and float(scan_result["bid_wall_px"]) < mid:
        raw_sl = mid - float(scan_result["bid_wall_px"])
        str_factor = 1.0 + bid_wall_str * 0.05
        long_sl = raw_sl * str_factor
    else:
        long_sl = pip * (float(min_tp_pips) * 0.5)

    if scan_result.get("bid_gap_px") and float(scan_result["bid_gap_px"]) < mid:
        short_tp_raw = mid - float(scan_result["bid_gap_px"])
        gap_bonus = float(scan_result.get("bid_gap_size", 0.0) or 0.0) * pip * 0.1
        short_tp = short_tp_raw + gap_bonus
    else:
        short_tp = pip * float(min_tp_pips)

    if scan_result.get("ask_wall_px") and float(scan_result["ask_wall_px"]) > mid:
        raw_sl = float(scan_result["ask_wall_px"]) - mid
        str_factor = 1.0 + ask_wall_str * 0.05
        short_sl = raw_sl * str_factor
    else:
        short_sl = pip * (float(min_tp_pips) * 0.5)

    long_sl = min(long_sl, pip * float(max_sl_pips))
    short_sl = min(short_sl, pip * float(max_sl_pips))
    long_tp = max(long_tp, pip * float(min_tp_pips) * 0.5)
    short_tp = max(short_tp, pip * float(min_tp_pips) * 0.5)

    return {
        "long_tp": round(float(long_tp), 6),
        "long_sl": round(float(long_sl), 6),
        "short_tp": round(float(short_tp), 6),
        "short_sl": round(float(short_sl), 6),
        "long_rr": round(float(long_tp / max(long_sl, 1e-8)), 3),
        "short_rr": round(float(short_tp / max(short_sl, 1e-8)), 3),
        "long_wall_size": round(float(bid_wall_str), 4),
        "short_wall_size": round(float(ask_wall_str), 4),
    }


def append_dynamic_orderbook_features(
    df: pd.DataFrame,
    tick_size: float = 0.0001,
    price_col: str = "price",
    cvd_col: str = "cvd",
    volume_col: str = "size",
    wall_mult: float = 3.0,
    gap_mult: float = 2.0,
    levels: int = 10,
    min_tp_pips: float = 15.0,
    max_sl_pips: float = 20.0,
) -> pd.DataFrame:
    """
    Append scanner features + dynamic target levels to a frame.
    """

    out = df.copy()
    scanner = OrderWallScanner(wall_mult=wall_mult, gap_mult=gap_mult, levels=levels)
    records = []

    for row in out.itertuples(index=False):
        row_d = row._asdict()
        current_price = float(row_d.get(price_col, row_d.get("close", row_d.get("mid_price", 0.0))) or 0.0)
        scan_result = scanner.scan(
            row_d,
            tick_size=tick_size,
            cvd=float(row_d.get(cvd_col, 0.0) or 0.0),
            volume=float(row_d.get(volume_col, 0.0) or 0.0),
        )
        dyn = compute_dynamic_levels(
            scan_result,
            current_price=current_price if current_price > 0 else float(scan_result.get("mid_price", 0.0) or 0.0),
            tick_size=tick_size,
            min_tp_pips=min_tp_pips,
            max_sl_pips=max_sl_pips,
        )
        fv = scan_result["feature_vector"]
        records.append(
            {
                "mid_price": scan_result.get("mid_price", 0.0),
                "micro_price": scan_result.get("micro_price", 0.0),
                "spread": scan_result.get("spread", 0.0),
                "bid_wall_px": scan_result.get("bid_wall_px"),
                "ask_wall_px": scan_result.get("ask_wall_px"),
                "bid_gap_px": scan_result.get("bid_gap_px"),
                "ask_gap_px": scan_result.get("ask_gap_px"),
                "bid_gap_size": scan_result.get("bid_gap_size", 0.0),
                "ask_gap_size": scan_result.get("ask_gap_size", 0.0),
                "bid_wall_str": scan_result.get("bid_wall_str", 0.0),
                "ask_wall_str": scan_result.get("ask_wall_str", 0.0),
                "obi": float(fv.obi),
                "cvd": float(fv.cvd),
                "volume": float(fv.volume),
                "bid_wall_strength": float(fv.bid_wall_strength),
                "ask_wall_strength": float(fv.ask_wall_strength),
                "distance_to_wall": float(fv.distance_to_wall),
                "gap_size": float(fv.gap_size),
                "liquidity_density": float(fv.liquidity_density),
                **dyn,
            }
        )

    if not records:
        return out

    feat_df = pd.DataFrame(records, index=out.index)
    for col in feat_df.columns:
        out[col] = feat_df[col].values
    return out


def _detect_quality(df: pd.DataFrame, t: int, direction: int, window: int = 10) -> int:
    """
    Quality based on imbalance persistence and CVD alignment.
    """

    if direction == DIR_NEUTRAL:
        return QUALITY_NONE

    if "obi" not in df.columns or "cvd" not in df.columns:
        return QUALITY_WEAK

    start = max(0, int(t) - int(window))
    if t <= start:
        return QUALITY_WEAK

    obi_mean = pd.to_numeric(df["obi"].iloc[start:t], errors="coerce").fillna(0.0).mean()
    cvd_trend = pd.to_numeric(df["cvd"].iloc[start:t], errors="coerce").fillna(0.0).diff().fillna(0.0).mean()

    if direction == DIR_LONG:
        strong = (obi_mean > 0.2) and (cvd_trend > 0.0)
    else:
        strong = (obi_mean < -0.2) and (cvd_trend < 0.0)

    return QUALITY_STRONG if strong else QUALITY_WEAK


def label_with_forward_scan(
    df_trades: pd.DataFrame,
    df_levels: pd.DataFrame,
    max_bars_forward: int = 50,
    tick_size: float = 0.0001,
) -> pd.DataFrame:
    """
    Strict forward barrier scan with three-layer label outputs.
    """

    if len(df_trades) == 0:
        out = df_trades.copy()
        out["long_label"] = np.array([], dtype=np.int8)
        out["short_label"] = np.array([], dtype=np.int8)
        out["bias_label"] = np.array([], dtype=np.int8)
        out["signal_quality"] = np.array([], dtype=np.int8)
        out["regime"] = np.array([], dtype=np.int8)
        return out

    price_col = "close" if "close" in df_trades.columns else ("price" if "price" in df_trades.columns else "micro_price")
    prices = pd.to_numeric(df_trades[price_col], errors="coerce").ffill().fillna(0.0).values.astype(np.float64)
    n = len(prices)

    long_labels = np.full(n, LABEL_CANCEL, dtype=np.int8)
    short_labels = np.full(n, LABEL_CANCEL, dtype=np.int8)

    ltp_arr = (
        pd.to_numeric(df_levels["long_tp"], errors="coerce").fillna(tick_size * 15).values.astype(np.float64)
        if "long_tp" in df_levels.columns
        else np.full(n, tick_size * 15, dtype=np.float64)
    )
    lsl_arr = (
        pd.to_numeric(df_levels["long_sl"], errors="coerce").fillna(tick_size * 10).values.astype(np.float64)
        if "long_sl" in df_levels.columns
        else np.full(n, tick_size * 10, dtype=np.float64)
    )
    stp_arr = (
        pd.to_numeric(df_levels["short_tp"], errors="coerce").fillna(tick_size * 15).values.astype(np.float64)
        if "short_tp" in df_levels.columns
        else np.full(n, tick_size * 15, dtype=np.float64)
    )
    ssl_arr = (
        pd.to_numeric(df_levels["short_sl"], errors="coerce").fillna(tick_size * 10).values.astype(np.float64)
        if "short_sl" in df_levels.columns
        else np.full(n, tick_size * 10, dtype=np.float64)
    )

    for t in range(n - 1):
        entry = prices[t]
        end = min(t + 1 + int(max_bars_forward), n)
        future = prices[t + 1:end]
        if len(future) == 0:
            continue

        long_target = entry + float(ltp_arr[t])
        long_stop = entry - float(lsl_arr[t])
        short_target = entry - float(stp_arr[t])
        short_stop = entry + float(ssl_arr[t])

        lw_idx = np.where(future >= long_target)[0]
        ll_idx = np.where(future <= long_stop)[0]
        sw_idx = np.where(future <= short_target)[0]
        sl_idx = np.where(future >= short_stop)[0]

        lw = lw_idx[0] if len(lw_idx) > 0 else np.inf
        ll = ll_idx[0] if len(ll_idx) > 0 else np.inf
        sw = sw_idx[0] if len(sw_idx) > 0 else np.inf
        sl = sl_idx[0] if len(sl_idx) > 0 else np.inf

        if lw < ll:
            long_labels[t] = LABEL_WIN
        elif ll < lw:
            long_labels[t] = LABEL_LOSE

        if sw < sl:
            short_labels[t] = LABEL_WIN
        elif sl < sw:
            short_labels[t] = LABEL_LOSE

    result = df_trades.copy()
    result["long_label"] = long_labels
    result["short_label"] = short_labels

    bias = np.full(n, DIR_NEUTRAL, dtype=np.int8)
    long_only = (long_labels == LABEL_WIN) & (short_labels != LABEL_WIN)
    short_only = (short_labels == LABEL_WIN) & (long_labels != LABEL_WIN)
    bias[long_only] = DIR_LONG
    bias[short_only] = DIR_SHORT
    result["bias_label"] = bias

    quality_arr = np.full(n, QUALITY_NONE, dtype=np.int8)
    for t in range(n):
        if bias[t] != DIR_NEUTRAL:
            quality_arr[t] = _detect_quality(result, t, int(bias[t]))
    result["signal_quality"] = quality_arr

    if "regime" not in result.columns:
        result["regime"] = REGIME_RANGING
    result["regime"] = pd.to_numeric(result["regime"], errors="coerce").fillna(REGIME_RANGING).astype(np.int8)

    total = max(len(result), 1)
    lw = int((result["long_label"] == LABEL_WIN).sum())
    ll = int((result["long_label"] == LABEL_LOSE).sum())
    lc = int((result["long_label"] == LABEL_CANCEL).sum())
    sw = int((result["short_label"] == LABEL_WIN).sum())
    bc = result["bias_label"].value_counts()
    sq = result["signal_quality"].value_counts()

    print(
        f"  Labels (Long):  WIN={lw}({lw / total:.0%}) "
        f"LOSE={ll}({ll / total:.0%}) CANCEL={lc}({lc / total:.0%})"
    )
    print(f"  Labels (Short): WIN={sw}({sw / total:.0%})")
    print(
        f"  Bias: LONG={bc.get(DIR_LONG, 0)}({bc.get(DIR_LONG, 0) / total:.0%}) "
        f"SHORT={bc.get(DIR_SHORT, 0)}({bc.get(DIR_SHORT, 0) / total:.0%}) "
        f"NEUTRAL={bc.get(DIR_NEUTRAL, 0)}({bc.get(DIR_NEUTRAL, 0) / total:.0%})"
    )
    print(f"  Quality: STRONG={sq.get(QUALITY_STRONG, 0)} WEAK={sq.get(QUALITY_WEAK, 0)}")

    return result


def build_training_dataset(
    df: pd.DataFrame,
    window: int = 100,
    horizon: int = 50,
    tick_size: float = 0.0001,
    vol_mult: float = 1.5,
    obi_thr: float = 0.3,
    drop_neutral: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Unified dataset builder for sequence models.
    """

    df_aug = append_dynamic_orderbook_features(df, tick_size=tick_size)
    df_eng = engineer_features(df_aug, roll_window=20)

    base_cols = [c for c in MarketFeatureVector.feature_names() if c in df_eng.columns]
    derived = [
        c
        for c in df_eng.columns
        if c.endswith("_diff") or c.endswith("_zscore") or c.endswith("_rmean")
    ]
    feature_cols = base_cols + derived
    event_mask = build_event_filter(df_eng, vol_mult=vol_mult, obi_thr=obi_thr)

    X_list, y_list = [], []
    price_col = "close" if "close" in df_eng.columns else ("price" if "price" in df_eng.columns else "micro_price")

    for i in range(window, len(df_eng) - horizon):
        if not bool(event_mask.iloc[i]):
            continue

        win = df_eng.iloc[i - window:i][feature_cols].values
        if win.shape[0] < window:
            continue

        entry = float(pd.to_numeric(df_eng[price_col].iloc[i], errors="coerce") or 0.0)
        tp_long = float(df_eng["long_tp"].iloc[i]) if "long_tp" in df_eng.columns else tick_size * 15
        sl_long = float(df_eng["long_sl"].iloc[i]) if "long_sl" in df_eng.columns else tick_size * 10
        tp_short = float(df_eng["short_tp"].iloc[i]) if "short_tp" in df_eng.columns else tick_size * 15
        sl_short = float(df_eng["short_sl"].iloc[i]) if "short_sl" in df_eng.columns else tick_size * 10
        future_px = pd.to_numeric(df_eng[price_col].iloc[i + 1:i + 1 + horizon], errors="coerce").ffill().fillna(0.0).values.astype(np.float64)

        lw = np.where(future_px >= entry + tp_long)[0]
        ll = np.where(future_px <= entry - sl_long)[0]
        sw = np.where(future_px <= entry - tp_short)[0]
        sl = np.where(future_px >= entry + sl_short)[0]

        l_win = lw[0] if len(lw) > 0 else np.inf
        l_lose = ll[0] if len(ll) > 0 else np.inf
        s_win = sw[0] if len(sw) > 0 else np.inf
        s_lose = sl[0] if len(sl) > 0 else np.inf

        if l_win < l_lose:
            direction = DIR_LONG
        elif s_win < s_lose:
            direction = DIR_SHORT
        else:
            if drop_neutral:
                continue
            direction = DIR_NEUTRAL

        quality = _detect_quality(df_eng, i, direction)
        regime = int(df_eng["regime"].iloc[i]) if "regime" in df_eng.columns else REGIME_RANGING

        X_list.append(win.astype(np.float32))
        y_list.append([direction, quality, regime])

    if not X_list:
        return np.empty((0, window, len(feature_cols))), np.empty((0, 3)), feature_cols

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int8)

    print("\n✅ Dataset جاهز:")
    print(f"   X.shape = {X.shape}   y.shape = {y.shape}")
    print(
        f"   LONG={int((y[:, 0] == DIR_LONG).sum())}  "
        f"SHORT={int((y[:, 0] == DIR_SHORT).sum())}  "
        f"NEUTRAL={int((y[:, 0] == DIR_NEUTRAL).sum())}"
    )
    print(
        f"   STRONG={int((y[:, 1] == QUALITY_STRONG).sum())}  "
        f"WEAK={int((y[:, 1] == QUALITY_WEAK).sum())}"
    )
    print(f"   Features ({len(feature_cols)}): {feature_cols[:6]} ...")

    return X, y, feature_cols
