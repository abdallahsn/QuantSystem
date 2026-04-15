"""
dynamic_labels.py — V2: Unified Pipeline (OrderBook → Features → Labels)
========================================================================
This module upgrades the old labeling path into a unified pipeline:

  1. OrderWallScanner returns a dense feature vector
  2. Labels are multi-layered: direction + quality + regime
  3. Event-based filtering suppresses quiet/noisy rows before window building
  4. Feature engineering happens on the full frame before slicing windows
  5. Direction/quality are derived from realized price movement, not TP/SL
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

    def _detect_gap(
        self,
        px_levels: list[float],
        sz_levels: list[float],
        tick: float,
        mean_sz: float,
    ) -> tuple[float | None, float]:
        """
        Detect both hard price gaps and softer liquidity voids.

        Sample books often have perfect 1-tick spacing, so a pure price-gap rule
        can leave `gap_size` dead. We therefore augment the signal with a
        size-scarcity score between adjacent levels.
        """

        gap_px = None
        gap_size = 0.0
        if len(px_levels) < 2:
            return gap_px, gap_size

        local_thr = max(float(self.gap_mult), 1.60)
        for i in range(len(px_levels) - 1):
            spacing_ticks = abs(float(px_levels[i]) - float(px_levels[i + 1])) / tick
            cur_sz = float(sz_levels[i])
            nxt_sz = float(sz_levels[i + 1])
            min_pair = min(cur_sz, nxt_sz)
            avg_pair = (cur_sz + nxt_sz) / 2.0

            scarcity = max(0.0, 1.0 - (min_pair / max(mean_sz, 1e-9)))
            void_ratio = max(0.0, 1.0 - (avg_pair / max(mean_sz, 1e-9)))
            effective_gap = spacing_ticks + 0.80 * scarcity + 0.60 * void_ratio

            is_hard_gap = spacing_ticks >= float(self.gap_mult)
            is_soft_gap = (spacing_ticks >= 1.0) and (scarcity >= 0.65) and (effective_gap >= local_thr)
            if is_hard_gap or is_soft_gap:
                gap_px = float(px_levels[i + 1])
                gap_size = max(gap_size, effective_gap)
                break

        return gap_px, gap_size

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

        bid_gap_px, bid_gap_size = self._detect_gap(bid_px, bid_sz, tick=tick, mean_sz=mean_sz)
        ask_gap_px, ask_gap_size = self._detect_gap(ask_px, ask_sz, tick=tick, mean_sz=mean_sz)

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


def append_dynamic_orderbook_features(
    df: pd.DataFrame,
    tick_size: float = 0.0001,
    price_col: str = "price",
    cvd_col: str = "cvd",
    volume_col: str = "size",
    wall_mult: float = 3.0,
    gap_mult: float = 2.0,
    levels: int = 10,
) -> pd.DataFrame:
    """
    Append order-book scanner features to a frame.
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
            }
        )

    if not records:
        return out

    feat_df = pd.DataFrame(records, index=out.index)
    for col in feat_df.columns:
        out[col] = feat_df[col].values
    return out


def _direction_from_future_return(
    future_return: float,
    tick_size: float = 0.0001,
    threshold_ticks: float = 5.0,
) -> int:
    threshold = max(float(tick_size or 0.0), 1e-9) * float(threshold_ticks)
    if future_return > threshold:
        return DIR_LONG
    if future_return < -threshold:
        return DIR_SHORT
    return DIR_NEUTRAL


def _build_quality_thresholds(
    move_strengths: np.ndarray,
    directional_mask: np.ndarray,
    tick_size: float = 0.0001,
    strong_quantile: float = 0.70,
    fallback_ticks: float = 10.0,
    min_history: int = 32,
    lookback: int = 256,
) -> np.ndarray:
    """
    Build causal, adaptive thresholds for STRONG vs WEAK moves.

    The threshold is based on the rolling quantile of previously realized
    directional moves so the quality layer adapts to volatility without using
    future rows from the same sample.
    """

    base_thr = max(float(tick_size or 0.0), 1e-9) * float(fallback_ticks)
    floor_thr = max(base_thr * 0.80, 1e-9)
    strengths = pd.Series(np.asarray(move_strengths, dtype=np.float64))
    directional = pd.Series(np.asarray(directional_mask, dtype=bool), index=strengths.index)
    hist = strengths.where(directional).shift(1)

    rolling_q = hist.rolling(window=max(int(lookback), int(min_history)), min_periods=max(8, int(min_history))).quantile(strong_quantile)
    thresholds = rolling_q.fillna(base_thr).clip(lower=floor_thr)
    return thresholds.to_numpy(dtype=np.float64, copy=False)


def label_with_forward_scan(
    df_trades: pd.DataFrame,
    df_levels: pd.DataFrame,
    max_bars_forward: int = 50,
    tick_size: float = 0.0001,
) -> pd.DataFrame:
    """
    Price-only forward scan with three-layer label outputs.
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
    bias = np.full(n, DIR_NEUTRAL, dtype=np.int8)
    quality_arr = np.full(n, QUALITY_NONE, dtype=np.int8)
    future_returns = np.zeros(n, dtype=np.float64)
    move_strengths = np.zeros(n, dtype=np.float64)

    for t in range(n - 1):
        entry = prices[t]
        end = min(t + 1 + int(max_bars_forward), n)
        future = prices[t + 1:end]
        if len(future) == 0:
            continue

        future_return = float(future[-1] - entry)
        direction = _direction_from_future_return(future_return, tick_size=tick_size, threshold_ticks=5.0)
        move_strength = abs(future_return)
        future_returns[t] = future_return
        move_strengths[t] = move_strength

        bias[t] = direction
        if direction == DIR_LONG:
            long_labels[t] = LABEL_WIN
            short_labels[t] = LABEL_LOSE
        elif direction == DIR_SHORT:
            long_labels[t] = LABEL_LOSE
            short_labels[t] = LABEL_WIN

    directional_mask = bias != DIR_NEUTRAL
    quality_thresholds = _build_quality_thresholds(
        move_strengths,
        directional_mask,
        tick_size=tick_size,
        strong_quantile=0.70,
        fallback_ticks=10.0,
        min_history=max(24, int(max_bars_forward)),
        lookback=max(96, int(max_bars_forward) * 6),
    )
    strong_mask = directional_mask & (move_strengths >= quality_thresholds)
    weak_mask = directional_mask & ~strong_mask
    quality_arr[strong_mask] = QUALITY_STRONG
    quality_arr[weak_mask] = QUALITY_WEAK

    result = df_trades.copy()
    result["long_label"] = long_labels
    result["short_label"] = short_labels
    result["bias_label"] = bias
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
        f"  Labels (Long view):  WIN={lw}({lw / total:.0%}) "
        f"LOSE={ll}({ll / total:.0%}) CANCEL={lc}({lc / total:.0%})"
    )
    print(f"  Labels (Short view): WIN={sw}({sw / total:.0%})")
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
    labeled = label_with_forward_scan(df_eng, df_eng, max_bars_forward=horizon, tick_size=tick_size)

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

        direction = int(labeled["bias_label"].iloc[i])
        if direction == DIR_NEUTRAL and drop_neutral:
            continue

        quality = int(labeled["signal_quality"].iloc[i]) if direction != DIR_NEUTRAL else QUALITY_NONE
        regime = int(labeled["regime"].iloc[i]) if "regime" in labeled.columns else REGIME_RANGING

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
