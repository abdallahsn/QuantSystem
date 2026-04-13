"""
labels_v19.py - Causal event labels for QuantSystem V19
=======================================================
V19 stops using session-level labels that depend on the future path of an
entire day/session. Instead, each row receives an event-time label defined by:

  1. Current price at time t
  2. Current volatility proxy (micro_atr fallback to local price noise)
  3. Fixed forward horizon in ticks/events
  4. Upper/lower barriers derived from the current volatility only

This keeps the labeling problem causal from the perspective of feature
construction while still allowing supervised learning against future outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BIAS_LONG = 0
BIAS_SHORT = 1
BIAS_NEUTRAL = 2

SETUP_ABSORPTION = 0
SETUP_SPOOFING = 1
SETUP_OBI = 2
SETUP_MIXED = 3


def _directional_efficiency(prices: np.ndarray, window: int = 12) -> np.ndarray:
    s = pd.Series(prices)
    gross = s.diff().abs().rolling(window, min_periods=2).sum()
    net = s.diff(window).abs()
    eff = (net / gross.clip(lower=1e-8)).fillna(0.0).clip(0.0, 1.0)
    return eff.values.astype(np.float32)


def _infer_setup_labels(df: pd.DataFrame) -> np.ndarray:
    absorb = df.get('absorption_intensity', pd.Series(np.zeros(len(df)))).fillna(0).abs().values
    spoof = df.get('spoofing_ratio', pd.Series(np.zeros(len(df)))).fillna(0).abs().values
    obi = df.get('obi', pd.Series(np.zeros(len(df)))).fillna(0).abs().values

    absorb_sig = absorb > 0.30
    spoof_sig = spoof > 0.20
    obi_sig = obi > 0.30
    sig_count = absorb_sig.astype(np.int8) + spoof_sig.astype(np.int8) + obi_sig.astype(np.int8)

    setup = np.full(len(df), SETUP_MIXED, dtype=np.int8)
    setup[(sig_count == 1) & absorb_sig] = SETUP_ABSORPTION
    setup[(sig_count == 1) & spoof_sig] = SETUP_SPOOFING
    setup[(sig_count == 1) & obi_sig] = SETUP_OBI
    return setup


def _liquidity_score(df: pd.DataFrame) -> np.ndarray:
    volume_burst = df.get('volume_burst', pd.Series(np.zeros(len(df)))).fillna(0).clip(lower=0).values
    absorption = df.get('absorption_intensity', pd.Series(np.zeros(len(df)))).fillna(0).abs().values
    obi = df.get('obi', pd.Series(np.zeros(len(df)))).fillna(0).abs().values

    score = np.log1p(volume_burst) * (1.0 + absorption) * (1.0 + obi)
    return np.clip(score, 0.0, 100.0).astype(np.float32)


def build_causal_event_labels(
    df: pd.DataFrame,
    horizon: int = 50,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    neutral_mult: float = 0.35,
    tick_size: float = 1e-4,
) -> pd.DataFrame:
    """
    Build causal event labels with forward barriers over a fixed event horizon.

    The label for row i uses only the row-i state to set barriers, then inspects
    the future path over `horizon` events to determine which barrier hits first.
    """
    out = df.copy()
    n = len(out)

    if n == 0:
        out['bias_label'] = np.array([], dtype=np.int8)
        out['setup_label'] = np.array([], dtype=np.int8)
        out['conf_label'] = np.array([], dtype=np.float32)
        out['is_expansion'] = np.array([], dtype=np.int8)
        out['liq_score'] = np.array([], dtype=np.float32)
        out['label_end_ts'] = pd.to_datetime([])
        out['forward_return'] = np.array([], dtype=np.float32)
        out['label_horizon_steps'] = np.array([], dtype=np.int32)
        return out

    ts = pd.to_datetime(
        out.get('ts_event', pd.Series(pd.RangeIndex(n))),
        utc=True,
        errors='coerce',
    ).dt.tz_localize(None)

    prices = pd.to_numeric(out.get('price', pd.Series(np.zeros(n))), errors='coerce').ffill().fillna(0).values.astype(np.float64)
    vol_raw = pd.to_numeric(out.get('micro_atr', pd.Series(np.zeros(n))), errors='coerce').fillna(0).abs().values.astype(np.float64)

    fallback_vol = float(np.nanmedian(vol_raw[vol_raw > 0])) if np.any(vol_raw > 0) else 0.0
    if fallback_vol <= 0:
        price_diff = np.abs(np.diff(prices))
        fallback_vol = float(np.nanmedian(price_diff[price_diff > 0])) if np.any(price_diff > 0) else tick_size
    fallback_vol = max(fallback_vol, tick_size)
    vols = np.where(vol_raw > tick_size, vol_raw, fallback_vol)
    liq_score = _liquidity_score(out).astype(np.float64)
    efficiency = _directional_efficiency(
        prices,
        window=max(6, min(24, max(2, horizon // 2))),
    )
    liq_positive = liq_score[liq_score > 0]
    liq_anchor = float(np.nanmedian(liq_positive)) if liq_positive.size else 1.0
    liq_low_cutoff = float(np.nanquantile(liq_positive, 0.30)) if liq_positive.size >= 5 else liq_anchor * 0.75
    liq_low_cutoff = max(liq_low_cutoff, 1e-6)

    bias = np.full(n, BIAS_NEUTRAL, dtype=np.int8)
    conf = np.zeros(n, dtype=np.float32)
    is_expansion = np.zeros(n, dtype=np.int8)
    forward_return = np.zeros(n, dtype=np.float32)
    horizon_steps = np.zeros(n, dtype=np.int32)
    label_end_ts = np.empty(n, dtype='datetime64[ns]')

    for i in range(n):
        end_idx = min(n - 1, i + horizon)
        label_end_ts[i] = np.datetime64(ts.iloc[end_idx], 'ns')
        horizon_steps[i] = end_idx - i

        if i >= n - 1 or end_idx <= i:
            continue

        p0 = prices[i]
        vol = max(vols[i], tick_size)
        upper = p0 + tp_mult * vol
        lower = p0 - sl_mult * vol
        future_path = prices[i + 1:end_idx + 1]
        future_deltas = np.diff(np.concatenate([[p0], future_path]))
        path_noise = float(np.abs(future_deltas).sum())
        max_up = float(future_path.max() - p0)
        max_down = float(p0 - future_path.min())
        path_range = max_up + max_down

        hit_up = np.where(future_path >= upper)[0]
        hit_down = np.where(future_path <= lower)[0]

        final_ret = future_path[-1] - p0
        forward_return[i] = float(final_ret)
        directionality = float(abs(final_ret) / max(path_noise, tick_size))
        local_eff = float(efficiency[i])
        local_liq = float(liq_score[i])
        low_liq = local_liq <= liq_low_cutoff
        range_like = (path_range < 1.75 * vol) or (max(local_eff, directionality) < 0.35)

        neutral_band = neutral_mult * vol
        if low_liq:
            neutral_band = max(neutral_band, 0.60 * vol)
        if range_like:
            neutral_band = max(neutral_band, 0.75 * vol)

        strict_terminal = low_liq or range_like
        dir_gate = 0.50 if strict_terminal else 0.30
        liq_ratio = min(local_liq / max(liq_anchor, 1e-6), 1.5)

        def _confidence(move_strength: float, breakout: bool) -> float:
            move_score = min(move_strength / max(vol, tick_size), 2.5) / 2.5
            breakout_bonus = 0.15 if breakout else 0.0
            raw = (
                0.35 * move_score +
                0.35 * max(local_eff, directionality) +
                0.20 * min(liq_ratio, 1.0) +
                breakout_bonus
            )
            return float(np.clip(raw, 0.0, 1.0))

        if hit_up.size and (not hit_down.size or hit_up[0] <= hit_down[0]):
            breakout_ret = float(future_path[hit_up[0]] - p0)
            if strict_terminal and breakout_ret < 1.40 * vol and max(local_eff, directionality) < 0.55:
                bias[i] = BIAS_NEUTRAL
            else:
                bias[i] = BIAS_LONG
                conf[i] = _confidence(max(final_ret, breakout_ret), breakout=True)
                is_expansion[i] = int((not low_liq) or breakout_ret >= 2.0 * vol)
        elif hit_down.size:
            breakout_ret = float(p0 - future_path[hit_down[0]])
            if strict_terminal and breakout_ret < 1.40 * vol and max(local_eff, directionality) < 0.55:
                bias[i] = BIAS_NEUTRAL
            else:
                bias[i] = BIAS_SHORT
                conf[i] = _confidence(max(-final_ret, breakout_ret), breakout=True)
                is_expansion[i] = int((not low_liq) or breakout_ret >= 2.0 * vol)
        elif final_ret > neutral_band and max(local_eff, directionality) >= dir_gate:
            bias[i] = BIAS_LONG
            conf[i] = _confidence(abs(final_ret), breakout=False)
            is_expansion[i] = int(abs(final_ret) >= max(vol, neutral_band) and max(local_eff, directionality) >= 0.45)
        elif final_ret < -neutral_band and max(local_eff, directionality) >= dir_gate:
            bias[i] = BIAS_SHORT
            conf[i] = _confidence(abs(final_ret), breakout=False)
            is_expansion[i] = int(abs(final_ret) >= max(vol, neutral_band) and max(local_eff, directionality) >= 0.45)
        else:
            bias[i] = BIAS_NEUTRAL
            conf[i] = 0.0
            is_expansion[i] = 0

    out['bias_label'] = bias
    out['setup_label'] = _infer_setup_labels(out)
    out['conf_label'] = conf.astype(np.float32)
    out['is_expansion'] = is_expansion.astype(np.int8)
    out['liq_score'] = liq_score.astype(np.float32)
    out['label_end_ts'] = pd.to_datetime(label_end_ts)
    out['forward_return'] = forward_return.astype(np.float32)
    out['label_horizon_steps'] = horizon_steps.astype(np.int32)
    return out
