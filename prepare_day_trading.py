"""
prepare_day_trading.py — Day Trading Refinery for QuantSystem V19
═══════════════════════════════════════════════════════════════════

## تصميم هجين (تكات + شموع) — مسار واحد

**المرحلة 1 — تجميع غني (هذا الملف = «جسر السيولة»، بدون mbo_aggregator منفصل):**
  - من MBO/MBP الخام → شمعة OHLCV + ميزات ميكروستراكتشر مُجمَّعة (CVD، امتصاص، تدفق، …).
  - بناء LOB tensor لكل شمعة (50 × 20 × 3): البعد الأول = آخر 50 شمعة تاريخية؛ مع MBP لقطة ذروة imbalance لكل شمعة.
  - محاذاة سطح CatBoost الـ31 مع prepare_training_data (أعمدة + raw__*).

**المرحلة 2 — تدريب موحّد (train_v19.py):**
  - CatBoost/XGB على الميزات الجدولية (+ soft_label أو bias).
  - DeepLOB CNN على tensors المُجمَّعة.
  - MetaLearner (LSTM/TCN) على تسلسلات الشموع بمدخلات: إحصاء + احتمالات الأشجار + embeddings.

**المرحلة 3 — حي (مستقبلاً):**
  - كل إغلاق شمعة: تجميع تيكات → نفس الميزات → نفس النماذج المحفوظة.

المخرج: نفس شكل المصفاة الكاملة بقدر الإمكان
        → train_v19.py يشتغل بدون تعديل على مسار الحدث.
"""

from __future__ import annotations

import argparse
import os
import sys
import glob
import json
import numpy as np
import pandas as pd

from modules.context_features import compute_daily_weekly_levels
from modules.tick_intrabar_slices import enrich_bars_with_intrabar
from modules.intrabar_mbp_microstructure import enrich_bars_with_intrabar_mbp

# ─── Regime Configuration (Section 9) ────────────────────────────────────────
try:
    from regime_config import (
        REGIME_TP_SL,
        REGIME_MAX_BARS,
        REGIME_EVENT_THRESHOLD,
        EVENT_ZSCORE_WINDOW,
        EVENT_ZSCORE_MIN_PERIODS,
        EVENT_SCORE_WEIGHTS,
    )
except ImportError:
    # Fallback إذا لم يُنشأ regime_config.py بعد
    REGIME_TP_SL = {'trending': (2.0, 1.0), 'ranging': (0.8, 0.6), 'volatile': (1.5, 1.5)}
    REGIME_MAX_BARS = {'trending': 12, 'ranging': 6, 'volatile': 3}
    REGIME_EVENT_THRESHOLD = {'trending': 0.60, 'ranging': 0.60, 'volatile': 0.75}
    EVENT_ZSCORE_WINDOW = 100
    EVENT_ZSCORE_MIN_PERIODS = 20
    EVENT_SCORE_WEIGHTS = {
        'hawkes_z_above_1': 0.30, 'absorb_z_above_1': 0.30,
        'kyle_z_above_05': 0.20, 'cvd_align_above_06': 0.20,
    }

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# ─── Session Windows (UTC) ────────────────────────────────────────────────────
SESSIONS = {
    'asia':    ('00:00', '07:00'),
    'london':  ('07:00', '12:00'),
    'overlap': ('12:00', '16:00'),
    'ny':      ('13:30', '20:00'),
}

# ─── Features للـ Day Trading (تُضاف لـ CATBOOST_ADVISOR_FEATURES) ────────────
DAY_TRADING_FEATURES = [
    # Technical
    'atr_14',           # Average True Range — حجم الحركة المتوقعة
    'rsi_14',           # RSI — زخم السعر
    'macd_hist',        # MACD Histogram — قوة الاتجاه
    'bar_range',        # High - Low — نطاق الشمعة
    'body_ratio',       # Body / Range — قوة الشمعة
    # Session
    'is_london',        # جلسة لندن (أعلى سيولة)
    'is_overlap',       # overlap لندن/NY (أعلى volatility)
    'is_ny',            # جلسة نيويورك
    # Multi-Timeframe Context
    'cvd_slope_6b',     # ميل CVD على ~30 دقيقة (Δ يعتمد على freq)
    'volume_ratio_6b',  # نسبة volume الحالي للمتوسط
    'vwap_dist_6b',     # مسافة السعر عن VWAP
    'return_6b',        # عائد آخر 6 bars
    # ~1 ساعة و ~4 ساعات وفق عدد الشموع المشتق من bar_freq (مثلاً 5→12،48)
    'cvd_slope_1h', 'volume_ratio_1h', 'vwap_dist_1h', 'return_1h',
    'cvd_slope_4h', 'volume_ratio_4h', 'vwap_dist_4h', 'return_4h',
    # Order Flow (مُجمَّع على bar)
    'buy_ratio',        # نسبة الشراء في الـ bar
    'bar_cvd_delta',    # تغير CVD داخل الـ bar
    'lob_imbalance',    # اختلال عمق السوق المُجمَّع
    # جسر السيولة (مُجمَّع من التكات داخل الشمعة — أسماء صريحة للمسار الهجين)
    'num_trades',       # = tick_count
    'avg_trade_size',   # volume / num_trades
    'absorption_bar',   # ضغط شراء داخل الشمعة (≈ buy_ratio)
    'order_flow_imbalance',  # (buy_vol - sell_vol) / total ∈ [-1,1]
    'spread_bar',       # متوسط سبريد التيكات إن وُجد عمود spread

    # Intrabar (5m → 12× slices) — spike-preserving microstructure
    'spoof_peak_slice',
    'spoof_burst_flag',
    'cancel_burst',
    'absorption_speed',
    'pressure_phase',
    'cvd_velocity_max',
    'cvd_direction_pct',
    'cvd_early_vs_late',
    'ofi_peak_slice',
    'size_dispersion',
    'absorption_bar_slice_max',
    # Bar-level dispersion / velocity / MBP telemetry
    'absorption_std',
    'cancel_std',
    'kyle_std',
    'micro_atr_max',
    'cvd_velocity',
    'imb_reversals',
    'cancel_volume_ratio',
    'mbp_bar_coverage',
    'mbp_roll_lob_coverage',
    'mbp_bid_slope_intrabar',
    'mbp_ask_slope_intrabar',
    'mbp_depth_accel',
]

# ─── يُطابق prepare_training_data.CATBOOST_ADVISOR_FEATURES حرفًا (N=31) ──────
CATBOOST_ADVISOR_FEATURES_DT = [
    'cvd', 'obi', 'absorption_intensity', 'cancel_ratio',
    'spoofing_ratio', 'spoofing_duration', 'liquidity_trap',
    'micro_atr', 'volume_burst', 'inter_event_time',
    'micro_price', 'bid_wall_strength', 'ask_wall_strength',
    'distance_to_wall', 'gap_size', 'liquidity_density',
    'fisher_signal', 'anomaly',
    'cvd_momentum', 'cvd_price_divergence',
    'trend_strength', 'correction_depth', 'liquidity_sweep',
    'pdh', 'pdl', 'dist_to_pdh', 'price_position',
    'kyle_lambda', 'hawkes_intensity', 'vnet',
    'vwap_z_score',
]


def _bar_period_seconds(freq: str) -> float:
    td = pd.to_timedelta(freq)
    return max(float(td.total_seconds()), 1e-6)


def _bars_for_target_minutes(freq: str, target_minutes: int) -> int:
    """كم شمعة تغطي تقريباً target_minutes دقيقة لهذا الشمع الأساس."""
    raw = str(freq).lower().replace('min', '').strip()
    try:
        m = float(raw)
        m = max(m, 1e-6)
    except ValueError:
        m = 5.0
    return max(1, int(round(float(target_minutes) / m)))


def _num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=np.float64)
    s = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return s.astype(np.float64)


def _combine_first_valid(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    p = primary.astype(np.float64)
    f = fallback.reindex(primary.index).astype(np.float64)
    pv = p.to_numpy()
    fv = f.to_numpy()
    ok = np.isfinite(pv)
    out = np.where(ok, pv, fv)
    return pd.Series(out, index=p.index, dtype=np.float64)


def _tick_resample_optional(
    df: pd.DataFrame, freq: str, col: str, how: str = "mean",
) -> pd.Series | None:
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    if how == "mean":
        out = s.resample(freq).mean()
    elif how == "max":
        out = s.resample(freq).max()
    elif how == "sum":
        out = s.resample(freq).sum()
    elif how == "last":
        out = s.resample(freq).last()
    else:
        out = s.resample(freq).mean()
    return out


def apply_bar_level_catboost_parities(df_bars: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    يضمن ظهور أعمدة الـ advisor الـ31 لـ train_v19؛ يكمِّل ما نجمّعه من التيكات بتقريبات bar-level
    عند النقص فقط (لا يستبدل إشارة المصفاة الصالحة).
    """
    df = df_bars.copy()
    bar_sec = float(_bar_period_seconds(freq))

    close = _num_series(df, 'close')
    high = _num_series(df, 'high')
    low = _num_series(df, 'low')
    open_ = _num_series(df, 'open')
    vol = _num_series(df, 'volume')
    br = _num_series(df, 'bar_range')
    br = _combine_first_valid(br, (high - low).abs())

    bv = _num_series(df, 'buy_volume')
    if 'buy_vol' in df.columns:
        bv = _combine_first_valid(bv, _num_series(df, 'buy_vol'))
    sv = _num_series(df, 'sell_volume')
    tot_flow = (bv + sv).replace(0, np.nan)
    ofi = ((bv - sv) / tot_flow).clip(-1.0, 1.0).fillna(0.0)

    buy_ratio = _num_series(df, 'buy_ratio').clip(0.0, 1.0)
    tick_n = _num_series(df, 'tick_count').clip(lower=1.0)
    if 'num_trades' in df.columns:
        tick_n = _combine_first_valid(tick_n, _num_series(df, 'num_trades').clip(lower=1.0))

    atr = _num_series(df, 'atr_14').clip(lower=1e-12)
    micro_atr = _num_series(df, 'micro_atr').clip(lower=1e-12)

    cr = _num_series(df, 'cancel_ratio').clip(0.0, 1.0)
    abs_i = _num_series(df, 'absorption_intensity').clip(0.0, 1.0)

    # مستويات اليوم (نفس prepare_training_data / context_features)
    if 'ts_event' in df.columns:
        levels = df[['ts_event', 'close']].copy()
        levels = levels.rename(columns={'close': 'price'})
        try:
            ctx = compute_daily_weekly_levels(levels, price_col='price', ts_col='ts_event')
            df['pdh'] = ctx['pdh'].to_numpy()
            df['pdl'] = ctx['pdl'].to_numpy()
            df['dist_to_pdh'] = ctx['dist_to_pdh'].to_numpy()
            df['price_position'] = ctx['price_position'].to_numpy()
        except Exception:
            df['pdh'] = close.to_numpy()
            df['pdl'] = close.to_numpy()
            df['dist_to_pdh'] = np.zeros(len(df), dtype=np.float64)
            df['price_position'] = np.full(len(df), 0.5, dtype=np.float64)
    else:
        df['pdh'] = close.to_numpy()
        df['pdl'] = close.to_numpy()
        df['dist_to_pdh'] = np.zeros(len(df), dtype=np.float64)
        df['price_position'] = np.full(len(df), 0.5, dtype=np.float64)

    obi_proxy = ofi
    if 'order_flow_imbalance' in df.columns:
        obi_proxy = _combine_first_valid(
            obi_proxy,
            _num_series(df, 'order_flow_imbalance').clip(-1.0, 1.0),
        )
    if 'lob_imbalance' in df.columns:
        obi_proxy = _combine_first_valid(obi_proxy, _num_series(df, 'lob_imbalance').clip(-1.0, 1.0))

    spoof_proxy = (cr * 0.85).clip(0.0, 1.0)
    dur_proxy = pd.Series(bar_sec * np.clip(cr.to_numpy(), 0.0, 1.0), index=df.index, dtype=np.float64)
    trap_proxy = (abs_i * cr * 2.0).clip(0.0, 1.0)

    iet_arr = bar_sec / np.maximum(tick_n.to_numpy(), 1.0)
    iet_proxy = pd.Series(np.clip(iet_arr, 0.0, 1e6), index=df.index, dtype=np.float64)

    mid = (high + low + close) / 3.0
    mp = _num_series(df, 'micro_price')
    mp_f = _combine_first_valid(mp, mid)
    mp_f = _combine_first_valid(mp_f, close)

    wall_bid = (buy_ratio * 2.0).clip(0.0, 2.0)
    wall_ask = ((1.0 - buy_ratio) * 2.0).clip(0.0, 2.0)
    dist_wall = (close - mp_f).abs() / micro_atr.replace(0, np.nan)
    dist_wall = dist_wall.replace([np.inf, -np.inf], np.nan)

    prev_close = close.shift(1)
    gap = (open_ - prev_close).abs() / atr.replace(0, np.nan)
    gap = gap.replace([np.inf, -np.inf], np.nan)

    denom = br.replace(0, np.nan) + micro_atr * 1e-3
    liq_dens = vol / denom
    liq_dens = liq_dens.replace([np.inf, -np.inf], np.nan)

    def _assign_advisor(name: str, primary: pd.Series, fallback: pd.Series) -> None:
        s = _combine_first_valid(primary, fallback)
        df[name] = s

    _assign_advisor('obi', _num_series(df, 'obi'), obi_proxy)
    _assign_advisor('spoofing_ratio', _num_series(df, 'spoofing_ratio'), spoof_proxy)
    _assign_advisor('spoofing_duration', _num_series(df, 'spoofing_duration'), dur_proxy)
    _assign_advisor('liquidity_trap', _num_series(df, 'liquidity_trap'), trap_proxy)
    _assign_advisor('inter_event_time', _num_series(df, 'inter_event_time'), iet_proxy)
    _assign_advisor('micro_price', _num_series(df, 'micro_price'), mp_f)
    _assign_advisor('bid_wall_strength', _num_series(df, 'bid_wall_strength'), wall_bid)
    _assign_advisor('ask_wall_strength', _num_series(df, 'ask_wall_strength'), wall_ask)
    _assign_advisor('distance_to_wall', _num_series(df, 'distance_to_wall'), dist_wall)
    _assign_advisor('gap_size', _num_series(df, 'gap_size'), gap)
    _assign_advisor('liquidity_density', _num_series(df, 'liquidity_density'), liq_dens)

    df['obi'] = df['obi'].clip(-1.0, 1.0)
    df['spoofing_ratio'] = df['spoofing_ratio'].clip(0.0, 1.0)
    df['price_position'] = pd.to_numeric(df['price_position'], errors='coerce').clip(0.0, 1.0).fillna(0.5)

    # التطهير النهائي: الحفاظ على مقياس السعر لـ pdh/pdl/micro_price
    for col in CATBOOST_ADVISOR_FEATURES_DT:
        if col not in df.columns:
            df[col] = np.nan
        raw = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        if col in ('pdh', 'pdl', 'micro_price'):
            df[col] = _combine_first_valid(raw, close).astype(np.float64)
        elif col in ('dist_to_pdh',):
            df[col] = raw.fillna(0.0).astype(np.float64)
        elif col == 'price_position':
            df[col] = raw.clip(0.0, 1.0).fillna(0.5).astype(np.float64)
        else:
            df[col] = raw.fillna(0.0).astype(np.float64)

    return df


def attach_hybrid_liquidity_bridge(
    bars: pd.DataFrame,
    df_ticks: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    """
    يضغط معلومات «صانع السوق» من التكات داخل كل شمعة إلى أعمدة صريحة
    (مكملة لـ OHLCV والميزات المُجمَّعة من المصفاة).

    ملاحظة: في MBO الجانب الشرائي = 'A' والبيعي = 'B' (ليس buy/sell نصاً).
    """
    out = bars.copy()
    tc = (
        pd.to_numeric(out['tick_count'], errors='coerce').fillna(0.0).astype(np.float64).clip(lower=1.0)
    )
    vol = pd.to_numeric(out['volume'], errors='coerce').fillna(0.0).astype(np.float64)
    out['num_trades'] = tc
    out['avg_trade_size'] = (vol / tc).replace([np.inf, -np.inf], 0).fillna(0.0)
    out['absorption_bar'] = (
        pd.to_numeric(out['buy_ratio'], errors='coerce').fillna(0.5).astype(np.float64).clip(0.0, 1.0)
    )
    bv = pd.to_numeric(out['buy_volume'], errors='coerce').fillna(0.0).astype(np.float64)
    sv = pd.to_numeric(out['sell_volume'], errors='coerce').fillna(0.0).astype(np.float64)
    sums = bv.to_numpy(dtype=np.float64) + sv.to_numpy(dtype=np.float64)
    tot = pd.Series(np.maximum(sums, 1e-9), index=out.index)
    out['order_flow_imbalance'] = np.clip((bv - sv) / tot, -1.0, 1.0)
    if 'spread' in df_ticks.columns:
        sp = pd.to_numeric(df_ticks['spread'], errors='coerce').resample(freq).mean()
        out['spread_bar'] = sp.reindex(out.index)
        out['spread_bar'] = pd.to_numeric(out['spread_bar'], errors='coerce').fillna(0.0)
    else:
        out['spread_bar'] = np.zeros(len(out), dtype=np.float32)
    return out

def aggregate_mbo_to_bars(df_mbo: pd.DataFrame, freq: str = '5min') -> pd.DataFrame:
    """
    يجمع التيكات في bars مع الحفاظ على order flow features.
    """
    df = df_mbo.copy()
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    df = df.set_index('ts_event').sort_index()

    # OHLCV
    bars = df['price'].resample(freq).agg(
        open='first', high='max', low='min', close='last'
    )
    bars['volume'] = df['size'].resample(freq).sum()

    # CVD
    bars['cvd']           = df['cvd'].resample(freq).last()
    bars['bar_cvd_delta'] = df['cvd'].resample(freq).agg(
        lambda x: float(x.iloc[-1] - x.iloc[0]) if len(x) > 1 else 0.0
    )
    bars['session_cvd'] = df['session_cvd'].resample(freq).last()

    # Order Flow (peak + dispersion داخل الشمعة)
    bars['kyle_lambda']        = df['kyle_lambda'].resample(freq).max()
    bars['hawkes_intensity']   = df['hawkes_intensity'].resample(freq).max()
    bars['absorption_intensity'] = df['absorption_intensity'].resample(freq).max()
    bars['cancel_ratio']       = df['cancel_ratio'].resample(freq).max()
    bars['absorption_std']     = df['absorption_intensity'].resample(freq).std().fillna(0.0)
    bars['cancel_std']         = df['cancel_ratio'].resample(freq).std().fillna(0.0)
    bars['kyle_std']           = df['kyle_lambda'].resample(freq).std().fillna(0.0)
    bars['vnet']               = df['vnet'].resample(freq).sum()
    bars['volume_burst']       = df['volume_burst'].resample(freq).max()
    bars['liquidity_sweep']    = df['liquidity_sweep'].resample(freq).max()

    # VWAP
    bars['vwap_z_score'] = df['vwap_z_score'].resample(freq).last()
    bars['current_vwap'] = df['current_vwap'].resample(freq).last()

    # Momentum
    bars['cvd_momentum']         = df['cvd_momentum'].resample(freq).last()
    bars['cvd_price_divergence'] = df['cvd_price_divergence'].resample(freq).last()
    bars['trend_strength']       = df['trend_strength'].resample(freq).last()
    bars['correction_depth']     = df['correction_depth'].resample(freq).last()

    # Microstructure
    bars['micro_atr']    = df['micro_atr'].resample(freq).mean()
    bars['micro_atr_max'] = df['micro_atr'].resample(freq).max()
    bars['fisher_signal'] = df['fisher_signal'].resample(freq).last()
    bars['anomaly']      = df['anomaly'].resample(freq).max()

    # Tick VWAP كنقطة أساس لـ micro_price (قبل الفلاتر؛ يكمّله apply_bar_level_catboost_parities لاحقًا)
    turnover = (
        pd.to_numeric(df['price'], errors='coerce').fillna(0)
        * pd.to_numeric(df['size'], errors='coerce').fillna(0)
    ).resample(freq).sum()
    bars['_turn_sum'] = turnover.astype(np.float64)
    vol_f = pd.to_numeric(bars['volume'], errors='coerce').astype(np.float64).clip(lower=1e-9)
    bars['micro_price'] = bars['_turn_sum'] / vol_f
    bars.drop(columns=['_turn_sum'], errors='ignore', inplace=True)

    # أعمدة advisor إن وُجدت على التيك (تُحمَّل من المصفاة قبل التجليع)
    for col, how in (
        ('obi', 'mean'),
        ('spoofing_ratio', 'max'),
        ('spoofing_duration', 'max'),
        ('liquidity_trap', 'max'),
        ('bid_wall_strength', 'mean'),
        ('ask_wall_strength', 'mean'),
        ('distance_to_wall', 'mean'),
        ('gap_size', 'mean'),
        ('liquidity_density', 'mean'),
        ('inter_event_time', 'mean'),
    ):
        rs = _tick_resample_optional(df, freq, col, how)
        if rs is not None:
            bars[col] = rs

    # Buy/Sell
    buy_vol  = df.loc[df['side'] == 'A', 'size'].resample(freq).sum()
    sell_vol = df.loc[df['side'] == 'B', 'size'].resample(freq).sum()
    bars['buy_volume']  = buy_vol
    bars['sell_volume'] = sell_vol
    total = (buy_vol + sell_vol).clip(lower=1)
    bars['buy_ratio']   = (buy_vol / total).fillna(0.5)

    # Tick count
    bars['tick_count'] = df['price'].resample(freq).count()

    # جسر السيولة الهجين: أسماء صريحة (num_trades, avg_trade_size, …)
    bars = attach_hybrid_liquidity_bridge(bars, df, freq)

    # ── Velocity / micro-dynamics داخل الشمعة (MBO ticks) ───────────────
    bars['cvd_velocity'] = df['cvd'].resample(freq).apply(
        lambda x: float(x.iloc[-1] - x.iloc[0]) / max(len(x), 1) if len(x) > 1 else 0.0,
    ).fillna(0.0)

    def _count_reversals(series: pd.Series) -> float:
        vals = pd.to_numeric(series, errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
        if vals.size < 2:
            return 0.0
        signs = np.sign(vals)
        return float(np.sum(np.diff(signs.astype(np.float64)) != 0))

    tick_ct = bars['tick_count'].clip(lower=1)
    if 'obi' in df.columns:
        bars['imb_reversals'] = df['obi'].resample(freq).apply(_count_reversals).fillna(0.0)
    else:
        bars['imb_reversals'] = 0.0
    cr_sum = df['cancel_ratio'].resample(freq).sum() if 'cancel_ratio' in df.columns else pd.Series(
        np.zeros(len(bars)),
        index=bars.index,
        dtype=np.float64,
    )
    bars['cancel_volume_ratio'] = (
        pd.to_numeric(cr_sum, errors='coerce').divide(pd.to_numeric(tick_ct, errors='coerce')).fillna(0.0)
    )

    # حذف bars فارغة
    bars = bars[bars['volume'] > 0].copy()
    bars = bars.reset_index()

    return bars


def add_day_trading_features(df: pd.DataFrame, freq: str = '5min') -> pd.DataFrame:
    """يضيف features تقنية للـ Day Trading."""
    df = df.copy().sort_values('ts_event').reset_index(drop=True)

    # Bar features
    df['bar_range']  = df['high'] - df['low']
    body = (df['close'] - df['open']).abs()
    df['body_ratio'] = (body / df['bar_range'].clip(lower=1e-8)).clip(0, 1)

    # ATR
    hl   = df['high'] - df['low']
    hcp  = (df['high'] - df['close'].shift(1)).abs()
    lcp  = (df['low']  - df['close'].shift(1)).abs()
    tr   = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14, min_periods=1).mean()

    # RSI
    delta = df['close'].diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs    = gain / loss.clip(lower=1e-8)
    df['rsi_14'] = (100 - (100 / (1 + rs))).clip(0, 100)

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    df['macd_hist'] = macd - macd.ewm(span=9, adjust=False).mean()

    nb_short = max(2, _bars_for_target_minutes(freq, 30))
    df['return_6b']       = df['close'].pct_change(nb_short)
    df['volume_ratio_6b'] = df['volume'] / df['volume'].rolling(nb_short, min_periods=1).mean().clip(lower=1)
    df['cvd_slope_6b']    = df['cvd'].diff(nb_short) / float(nb_short)
    cvw = pd.to_numeric(df['current_vwap'], errors='coerce').astype(np.float64).clip(lower=1e-8)
    df['vwap_dist_6b']    = (pd.to_numeric(df['close'], errors='coerce').astype(np.float64) - cvw) / cvw

    for label, tgt_min in (('1h', 60), ('4h', 240)):
        nb = max(2, _bars_for_target_minutes(freq, tgt_min))
        df[f'return_{label}'] = df['close'].pct_change(nb)
        df[f'volume_ratio_{label}'] = df['volume'] / df['volume'].rolling(nb, min_periods=1).mean().clip(lower=1)
        df[f'cvd_slope_{label}'] = df['cvd'].diff(nb) / float(nb)
        df[f'vwap_dist_{label}'] = (pd.to_numeric(df['close'], errors='coerce').astype(np.float64) - cvw) / cvw

    # Session tags
    t = df['ts_event'].dt.time
    df['is_london']  = ((t >= pd.to_datetime('07:00').time()) & (t < pd.to_datetime('12:00').time())).astype(np.int8)
    df['is_overlap'] = ((t >= pd.to_datetime('12:00').time()) & (t < pd.to_datetime('16:00').time())).astype(np.int8)
    df['is_ny']      = ((t >= pd.to_datetime('13:30').time()) & (t < pd.to_datetime('20:00').time())).astype(np.int8)

    # LOB imbalance مُجمَّع (buy pressure)
    df['lob_imbalance'] = (df['buy_ratio'] - 0.5) * 2  # [-1, 1]

    df = apply_bar_level_catboost_parities(df, freq=freq)
    return df.fillna(0)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: DeepLOB tensors — نافذة 50 بار تاريخية (محاذاة DeepLOBCNN)
# ══════════════════════════════════════════════════════════════════════════════

DEEPLOB_TIME_STEPS_DEFAULT = 50  # متوافق مع modules.deeplob_cnn.N_TIME_STEPS


def attach_mbp_bar_coverage(
    df_bars: pd.DataFrame,
    df_mbp: pd.DataFrame | None,
    freq: str,
    *,
    expected_snap_s: float = 0.5,
) -> pd.DataFrame:
    """نسبة تغطية MBP المتوقعة داخل كل شمعة (0–1)، حسب عدد snapshots / توقع كل expected_snap_s."""
    out = df_bars.copy()
    if df_mbp is None or len(df_mbp) == 0:
        out['mbp_bar_coverage'] = np.float32(0.0)
        return out
    mbp = df_mbp.copy()
    mbp['ts_event'] = pd.to_datetime(mbp['ts_event'], utc=True, errors='coerce').dt.tz_localize(None)
    mbp = mbp.dropna(subset=['ts_event'])
    mbp['_bk'] = mbp['ts_event'].dt.floor(freq)
    cnt_ser = mbp.groupby('_bk', sort=False).size()
    bk = pd.to_datetime(out['ts_event'], errors='coerce').dt.floor(freq)
    merged = bk.map(cnt_ser).fillna(0).astype(np.float64).to_numpy()
    expect_raw = float(_bar_period_seconds(freq)) / float(max(expected_snap_s, 1e-6))
    expect_slots = float(max(expect_raw, 1.0))
    cov = np.clip(merged / expect_slots, 0.0, 1.0).astype(np.float32)
    out['mbp_bar_coverage'] = cov
    del mbp
    return out


def _normalize_lob_tensor_nonflat(
    tensors: np.ndarray,
    *,
    min_samples: int = 10,
    eps: float = 1e-12,
) -> np.ndarray:
    """(µ,σ) لكل قناة من البكسل غير القريبة من الصفر — يقلّل هيمنة الهدوء/الصفور."""
    out = tensors.astype(np.float32, copy=True)
    for c in range(out.shape[-1]):
        ch = out[..., c]
        flat = ch.reshape(-1)
        nz = flat[np.abs(flat) > eps]
        if len(nz) < min_samples:
            continue
        mu = float(np.mean(nz))
        sd = float(np.std(nz))
        if sd > 1e-8:
            out[..., c] = ((ch - mu) / sd).astype(np.float32)
    return out


def fit_lob_norm_params(tensors: np.ndarray, ids: np.ndarray, eps: float = 1e-12) -> dict[int, tuple[float, float]]:
    """Fit DeepLOB channel normalization on a caller-supplied train fold only."""
    sample = np.asarray(tensors[np.asarray(ids, dtype=np.int64)], dtype=np.float32)
    params: dict[int, tuple[float, float]] = {}
    if sample.size == 0:
        return params
    for c in range(sample.shape[-1]):
        flat = sample[..., c].reshape(-1)
        nz = flat[np.abs(flat) > eps]
        if len(nz):
            mu = float(np.mean(nz))
            sd = float(np.std(nz))
        else:
            mu, sd = 0.0, 1.0
        params[int(c)] = (mu, sd if sd > 1e-8 else 1.0)
    return params


def apply_lob_norm(tensors: np.ndarray, ids: np.ndarray, params: dict[int, tuple[float, float]]) -> np.ndarray:
    """Apply previously fitted DeepLOB normalization to selected rows."""
    out = np.asarray(tensors[np.asarray(ids, dtype=np.int64)], dtype=np.float32).copy()
    for c, (mu, sd) in (params or {}).items():
        ci = int(c)
        if 0 <= ci < out.shape[-1]:
            out[..., ci] = ((out[..., ci] - float(mu)) / max(float(sd), 1e-8)).astype(np.float32)
    return out


def _trade_footprint_bar(
    mbo_slice_lo: int,
    mbo_slice_hi: int,
    *,
    mbo_ts: np.ndarray,
    mbo_action: np.ndarray,
    mbo_side: np.ndarray,
    mbo_price: np.ndarray,
    mbo_size: np.ndarray,
    bid0: float,
    ask0: float,
    levels: int,
    tick_med: float,
) -> tuple[np.ndarray, np.ndarray]:
    buy_fp = np.zeros(levels, dtype=np.float32)
    sell_fp = np.zeros(levels, dtype=np.float32)
    tick = float(ask0 - bid0) if (ask0 > 0 and bid0 > 0 and ask0 > bid0) else 0.0
    if tick <= 0 or mbo_slice_hi <= mbo_slice_lo:
        return buy_fp, sell_fp
    for k in range(mbo_slice_lo, mbo_slice_hi):
        if str(mbo_action[k]).strip().upper() != 'T':
            continue
        sz = float(mbo_size[k])
        px = float(mbo_price[k])
        if sz <= 0 or px <= 0:
            continue
        if mbo_side[k] in ('A', 'BID', 'BUY'):
            dist = abs((px - ask0) / tick)
            lvl = max(0, min(levels - 1, int(round(dist))))
            buy_fp[lvl] += np.float32(sz / tick_med)
        elif mbo_side[k] in ('B', 'ASK', 'SELL'):
            dist = abs((bid0 - px) / tick)
            lvl = max(0, min(levels - 1, int(round(dist))))
            sell_fp[lvl] += np.float32(sz / tick_med)
    return buy_fp, sell_fp


def build_rolling_lob_tensors_from_mbp(
    df_mbo: pd.DataFrame,
    df_mbp: pd.DataFrame,
    df_bars: pd.DataFrame,
    *,
    freq: str,
    lookback_bars: int = DEEPLOB_TIME_STEPS_DEFAULT,
    levels: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    لكل شمعة i: tensor (lookback, 20, 3) حيث البعد الزمني = شموع سابقة حقيقية.
    لكل شمعة مصدر: لقطة MBP ذات أقصى |imbalance| داخل الشمعة + بصمة تداول كاملة بنفس الهندسة.
    """
    bars = df_bars.sort_values('ts_event').reset_index(drop=True)
    df_mbp = df_mbp.copy()
    df_mbp['ts_event'] = pd.to_datetime(df_mbp['ts_event'], utc=True, errors='coerce').dt.tz_localize(None)
    df_mbp = df_mbp.dropna(subset=['ts_event']).sort_values('ts_event')
    df_mbo = df_mbo.sort_values('ts_event').reset_index(drop=True)

    bid_px_cols = [f'bid_px_{i:02d}' for i in range(levels)]
    ask_px_cols = [f'ask_px_{i:02d}' for i in range(levels)]
    bid_sz_cols = [f'bid_sz_{i:02d}' for i in range(levels)]
    ask_sz_cols = [f'ask_sz_{i:02d}' for i in range(levels)]

    mbp_ts = df_mbp['ts_event'].to_numpy(dtype='datetime64[ns]', copy=False)
    bid_px = np.vstack([
        pd.to_numeric(df_mbp.get(c), errors='coerce').fillna(0.0).to_numpy(dtype=np.float64, copy=False)
        for c in bid_px_cols]).T
    ask_px = np.vstack([
        pd.to_numeric(df_mbp.get(c), errors='coerce').fillna(0.0).to_numpy(dtype=np.float64, copy=False)
        for c in ask_px_cols]).T
    bid_sz = np.vstack([
        pd.to_numeric(df_mbp.get(c), errors='coerce').fillna(0.0).to_numpy(dtype=np.float64, copy=False)
        for c in bid_sz_cols]).T
    ask_sz = np.vstack([
        pd.to_numeric(df_mbp.get(c), errors='coerce').fillna(0.0).to_numpy(dtype=np.float64, copy=False)
        for c in ask_sz_cols]).T

    mbo_ts = df_mbo['ts_event'].to_numpy(dtype='datetime64[ns]', copy=False)
    mbo_action = df_mbo.get('action', pd.Series('', index=df_mbo.index)).astype(str).str.upper().to_numpy()
    mbo_side = df_mbo.get('side', pd.Series('', index=df_mbo.index)).astype(str).str.upper().to_numpy()
    mbo_price = pd.to_numeric(df_mbo.get('price', 0.0), errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
    mbo_size = pd.to_numeric(df_mbo.get('size', 0.0), errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)

    tick_med = float(
        pd.to_numeric(df_mbo.get('size', 0), errors='coerce').replace(0.0, np.nan).median() or 1.0,
    )
    tick_med = max(tick_med, 1e-6)

    n_bars = len(bars)
    n_lv2 = levels * 2
    T = int(max(lookback_bars, 1))
    bar_ns = np.timedelta64(int(_bar_period_seconds(freq) * 1e9), 'ns')

    bar_snapshots: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = []
    ts_bar = bars['ts_event'].to_numpy(dtype='datetime64[ns]', copy=False)

    for bi in range(n_bars):
        t0 = ts_bar[bi]
        t1 = t0 + bar_ns
        lo_m = int(np.searchsorted(mbp_ts, t0, side='left'))
        hi_m = int(np.searchsorted(mbp_ts, t1, side='left'))
        if hi_m <= lo_m:
            bar_snapshots.append(None)
            continue
        idx_m = np.arange(lo_m, hi_m, dtype=np.int32)
        b_depth_row = bid_sz[idx_m].sum(axis=1)
        a_depth_row = ask_sz[idx_m].sum(axis=1)
        tot_d = b_depth_row + a_depth_row + 1e-9
        imb_mag = np.abs((b_depth_row - a_depth_row) / tot_d)
        peak_local = int(idx_m[np.argmax(imb_mag)])
        depth_raw = np.concatenate([bid_sz[peak_local][::-1], ask_sz[peak_local]], axis=0)
        depth_feat = np.log1p(np.maximum(depth_raw.astype(np.float64), 0.0)).astype(np.float32)
        bid0 = float(bid_px[peak_local][0]) if bid_px.shape[1] else 0.0
        ask0 = float(ask_px[peak_local][0]) if ask_px.shape[1] else 0.0
        lo_t = int(np.searchsorted(mbo_ts, t0, side='left'))
        hi_t = int(np.searchsorted(mbo_ts, t1, side='left'))
        buy_fp, sell_fp = _trade_footprint_bar(
            lo_t,
            hi_t,
            mbo_ts=mbo_ts,
            mbo_action=mbo_action,
            mbo_side=mbo_side,
            mbo_price=mbo_price,
            mbo_size=mbo_size,
            bid0=bid0,
            ask0=ask0,
            levels=levels,
            tick_med=tick_med,
        )
        bar_snapshots.append((depth_feat, buy_fp, sell_fp))

    tensors = np.zeros((n_bars, T, n_lv2, 3), dtype=np.float32)
    timestamps = np.zeros(n_bars, dtype='datetime64[ns]')
    roll_cov = np.zeros(n_bars, dtype=np.float32)

    for bi in range(n_bars):
        timestamps[bi] = ts_bar[bi]
        filled = 0
        for lag in range(T):
            src = bi - (T - 1 - lag)
            if src < 0:
                continue
            snap = bar_snapshots[src]
            if snap is None:
                continue
            depth_feat, buy_fp, sell_fp = snap
            tensors[bi, lag, :, 0] = depth_feat
            tensors[bi, lag, levels:, 1] = buy_fp
            tensors[bi, lag, :levels, 2] = sell_fp[::-1]
            filled += 1
        roll_cov[bi] = float(filled) / float(T)

    if float(np.mean(roll_cov)) < 0.5:
        print(
            '  ⚠️ rolling LOB: متوسط mbp_roll_lob_coverage منخفض — كثير من الشموع غير مكتملة في نافذة الـ 50 بار.'
        )
    return tensors, timestamps, roll_cov


def build_rolling_lob_tensors_mbo_only(
    df_mbo: pd.DataFrame,
    df_bars: pd.DataFrame,
    *,
    freq: str,
    lookback_bars: int = DEEPLOB_TIME_STEPS_DEFAULT,
    n_levels: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """بدون MBP: كل «خطوة زمنية» = شمعة سابقة؛ القنوات من تدفق التيكات فقط (نفس الشكل DeepLOB)."""
    df_mbo = df_mbo.copy()
    df_mbo['ts_event'] = pd.to_datetime(df_mbo['ts_event'])
    df_mbo['bar_key'] = df_mbo['ts_event'].dt.floor(freq)
    tick_med = max(float(pd.to_numeric(df_mbo.get('size', 0), errors='coerce').median() or 1.0), 1e-6)

    bars = df_bars.sort_values('ts_event').reset_index(drop=True)
    n_bars = len(bars)
    T = int(max(lookback_bars, 1))

    feats: list[tuple[float, float, float] | None] = []
    ts_bar = bars['ts_event'].to_numpy()

    for bi in range(n_bars):
        bar_time = pd.Timestamp(ts_bar[bi]).floor(freq)
        sub = df_mbo[df_mbo['bar_key'] == bar_time]
        if sub.empty:
            feats.append(None)
            continue
        buy_vol = float(sub.loc[sub['side'] == 'A', 'size'].sum())
        sell_vol = float(sub.loc[sub['side'] == 'B', 'size'].sum())
        tot_v = buy_vol + sell_vol
        imb = (buy_vol - sell_vol) / max(tot_v, 1.0)
        buy_t = float(sub[(sub['side'] == 'A') & (sub['action'] == 'T')]['size'].sum())
        sell_t = float(sub[(sub['side'] == 'B') & (sub['action'] == 'T')]['size'].sum())
        feats.append((imb, buy_t / tick_med, sell_t / tick_med))

    tensors = np.zeros((n_bars, T, n_levels, 3), dtype=np.float32)
    timestamps = bars['ts_event'].to_numpy(dtype='datetime64[ns]', copy=False)
    roll_cov = np.zeros(n_bars, dtype=np.float32)

    for bi in range(n_bars):
        filled = 0
        for lag in range(T):
            src = bi - (T - 1 - lag)
            if src < 0:
                continue
            f = feats[src]
            if f is None:
                continue
            imb, c1, c2 = f
            tensors[bi, lag, :, 0] = np.float32(imb)
            tensors[bi, lag, :, 1] = np.float32(c1)
            tensors[bi, lag, :, 2] = np.float32(c2)
            filled += 1
        roll_cov[bi] = float(filled) / float(T)

    return tensors, timestamps.copy(), roll_cov


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3a: Event Gate — يكشف لحظات الـ Informed Flow الحقيقي (المشكلة F + I)
# ══════════════════════════════════════════════════════════════════════════════

def detect_microstructure_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    يكشف اللحظات التي يكون فيها informed flow حقيقي ويُضيف:
        - is_event    (int8)   : 1 = حدث حقيقي، 0 = ضجيج
        - event_score (float32): درجة قوة الحدث [0.0, 1.0]

    المنطق:
        hawkes_z  > 1.0 → أوردرات تتسارع فوق الطبيعي
        absorb_z  > 1.0 → صانع سوق نشط بشكل غير عادي
        kyle_z    > 0.5 → السعر يستجيب للـ flow
        cvd_align > 0.6 → ضغط في اتجاه واحد

    Regime Gate (المشكلة I):
        Volatile regime → عتبة أعلى (0.75) لأن الإشارات الضعيفة مضللة
        Trending/Ranging → عتبة قياسية (0.60)

    النتيجة المتوقعة:
        من 100% bars → ~20-30% events حقيقية
        NEUTRAL ينخفض من 70-80% إلى ~35-45%
    """
    df = df.copy()

    def _zscore(series: pd.Series) -> pd.Series:
        roll = series.rolling(EVENT_ZSCORE_WINDOW, min_periods=EVENT_ZSCORE_MIN_PERIODS)
        return ((series - roll.mean()) / (roll.std() + 1e-9)).fillna(0.0)

    # ── حساب Z-scores ────────────────────────────────────────────────────────
    hawkes_z = _zscore(pd.to_numeric(df.get('hawkes_intensity', pd.Series(0.0, index=df.index)),
                                      errors='coerce').fillna(0.0))
    absorb_z = _zscore(pd.to_numeric(df.get('absorption_intensity', pd.Series(0.0, index=df.index)),
                                      errors='coerce').fillna(0.0))
    kyle_z   = _zscore(pd.to_numeric(df.get('kyle_lambda', pd.Series(0.0, index=df.index)),
                                      errors='coerce').fillna(0.0))

    # CVD alignment: نسبة الـ slices في اتجاه واحد (أو CVD momentum كبديل)
    if 'cvd_direction_pct' in df.columns:
        cvd_align = pd.to_numeric(df['cvd_direction_pct'], errors='coerce').fillna(0.5)
    elif 'cvd_momentum' in df.columns:
        # normalize CVD momentum إلى [0,1] كبديل
        cm = pd.to_numeric(df['cvd_momentum'], errors='coerce').fillna(0.0)
        cvd_align = (cm.abs() / (cm.abs().rolling(50, min_periods=5).max() + 1e-9)).clip(0.0, 1.0)
    else:
        cvd_align = pd.Series(0.5, index=df.index)

    # ── حساب event_score المرجح ──────────────────────────────────────────────
    w = EVENT_SCORE_WEIGHTS
    event_score = (
        (hawkes_z  > 1.0).astype(np.float32) * w['hawkes_z_above_1']  +
        (absorb_z  > 1.0).astype(np.float32) * w['absorb_z_above_1']  +
        (kyle_z    > 0.5).astype(np.float32) * w['kyle_z_above_05']   +
        (cvd_align > 0.6).astype(np.float32) * w['cvd_align_above_06']
    ).astype(np.float32)

    # ── Regime Gate (المشكلة I) ───────────────────────────────────────────────
    if 'regime_label' in df.columns:
        regime_s = df['regime_label'].astype(str)
        # عتبة ديناميكية: volatile يحتاج score أعلى
        threshold = regime_s.map(REGIME_EVENT_THRESHOLD).fillna(0.60).astype(np.float32)
    else:
        threshold = pd.Series(0.60, index=df.index, dtype=np.float32)

    df['event_score'] = event_score
    df['is_event'] = (event_score >= threshold.values).astype(np.int8)

    # ── إحصاءات التشخيص ──────────────────────────────────────────────────────
    n_total = len(df)
    n_event = int(df['is_event'].sum())
    print(f"  📊 Event Detection: {n_event:,}/{n_total:,} bars = {n_event/max(n_total,1):.1%} events")
    if 'regime_label' in df.columns:
        for reg in ['trending', 'ranging', 'volatile']:
            mask = df['regime_label'] == reg
            n_reg = int(mask.sum())
            n_ev  = int((mask & (df['is_event'] == 1)).sum())
            if n_reg > 0:
                print(f"     {reg:8s}: {n_ev:,}/{n_reg:,} = {n_ev/n_reg:.1%}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3b: Label by Outcome — First Barrier Hit (المشكلة G + H + J + K)
# ══════════════════════════════════════════════════════════════════════════════

# حدود الجلسات (UTC hour) — نهاية كل جلسة تداول
_SESSION_END_HOUR: dict[str, int] = {
    'asia'   : 8,
    'london' : 16,
    'overlap': 16,
    'ny'     : 21,
}
_FALLBACK_MAX_BARS: int = 20  # حد أقصى مطلق عند غياب معلومات الجلسة


def label_by_outcome(
    df: pd.DataFrame,
    *,
    default_tp_mult: float = 1.5,
    default_sl_mult: float = 1.0,
    default_max_bars: int = 6,
    min_atr: float = 0.0003,
) -> pd.DataFrame:
    """
    يلصق الليبل بناءً على أول حاجز يُضرب (First Barrier Hit).

    الفروق الجوهرية عن build_day_trading_labels:
        1. يعمل على events فقط (is_event == 1) → كفاءة حسابية
        2. Timeout = نهاية الجلسة الحالية (لا يتجاوز حدودها)
        3. TP/SL مخصص لكل Regime من REGIME_TP_SL
        4. Max horizon مخصص لكل Regime من REGIME_MAX_BARS
        5. في حالة تعارض TP+SL في نفس الـ bar → يُفضّل الأقرب من open

    الأعمدة المُضافة:
        bias_label     (int8):   0=LONG, 1=SHORT, 2=NEUTRAL
        path_outcome   (int8):   0=long_tp, 1=short_tp, 2=long_sl, 3=short_sl, 4=timeout
        trade_duration (int16):  عدد bars لنهاية الحدث
        signal_quality (int8):   0=ضعيف, 1=جيد, 2=ممتاز (جلسة لندن/overlap)
        forward_return (float32): عائد نهاية الأفق (للتشخيص فقط)
    """
    df = df.copy().sort_values('ts_event').reset_index(drop=True)
    n = len(df)

    # ── arrays الخروج ─────────────────────────────────────────────────────────
    bias_label     = np.full(n, 2, dtype=np.int8)
    path_outcome   = np.full(n, 4, dtype=np.int8)   # 4 = timeout
    trade_duration = np.zeros(n, dtype=np.int16)
    signal_quality = np.zeros(n, dtype=np.int8)
    forward_return = np.zeros(n, dtype=np.float32)
    label_end_ts = df['ts_event'].copy()
    label_horizon_steps = np.zeros(n, dtype=np.int32)

    # ── arrays السعر ──────────────────────────────────────────────────────────
    close  = pd.to_numeric(df['close'], errors='coerce').to_numpy(dtype=np.float64)
    high   = pd.to_numeric(df['high'],  errors='coerce').to_numpy(dtype=np.float64)
    low    = pd.to_numeric(df['low'],   errors='coerce').to_numpy(dtype=np.float64)
    open_  = pd.to_numeric(df['open'],  errors='coerce').to_numpy(dtype=np.float64) if 'open' in df.columns else close.copy()
    atr    = pd.to_numeric(df['atr_14'], errors='coerce').to_numpy(dtype=np.float64)

    # ── regime + session ──────────────────────────────────────────────────────
    has_regime  = 'regime_label'  in df.columns
    has_session = 'session'       in df.columns
    has_event   = 'is_event'      in df.columns

    regime_arr  = df['regime_label'].astype(str).to_numpy()  if has_regime  else None
    session_arr = df['session'].astype(str).to_numpy()       if has_session else None
    is_ev_arr   = df['is_event'].to_numpy(dtype=np.int8)     if has_event   else np.ones(n, dtype=np.int8)

    # جلسات لندن/overlap → signal_quality = 2 (premium)
    london_ok = np.zeros(n, dtype=bool)
    for col in ('is_london', 'is_overlap'):
        if col in df.columns:
            london_ok |= pd.to_numeric(df[col], errors='coerce').fillna(0).astype(bool).to_numpy()

    # ── helper: نهاية الجلسة ────────────────────────────────────────────────
    ts_arr = df['ts_event'].to_numpy()

    def _session_end(i: int, max_bars_r: int) -> int:
        """يحسب آخر index مسموح به (session-end OR regime max_bars، أيهما أصغر)."""
        cap = min(i + max_bars_r, n - 1)
        if session_arr is None:
            return cap

        sess = session_arr[i]
        end_hour = _SESSION_END_HOUR.get(sess, 21)

        for k in range(i + 1, cap + 1):
            ts = ts_arr[k]
            try:
                hour = pd.Timestamp(ts).hour
            except Exception:
                continue
            if hour >= end_hour:
                return k - 1   # آخر bar قبل نهاية الجلسة
        return cap

    # ── الحلقة الرئيسية ───────────────────────────────────────────────────────
    for i in range(n):
        if not is_ev_arr[i]:
            # ليس حدثاً → لا نُصنِّف (يبقى NEUTRAL افتراضياً)
            continue

        # Regime-aware TP/SL
        regime_i = str(regime_arr[i]) if has_regime else 'ranging'
        tp_mult, sl_mult = REGIME_TP_SL.get(regime_i, (default_tp_mult, default_sl_mult))
        max_bars_r = REGIME_MAX_BARS.get(regime_i, default_max_bars)

        atr_i = max(float(atr[i]), min_atr)
        tp_dist = tp_mult * atr_i
        sl_dist = sl_mult * atr_i

        entry   = float(close[i])
        tp_long  = entry + tp_dist
        sl_long  = entry - sl_dist
        tp_short = entry - tp_dist
        sl_short = entry + sl_dist

        sess_end = _session_end(i, max_bars_r)
        label_end_ts.iloc[i] = ts_arr[sess_end]
        label_horizon_steps[i] = max(0, int(sess_end - i))

        # forward return للتشخيص
        fwd_idx = sess_end
        forward_return[i] = float((close[fwd_idx] - entry) / max(entry, 1e-8))

        # ── امشِ bar بـ bar حتى أول ضربة ──────────────────────────────────────
        first_ev: tuple[str, int] | None = None

        for j in range(i + 1, sess_end + 1):
            h = float(high[j])
            l = float(low[j])
            o = float(open_[j]) if np.isfinite(open_[j]) else 0.5 * (h + l)

            lt = h >= tp_long
            ls = l <= sl_long
            st = l <= tp_short
            ss = h >= sl_short

            picked: str | None = None

            # LONG side
            if lt and ls:
                d_tp = abs(tp_long  - o)
                d_sl = abs(sl_long  - o)
                picked = 'long_tp' if d_tp <= d_sl else 'long_sl'
            elif lt:
                picked = 'long_tp'
            elif ls:
                picked = 'long_sl'

            # SHORT side (لو لم يُحسم من LONG)
            if picked is None:
                if st and ss:
                    d_tp = abs(tp_short - o)
                    d_sl = abs(sl_short - o)
                    picked = 'short_tp' if d_tp <= d_sl else 'short_sl'
                elif st:
                    picked = 'short_tp'
                elif ss:
                    picked = 'short_sl'

            if picked is not None:
                first_ev = (picked, j)
                label_end_ts.iloc[i] = ts_arr[j]
                label_horizon_steps[i] = max(0, int(j - i))
                break

        # ── تعيين الليبل ─────────────────────────────────────────────────────
        if first_ev is None:
            # Timeout = نهاية الجلسة بدون حسم
            bias_label[i]     = 2
            path_outcome[i]   = 4
            trade_duration[i] = max(0, sess_end - i)
        else:
            ev_type, ev_j = first_ev
            trade_duration[i] = ev_j - i
            sq = 2 if london_ok[i] else 1

            if ev_type == 'long_tp':
                bias_label[i]   = 0   # LONG ✅
                path_outcome[i] = 0
                signal_quality[i] = sq
            elif ev_type == 'short_tp':
                bias_label[i]   = 1   # SHORT ✅
                path_outcome[i] = 1
                signal_quality[i] = sq
            elif ev_type == 'long_sl':
                bias_label[i]   = 2   # NEUTRAL (SL = إشارة خاطئة)
                path_outcome[i] = 2
            elif ev_type == 'short_sl':
                bias_label[i]   = 2   # NEUTRAL
                path_outcome[i] = 3

    # ── كتابة النتائج ─────────────────────────────────────────────────────────
    df['bias_label']     = bias_label
    df['path_outcome']   = path_outcome
    df['trade_duration'] = trade_duration
    df['signal_quality'] = signal_quality
    df['forward_return'] = forward_return
    df['label_end_ts'] = label_end_ts
    df['label_horizon_steps'] = label_horizon_steps.astype(np.int32)
    df['effective_horizon'] = label_horizon_steps.astype(np.int32)

    # event_flag: فاز بـ TP فقط (للتدريب الفعلي)
    df['event_flag'] = (
        (df['is_event'] == 1) &
        (df['bias_label'].isin([0, 1])) &
        (df['signal_quality'] > 0)
    ).astype(np.int8)

    # train_event_flag: pool التدريب = events + label != NEUTRAL
    df['train_event_flag'] = (
        (df['is_event'] == 1) &
        (df['bias_label'] != 2)
    ).astype(np.int8)

    # إحصاءات التشخيص
    lbl_counts = {
        'LONG'    : int((df['bias_label'] == 0).sum()),
        'SHORT'   : int((df['bias_label'] == 1).sum()),
        'NEUTRAL' : int((df['bias_label'] == 2).sum()),
    }
    ev_rate = float(df['event_flag'].mean())
    train_r = float(df['train_event_flag'].mean())
    print(f"  🏷️  Labels: {lbl_counts} | event_flag={ev_rate:.1%} | train_pool={train_r:.1%}")

    n0, n1 = lbl_counts['LONG'], lbl_counts['SHORT']
    if min(n0, n1) > 0:
        imb = max(n0, n1) / min(n0, n1)
        print(f"     LONG/SHORT imbalance ratio: {imb:.2f}:1 "
              f"({'✅ متوازن' if imb < 2.0 else '⚠️ يحتاج class_weight'})")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Day Trading Labels
# ══════════════════════════════════════════════════════════════════════════════


def _apply_train_event_pool(
    df: pd.DataFrame,
    *,
    strict_session_atr: bool,
) -> pd.DataFrame:
    """
    train_event_flag: افتراضيًا كل LONG/SHORT (فوز TP فقط — خسائر SL تصبح NEUTRAL بعد إصلاح الليبل).

    لو strict_session_atr=True: يضيق المجموعة إلى جلسات نشطة + حد أدنى للـ ATR.
    """
    out = df.copy()
    base = (out['bias_label'] != 2).astype(np.int8)
    if not strict_session_atr:
        out['train_event_flag'] = base
        return out

    atr_median = float(pd.to_numeric(out['atr_14'], errors='coerce').median() or 0.0)
    floor = max(atr_median * 0.5, 1e-12)
    strong_session = (
        (pd.to_numeric(out.get('is_london', 0), errors='coerce').fillna(0) > 0)
        | (pd.to_numeric(out.get('is_overlap', 0), errors='coerce').fillna(0) > 0)
        | (pd.to_numeric(out.get('is_ny', 0), errors='coerce').fillna(0) > 0)
    )
    above_atr = pd.to_numeric(out['atr_14'], errors='coerce').fillna(0.0) >= floor
    out['train_event_flag'] = (base.astype(bool) & strong_session & above_atr).astype(np.int8)
    return out


def build_day_trading_labels(
    df: pd.DataFrame,
    horizon_bars: int = 6,       # 6 bars × 5min = 30 دقيقة
    tp_atr_mult: float = 1.5,    # TP = 1.5 × ATR
    sl_atr_mult: float = 1.0,    # SL = 1.0 × ATR
    min_atr: float = 0.0003,     # حد أدنى لـ ATR (3 pips لـ GBPUSD)
    *,
    strict_train_pool: bool = False,
) -> pd.DataFrame:
    """
    يبني labels للـ Day Trading على مسار سببي داخل الأفق:

    - يمشي bar-by-bar للأمام حتى أول لمس لأي حاجز TP/SL (long أو short).
    - داخل نفس الشمعة إذا لُمس TP وSL معًا لنفس الاتجاه، يُفضّل الأقرب لسعر الافتتاح (proxy بسيط).
    - فوز TP → bias LONG/SHORT مع جودة جلسة.
    - ضرب SL → NEUTRAL (لا نُصنّف صفقة خاسرة كـ LONG/SHORT).
    - timeout → NEUTRAL.

    لا يعتمد على price_move عند إغلاق الأفق لاختيار الاتجاه (كان يسبب LONG رغم خسارة SL الحقيقية).
    """
    df = df.copy().sort_values('ts_event').reset_index(drop=True)
    n = len(df)

    bias_label = np.full(n, 2, dtype=np.int8)
    signal_quality = np.zeros(n, dtype=np.int8)
    forward_return = np.zeros(n, dtype=np.float32)
    path_outcome = np.full(n, 4, dtype=np.int8)   # 4 = TIMEOUT
    label_end_ts = df['ts_event'].copy()
    label_horizon_steps = np.zeros(n, dtype=np.int32)

    atr = df['atr_14'].to_numpy(dtype=np.float64)
    close = df['close'].to_numpy(dtype=np.float64)
    high = df['high'].to_numpy(dtype=np.float64)
    low = df['low'].to_numpy(dtype=np.float64)
    if 'open' in df.columns:
        open_px = pd.to_numeric(df['open'], errors='coerce').to_numpy(dtype=np.float64)
    else:
        open_px = np.array(close, dtype=np.float64)
    ts = df['ts_event'].values

    for i in range(n - horizon_bars):
        atr_i = max(float(atr[i]), float(min_atr))
        tp_dist = float(tp_atr_mult * atr_i)
        sl_dist = float(sl_atr_mult * atr_i)
        entry = float(close[i])

        tp_long = entry + tp_dist
        sl_long = entry - sl_dist
        tp_short = entry - tp_dist
        sl_short = entry + sl_dist

        horizon_end = min(i + horizon_bars, n - 1)
        label_end_ts.iloc[i] = ts[horizon_end]
        label_horizon_steps[i] = max(0, int(horizon_end - i))
        fwd_close = float(close[horizon_end])
        forward_return[i] = float((fwd_close - entry) / max(entry, 1e-8))

        first_ev: tuple[str, int] | None = None

        for j in range(i + 1, min(i + horizon_bars + 1, n)):
            h = float(high[j])
            l = float(low[j])
            o = float(open_px[j]) if np.isfinite(open_px[j]) else 0.5 * (h + l)

            lt = h >= tp_long
            ls = l <= sl_long
            st = l <= tp_short
            ss = h >= sl_short

            picked: str | None = None

            if lt and ls:
                d_tp = abs(tp_long - o)
                d_sl = abs(sl_long - o)
                picked = 'long_tp' if d_tp <= d_sl else 'long_sl'
            elif lt:
                picked = 'long_tp'
            elif ls:
                picked = 'long_sl'
            elif st and ss:
                d_tp = abs(tp_short - o)
                d_sl = abs(sl_short - o)
                picked = 'short_tp' if d_tp <= d_sl else 'short_sl'
            elif st:
                picked = 'short_tp'
            elif ss:
                picked = 'short_sl'

            if picked is not None:
                first_ev = (picked, j)
                label_end_ts.iloc[i] = ts[j]
                label_horizon_steps[i] = max(0, int(j - i))
                break

        london_ok = bool(df['is_london'].iloc[i] or df['is_overlap'].iloc[i])

        if first_ev is None:
            bias_label[i] = 2
            path_outcome[i] = 4
            signal_quality[i] = 0
        elif first_ev[0] == 'long_tp':
            bias_label[i] = 0
            path_outcome[i] = 0
            signal_quality[i] = 2 if london_ok else 1
        elif first_ev[0] == 'long_sl':
            bias_label[i] = 2
            path_outcome[i] = 2
            signal_quality[i] = 0
        elif first_ev[0] == 'short_tp':
            bias_label[i] = 1
            path_outcome[i] = 1
            signal_quality[i] = 2 if london_ok else 1
        elif first_ev[0] == 'short_sl':
            bias_label[i] = 2
            path_outcome[i] = 3
            signal_quality[i] = 0

    out = df.copy()
    out['bias_label'] = bias_label
    out['signal_quality'] = signal_quality
    out['forward_return'] = forward_return
    out['path_outcome'] = path_outcome
    out['label_end_ts'] = label_end_ts
    out['label_horizon_steps'] = label_horizon_steps.astype(np.int32)
    out['effective_horizon'] = label_horizon_steps.astype(np.int32)
    out['timeout_move_exceeded_band'] = np.zeros(n, dtype=np.int8)

    out['event_flag'] = ((out['bias_label'] != 2) & (out['signal_quality'] > 0)).astype(np.int8)
    out = _apply_train_event_pool(out, strict_session_atr=strict_train_pool)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Soft Labels (نفس المنطق الأصلي)
# ══════════════════════════════════════════════════════════════════════════════

def attach_soft_labels_dt(df: pd.DataFrame) -> pd.DataFrame:
    """
    يطبق soft labels على مستوى الـ bars.
    نفس منطق soft_label_engine.py بدون تعديل.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from modules.soft_label_engine import SoftLabelEngine, SoftLabelConfig
        config = SoftLabelConfig(mode='analytical')
        engine = SoftLabelEngine(config=config)
        df = engine.attach_soft_labels(df)
        print("  ✅ Soft Labels مُرفقة (analytical mode)")
    except Exception as e:
        print(f"  ⚠️ Soft Labels غير متاحة: {e} — يتم التدريب بـ hard labels")
        bi = df['bias_label'].to_numpy()
        df['soft_label'] = np.where(bi == 0, np.float32(0.75), np.where(bi == 1, np.float32(0.25), np.float32(0.50)))
        df['label_confidence'] = 0.5
        df['soft_sample_weight'] = 1.0

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_day_trading_refinery(
    mbo_dir: str,
    mbp_path: str | None,
    output_dir: str,
    freq: str = '5min',
    horizon_bars: int = 6,
    tp_atr_mult: float = 1.5,
    sl_atr_mult: float = 1.0,
    build_lob_tensors: bool = True,
    *,
    strict_train_pool: bool = False,
) -> str:
    """
    Pipeline كاملة: MBO → Day Trading Dataset
    المخرج جاهز مباشرة لـ train_v19.py
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*65)
    print("📊 Day Trading Refinery — QuantSystem V19")
    print(f"   Timeframe: {freq} | Horizon: {horizon_bars} bars")
    print("="*65)

    # ── 1. تحميل MBO ──────────────────────────────────────────────
    print("\n📥 تحميل MBO data...")
    mbo_path = str(mbo_dir)
    if os.path.isfile(mbo_path):
        ext = os.path.splitext(mbo_path)[1].lower()
        if ext in (".csv", ".gz", ".zst"):
            df_mbo = pd.read_csv(mbo_path, low_memory=False, compression="infer")
        elif ext in (".parquet", ".pq", ".snappy"):
            df_mbo = pd.read_parquet(mbo_path)
        else:
            raise FileNotFoundError(f"❌ صيغة ملف MBO غير مدعومة: {mbo_path}")
    else:
        files = sorted(glob.glob(os.path.join(mbo_path, "mbo_final_*.parquet")))
        if not files:
            files = sorted(glob.glob(os.path.join(mbo_path, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"❌ لا توجد ملفات parquet في: {mbo_path}")
        chunks = [pd.read_parquet(f) for f in files]
        df_mbo = pd.concat(chunks, ignore_index=True)
    df_mbo['ts_event'] = pd.to_datetime(df_mbo['ts_event'])
    df_mbo = df_mbo.sort_values('ts_event').reset_index(drop=True)
    print(f"  ✅ {len(df_mbo):,} تيك | {df_mbo['ts_event'].min()} → {df_mbo['ts_event'].max()}")

    # ── 1b. تحميل MBP (اختياري) ─────────────────────────────────────────────
    df_mbp = None
    if mbp_path:
        print("\n📥 تحميل MBP10 data (optional)...")
        mbp_p = str(mbp_path)
        if os.path.isfile(mbp_p):
            ext = os.path.splitext(mbp_p)[1].lower()
            if ext in (".csv", ".gz", ".zst"):
                df_mbp = pd.read_csv(mbp_p, low_memory=False, compression="infer")
            elif ext in (".parquet", ".pq", ".snappy"):
                df_mbp = pd.read_parquet(mbp_p)
            else:
                raise FileNotFoundError(f"❌ صيغة ملف MBP غير مدعومة: {mbp_p}")
        else:
            files = sorted(glob.glob(os.path.join(mbp_p, "*.parquet")))
            if not files:
                raise FileNotFoundError(f"❌ لا توجد ملفات parquet في: {mbp_p}")
            chunks = [pd.read_parquet(f) for f in files]
            df_mbp = pd.concat(chunks, ignore_index=True)
        if df_mbp is not None:
            df_mbp['ts_event'] = pd.to_datetime(df_mbp['ts_event'])
            df_mbp = df_mbp.sort_values('ts_event').reset_index(drop=True)
            print(f"  ✅ {len(df_mbp):,} mbp rows | {df_mbp['ts_event'].min()} → {df_mbp['ts_event'].max()}")

    # ── 2. Aggregate → Bars ───────────────────────────────────────
    print(f"\n⏱️  تجميع في {freq} bars...")
    df_bars = aggregate_mbo_to_bars(df_mbo, freq=freq)
    if df_mbp is not None and len(df_mbp):
        df_bars = attach_mbp_bar_coverage(df_bars, df_mbp, freq)
    else:
        df_bars['mbp_bar_coverage'] = np.float32(0.0)
    print(f"  ✅ {len(df_bars):,} bar")

    # ── 3. Day Trading Features ───────────────────────────────────
    print("\n🔧 بناء Day Trading features...")
    df_bars = add_day_trading_features(df_bars, freq=freq)
    df_bars = enrich_bars_with_intrabar(df_bars, df_mbo, freq=freq, n_slices=12)
    if df_mbp is not None and len(df_mbp):
        df_bars = enrich_bars_with_intrabar_mbp(df_bars, df_mbp, freq=freq, n_slices=12, levels=10)
    print(f"  ✅ {len(df_bars.columns)} feature")

    # ── 4. Event Gate + Labels ────────────────────────────────────
    print(f"\n🔍 كشف Microstructure Events (Event Gate — المشكلة F+I)...")
    df_bars = detect_microstructure_events(df_bars)

    print(f"\n🏷️  بناء Labels — First Barrier Hit — Regime-Aware (المشكلة G+H+J+K)...")
    print(f"   TP/SL per regime: {REGIME_TP_SL}")
    print(f"   Max bars per regime: {REGIME_MAX_BARS}")
    df_labeled = label_by_outcome(
        df_bars,
        default_tp_mult=tp_atr_mult,
        default_sl_mult=sl_atr_mult,
        default_max_bars=horizon_bars,
    )

    # توافق backward: أضف label_end_ts إذا لم توجد
    if 'label_end_ts' not in df_labeled.columns:
        df_labeled['label_end_ts'] = df_labeled['ts_event']
    if 'label_horizon_steps' not in df_labeled.columns:
        df_labeled['label_horizon_steps'] = horizon_bars
    if 'effective_horizon' not in df_labeled.columns:
        df_labeled['effective_horizon'] = df_labeled.get('trade_duration', horizon_bars)

    label_dist   = df_labeled['bias_label'].value_counts().to_dict()
    event_strict = float(df_labeled['event_flag'].mean())
    train_pool   = float(df_labeled['train_event_flag'].mean())
    nc0 = int((df_labeled['bias_label'] == 0).sum())
    nc1 = int((df_labeled['bias_label'] == 1).sum())
    imb = float(max(nc0, nc1)) / float(max(min(nc0, nc1), 1))
    print(
        f"  ✅ Labels: {label_dist} | event_flag={event_strict:.1%} | "
        f"train_pool={train_pool:.1%} | LONG/SHORT ratio≈{imb:.2f}:1"
    )

    # ── 5. Soft Labels ────────────────────────────────────────────
    print("\n🧪 إرفاق Soft Labels...")
    df_final = attach_soft_labels_dt(df_labeled).copy()
    df_final['mbp_roll_lob_coverage'] = np.float32(0.0)

    # ── 6. LOB tensors: نافذة DeepLOB = 50 شمعة تاريخية حقيقية (وليس شرائح داخل bar واحد)
    if build_lob_tensors:
        print("\n👁️  بناء Rolling LOB tensors لـ DeepLOB (50-bar history × 20 levels × 3ch)...")
        if df_mbp is not None and len(df_mbp):
            tensors, tensor_ts, roll_cov = build_rolling_lob_tensors_from_mbp(
                df_mbo,
                df_mbp,
                df_final,
                freq=freq,
                lookback_bars=DEEPLOB_TIME_STEPS_DEFAULT,
                levels=10,
            )
        else:
            tensors, tensor_ts, roll_cov = build_rolling_lob_tensors_mbo_only(
                df_mbo,
                df_final,
                freq=freq,
                lookback_bars=DEEPLOB_TIME_STEPS_DEFAULT,
                n_levels=20,
            )
        df_final['mbp_roll_lob_coverage'] = roll_cov.astype(np.float32)

        tensors_path = os.path.join(output_dir, 'lob_tensors.npy')
        ts_path      = os.path.join(output_dir, 'lob_tensor_timestamps.npy')
        np.save(tensors_path, tensors)
        np.save(ts_path, tensor_ts.astype('datetime64[ns]').astype(np.int64))
        print(f"  ✅ LOB Tensors: {tensors.shape} → {tensors_path}")
        print(
            "  💡 البعد الزمني = الشموع السابقة؛ مع MBP: لقطة peak-imbalance لكل شمعة. "
            f"متوسط التغطية في النافذة={float(np.mean(roll_cov)):.2f}"
        )

    # ── 7. حفظ Dataset ────────────────────────────────────────────
    # إضافة raw__ prefix للـ features (مطلوب لـ train_v19.py)
    raw_cols = {}
    for col in CATBOOST_ADVISOR_FEATURES_DT:
        if col in df_final.columns:
            raw_cols[f'raw__{col}'] = df_final[col]

    df_raw = pd.DataFrame(raw_cols, index=df_final.index)
    df_out = pd.concat([df_final, df_raw], axis=1)

    # تأكد من وجود الأعمدة المطلوبة لـ train_v19.py
    required_cols = ['ts_event', 'label_end_ts', 'bias_label', 'signal_quality',
                     'forward_return', 'event_flag', 'train_event_flag',
                     'soft_label', 'label_confidence', 'soft_sample_weight',
                     'is_event', 'event_score', 'path_outcome', 'trade_duration']
    for col in required_cols:
        if col not in df_out.columns:
            df_out[col] = 0

    # السعر المرجعي للباك تست والمحرك
    if 'price' not in df_out.columns and 'close' in df_out.columns:
        df_out['price'] = pd.to_numeric(df_out['close'], errors='coerce')

    out_path = os.path.join(output_dir, 'day_trading_features.parquet')
    df_out.to_parquet(out_path, index=False)
    print(f"\n💾 Dataset محفوظ: {out_path}")
    print(f"   Rows: {len(df_out):,} | Columns: {len(df_out.columns)}")

    # فلتر التدريب — للتحقق (لا يُحذف، train_v19 يُفلتر بنفسه)
    n_train_events = int(df_out['train_event_flag'].sum())
    n_long  = int((df_out['bias_label'] == 0).sum())
    n_short = int((df_out['bias_label'] == 1).sum())
    print(f"\n   📊 Training pool:")
    print(f"      train_event_flag = True : {n_train_events:,} rows ({n_train_events/max(len(df_out),1):.1%})")
    print(f"      LONG  : {n_long:,} | SHORT : {n_short:,}")
    if 'regime_label' in df_out.columns:
        for reg in ['trending', 'ranging', 'volatile']:
            m = (df_out['regime_label'] == reg) & (df_out['train_event_flag'] == 1)
            print(f"      {reg:8s} events: {int(m.sum()):,}")

    # ── 8. Manifest ────────────────────────────────────────────────
    regime_ev_counts: dict = {}
    if 'regime_label' in df_out.columns:
        for reg in ['trending', 'ranging', 'volatile']:
            m = (df_out['regime_label'] == reg) & (df_out['train_event_flag'] == 1)
            regime_ev_counts[reg] = int(m.sum())

    manifest = {
        'mode'                        : 'day_trading',
        'version'                     : 'v19-event-gate',
        'freq'                        : freq,
        'horizon_bars_default'        : horizon_bars,
        'tp_atr_mult_default'         : tp_atr_mult,
        'sl_atr_mult_default'         : sl_atr_mult,
        'regime_tp_sl'                : {k: list(v) for k, v in REGIME_TP_SL.items()},
        'regime_max_bars'             : REGIME_MAX_BARS,
        'regime_event_threshold'      : REGIME_EVENT_THRESHOLD,
        'rows'                        : len(df_out),
        'label_distribution'          : {str(k): int(v) for k, v in label_dist.items()},
        'event_rate_tp_wins'          : float(event_strict),
        'train_event_pool_rate'       : float(train_pool),
        'train_event_count'           : n_train_events,
        'bias_long_short_counts'      : {'LONG': nc0, 'SHORT': nc1},
        'long_short_imbalance'        : round(imb, 3),
        'regime_train_event_counts'   : regime_ev_counts,
        'catboost_surface_n'          : len(CATBOOST_ADVISOR_FEATURES_DT),
        'catboost_advisor_features'   : CATBOOST_ADVISOR_FEATURES_DT,
        'day_trading_context_features': DAY_TRADING_FEATURES,
        'lob_tensors_built'           : bool(build_lob_tensors),
        'lob_roll_lookback_bars'      : DEEPLOB_TIME_STEPS_DEFAULT,
        'lob_tensor_layout'           : 'rolling_bars_time_x_20_levels_x_3ch_peak_mbp_when_available',
        'ts_min'                      : str(df_out['ts_event'].min()),
        'ts_max'                      : str(df_out['ts_event'].max()),
        'label_logic'                 : (
            'Event Gate (hawkes+absorption+kyle+cvd zscore) → '
            'First Barrier Hit (Regime-Aware TP/SL + Session Timeout) → '
            'SL outcomes → NEUTRAL'
        ),
    }
    with open(os.path.join(output_dir, 'day_trading_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "="*65)
    print("✅ Day Trading Refinery اكتمل")
    print(f"   الخطوة التالية:")
    print(f"   py -3.13 train_v19.py --data {out_path} --output outputs_dt")
    print("="*65)

    return out_path


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Day Trading Refinery — QuantSystem V19')
    p.add_argument('--mbo',     required=True, help='مسار مجلد MBO parquet files')
    p.add_argument('--mbp',     default=None, help='اختياري: مسار ملف/مجلد MBP10 (csv/parquet) لاستخراج ميزات book قوية')
    p.add_argument('--output',  default='pipeline_day_trading/features', help='مسار الـ output')
    p.add_argument('--freq',    default='5min', choices=['5min', '15min', '30min'])
    p.add_argument('--horizon', type=int, default=6,   help='عدد bars للـ label horizon')
    p.add_argument('--tp_mult', type=float, default=1.5, help='TP = tp_mult × ATR')
    p.add_argument('--sl_mult', type=float, default=1.0, help='SL = sl_mult × ATR')
    p.add_argument('--no_lob',  action='store_true', help='تخطي بناء LOB tensors')
    p.add_argument(
        '--strict_train_pool',
        action='store_true',
        help='يضيق train_event_flag: جلسات نشطة فقط + ATR >= 0.5× الوسيط (اتجاهي = فوز TP فقط)',
    )
    args = p.parse_args()

    run_day_trading_refinery(
        mbo_dir=args.mbo,
        mbp_path=args.mbp,
        output_dir=args.output,
        freq=args.freq,
        horizon_bars=args.horizon,
        tp_atr_mult=args.tp_mult,
        sl_atr_mult=args.sl_mult,
        build_lob_tensors=not args.no_lob,
        strict_train_pool=args.strict_train_pool,
    )
