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
try:
    from pykalman import KalmanFilter  # type: ignore
    _KALMAN_OK = True
except Exception:
    KalmanFilter = None  # type: ignore[assignment]
    _KALMAN_OK = False

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
    'lob_imbalance',    # MBP depth imbalance when available; explicit flow proxy fallback otherwise
    'lob_imbalance_is_depth',
    # جسر السيولة (مُجمَّع من التكات داخل الشمعة — أسماء صريحة للمسار الهجين)
    'num_trades',       # = tick_count
    'avg_trade_size',   # volume / num_trades
    'absorption_bar',   # ضغط شراء داخل الشمعة (≈ buy_ratio)
    'order_flow_imbalance',  # (buy_vol - sell_vol) / total ∈ [-1,1]
    'trade_flow_imbalance_proxy',
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
    'cvd_velocity_signed',
    'cvd_net_direction',
    'obi_direction',
    'imb_reversals',
    'cancel_volume_ratio',
    'mbp_bar_coverage',
    'mbp_roll_lob_coverage',
    'mbp_bid_slope_intrabar',
    'mbp_ask_slope_intrabar',
    'mbp_depth_accel',
    'event_direction',
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

TRADE_ACTIONS = {'T', 'F', 'TRADE', 'EXECUTE', 'E', '0'}
# Aggressor side convention for this feed:
# - A / ASK / BUY -> buy-initiated (lifting ask)
# - B / BID / SELL -> sell-initiated (hitting bid)
BUY_SIDES = {'A', 'ASK', 'BUY', 'BOT'}
SELL_SIDES = {'B', 'BID', 'S', 'SELL'}
CVD_DIRECTION_MIN_ABS = 1.0
OBI_DIRECTION_MIN_ABS = 0.05
CORE_MBO_REQUIRED_COLS = [
    'cvd',
    'session_cvd',
    'absorption_intensity',
    'cancel_ratio',
    'micro_atr',
    'volume_burst',
    'inter_event_time',
    'fisher_signal',
    'anomaly',
    'cvd_momentum',
    'cvd_price_divergence',
    'trend_strength',
    'correction_depth',
    'liquidity_sweep',
    'kyle_lambda',
    'hawkes_intensity',
    'vnet',
    'current_vwap',
    'vwap_z_score',
]


def _resolve_mbo_size_column(df_mbo: pd.DataFrame) -> str:
    for candidate in ('size', 'qty', 'volume'):
        if candidate in df_mbo.columns:
            return candidate
    raise KeyError("MBO data must include one of size/qty/volume")


def enrich_mbo_with_core_microstructure(df_mbo: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct core tick-level microstructure features when raw MBO lacks them.
    Uses the same engine families as stage-1 refinery to avoid losing signal quality.
    """
    missing = [c for c in CORE_MBO_REQUIRED_COLS if c not in df_mbo.columns]
    if not missing:
        return df_mbo

    print(
        "  🧠 Core microstructure reconstruction: "
        f"{len(missing)} missing columns -> rebuilding from raw ticks",
    )

    from modules.auto_calibrator import AutoCalibrator
    from modules.microstructure import FastMicrostructureEngine, AbsorptionIntensityEngine, CancelRatioEngine
    from modules.micro_volatility import MicroVolatilityEngine
    from modules.market_research_features import KylesLambdaEngine, HawkesIntensityEngine, VNETEngine
    from modules.context_features import MomentumContextEngine, LiquiditySweepDetector
    from modules.session_features import SessionVWAPEngine
    from modules.fisher_alpha import FastFisherAlpha
    from modules.fim_anomaly import FastFIMDetector

    out = df_mbo.copy()
    out['ts_event'] = pd.to_datetime(out['ts_event'], utc=True, errors='coerce').dt.tz_localize(None)
    out = out.dropna(subset=['ts_event']).sort_values('ts_event').reset_index(drop=True)
    if out.empty:
        return out

    size_col = _resolve_mbo_size_column(out)
    if size_col != 'size':
        out['size'] = pd.to_numeric(out[size_col], errors='coerce').fillna(0.0).astype(np.float64)
    else:
        out['size'] = pd.to_numeric(out['size'], errors='coerce').fillna(0.0).astype(np.float64)
    out['price'] = pd.to_numeric(out['price'], errors='coerce').fillna(0.0).astype(np.float64)
    action_s = out['action'].astype(str).str.upper() if 'action' in out.columns else pd.Series('T', index=out.index)
    side_s = out['side'].astype(str).str.upper() if 'side' in out.columns else pd.Series('', index=out.index)
    order_ids = out['order_id'] if 'order_id' in out.columns else pd.Series([None] * len(out), index=out.index)

    calibrator = AutoCalibrator(n_ticks=2000).fit(out, price_col='price', size_col='size', action_col='action')
    micro = FastMicrostructureEngine()
    absorb = AbsorptionIntensityEngine(min_price_move=max(float(calibrator.min_price_move), 1e-8))
    cancel = CancelRatioEngine()
    mv = MicroVolatilityEngine()
    kyle = KylesLambdaEngine(window=50)
    hawkes = HawkesIntensityEngine()
    vnet = VNETEngine()
    fisher = FastFisherAlpha()
    fim = FastFIMDetector()
    momentum = MomentumContextEngine()
    sweep = LiquiditySweepDetector(
        sweep_threshold=max(0.05, min(float(calibrator.volatility) * 2.0, 0.30)),
    )
    vwap = SessionVWAPEngine()

    n = len(out)
    cvd_arr = np.zeros(n, dtype=np.float64)
    sess_cvd_arr = np.zeros(n, dtype=np.float64)
    absorb_arr = np.zeros(n, dtype=np.float64)
    cancel_arr = np.zeros(n, dtype=np.float64)
    micro_atr_arr = np.zeros(n, dtype=np.float64)
    vol_burst_arr = np.zeros(n, dtype=np.float64)
    iet_arr = np.zeros(n, dtype=np.float64)
    fisher_arr = np.zeros(n, dtype=np.float64)
    anomaly_arr = np.zeros(n, dtype=np.float64)
    cvd_mom_arr = np.zeros(n, dtype=np.float64)
    cvd_div_arr = np.zeros(n, dtype=np.float64)
    trend_arr = np.zeros(n, dtype=np.float64)
    corr_arr = np.zeros(n, dtype=np.float64)
    sweep_arr = np.zeros(n, dtype=np.float64)
    kyle_arr = np.zeros(n, dtype=np.float64)
    hawkes_arr = np.zeros(n, dtype=np.float64)
    vnet_arr = np.zeros(n, dtype=np.float64)
    vwap_arr = np.zeros(n, dtype=np.float64)
    vwap_z_arr = np.zeros(n, dtype=np.float64)
    spoof_arr = np.zeros(n, dtype=np.float64)

    cvd = 0.0
    last_cancel = 0.0
    last_absorb = 0.0
    last_fisher = 0.0
    last_anomaly = 0.0
    last_cvd_mom = 0.0
    last_cvd_div = 0.0
    last_trend = 0.0
    last_corr = 0.0
    last_kyle = 0.0
    last_vnet = 0.0
    last_vwap = 0.0
    last_vwap_z = 0.0
    last_sess_cvd = 0.0
    last_day = None

    for i in range(n):
        px = float(out.at[i, 'price'])
        sz = float(out.at[i, 'size'])
        act = str(action_s.iat[i]).strip().upper()
        sd = str(side_s.iat[i]).strip().upper()
        oid = order_ids.iat[i]
        ts = pd.Timestamp(out.at[i, 'ts_event'])
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        day = ts.date()
        if last_day is None:
            last_day = day
        elif day != last_day:
            # Keep CVD session-local instead of carrying net month bias across days.
            cvd = 0.0
            last_sess_cvd = 0.0
            last_day = day
        ts_ns = int(ts.value)

        spoof_arr[i] = float(micro.process_mbo_tick(act, oid, sd, sz, px, ts_ns))
        cr = float(cancel.process_tick(act, oid, sz))
        if cr != 0.0:
            last_cancel = cr
        cancel_arr[i] = last_cancel
        mu_atr, vol_burst, iet = mv.process_tick(act, px, sz, ts_ns)
        micro_atr_arr[i] = float(mu_atr)
        vol_burst_arr[i] = float(vol_burst)
        iet_arr[i] = float(iet)
        hawkes_arr[i] = float(hawkes.update(ts_ns, act))
        sweep_arr[i] = float(sweep.update(px))

        is_trade = act in TRADE_ACTIONS
        if is_trade:
            is_buy = sd in BUY_SIDES
            is_sell = sd in SELL_SIDES
            if is_buy:
                cvd += sz
            elif is_sell:
                cvd -= sz

            last_absorb = float(absorb.update(px, cvd))
            last_fisher = float(fisher.update_and_get_signal(px, cvd))
            last_anomaly = float(fim.detect_stop_hunts(px))
            last_cvd_mom, last_cvd_div, last_trend, last_corr = momentum.update(px, cvd)
            last_kyle = float(kyle.update(px, sz))
            if is_buy:
                vnet_side = 'B'
            elif is_sell:
                vnet_side = 'A'
            else:
                vnet_side = sd
            last_vnet = float(vnet.update(px, sz, vnet_side))
            last_vwap, last_vwap_z, _, last_sess_cvd = vwap.update(ts, px, float(sz), bool(is_buy))

        cvd_arr[i] = cvd
        sess_cvd_arr[i] = float(last_sess_cvd)
        absorb_arr[i] = float(last_absorb)
        fisher_arr[i] = float(last_fisher)
        anomaly_arr[i] = float(last_anomaly)
        cvd_mom_arr[i] = float(last_cvd_mom)
        cvd_div_arr[i] = float(last_cvd_div)
        trend_arr[i] = float(last_trend)
        corr_arr[i] = float(last_corr)
        kyle_arr[i] = float(last_kyle)
        vnet_arr[i] = float(last_vnet)
        vwap_arr[i] = float(last_vwap)
        vwap_z_arr[i] = float(last_vwap_z)

    rebuilt_cols = {
        'cvd': cvd_arr,
        'session_cvd': sess_cvd_arr,
        'absorption_intensity': absorb_arr,
        'cancel_ratio': cancel_arr,
        'micro_atr': micro_atr_arr,
        'volume_burst': vol_burst_arr,
        'inter_event_time': iet_arr,
        'fisher_signal': fisher_arr,
        'anomaly': anomaly_arr,
        'cvd_momentum': cvd_mom_arr,
        'cvd_price_divergence': cvd_div_arr,
        'trend_strength': trend_arr,
        'correction_depth': corr_arr,
        'liquidity_sweep': sweep_arr,
        'kyle_lambda': kyle_arr,
        'hawkes_intensity': hawkes_arr,
        'vnet': vnet_arr,
        'current_vwap': vwap_arr,
        'vwap_z_score': vwap_z_arr,
        'spoofing_ratio': spoof_arr,
    }

    for col, arr in rebuilt_cols.items():
        if col not in df_mbo.columns:
            out[col] = pd.Series(arr, index=out.index).astype(np.float64)

    print(
        "  ✅ Core microstructure reconstructed "
        f"(rows={len(out):,} | rebuilt_cols={sum(1 for c in rebuilt_cols if c not in df_mbo.columns)})",
    )
    return out


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
    out['trade_flow_imbalance_proxy'] = out['order_flow_imbalance'].astype(np.float32)
    if 'spread' in df_ticks.columns:
        sp = pd.to_numeric(df_ticks['spread'], errors='coerce').resample(freq).mean()
        out['spread_bar'] = sp.reindex(out.index)
        out['spread_bar'] = pd.to_numeric(out['spread_bar'], errors='coerce').fillna(0.0)
    else:
        out['spread_bar'] = np.zeros(len(out), dtype=np.float32)
    return out


def _sign_with_deadband(values, *, min_abs: float) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
    out = np.zeros(arr.shape[0], dtype=np.int8)
    thr = float(max(min_abs, 0.0))
    out[arr > thr] = 1
    out[arr < -thr] = -1
    return out


def apply_mbp_lob_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the legacy `lob_imbalance` column, but make its source explicit:
      - preferred: signed MBP depth imbalance from intrabar MBP aggregation;
      - fallback: trade-flow proxy from MBO buy/sell volume when MBP is unavailable.

    This prevents the tabular model from silently treating order-flow imbalance as
    independent depth information.
    """
    out = df.copy()
    if 'trade_flow_imbalance_proxy' in out.columns:
        proxy = pd.to_numeric(out['trade_flow_imbalance_proxy'], errors='coerce')
    elif 'order_flow_imbalance' in out.columns:
        proxy = pd.to_numeric(out['order_flow_imbalance'], errors='coerce')
    elif 'buy_ratio' in out.columns:
        proxy = (pd.to_numeric(out['buy_ratio'], errors='coerce') - 0.5) * 2.0
    else:
        proxy = pd.Series(0.0, index=out.index)
    proxy = proxy.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0).astype(np.float32)
    out['trade_flow_imbalance_proxy'] = proxy

    depth = None
    if 'mbp_imbalance_signed_peak' in out.columns:
        depth = pd.to_numeric(out['mbp_imbalance_signed_peak'], errors='coerce')
    elif {'mbp_depth_bid_max', 'mbp_depth_ask_max'}.issubset(out.columns):
        bid = pd.to_numeric(out['mbp_depth_bid_max'], errors='coerce').fillna(0.0).astype(np.float64)
        ask = pd.to_numeric(out['mbp_depth_ask_max'], errors='coerce').fillna(0.0).astype(np.float64)
        depth = (bid - ask) / (bid + ask).clip(lower=1e-9)

    if depth is None:
        out['lob_imbalance'] = proxy
        out['lob_imbalance_is_depth'] = np.zeros(len(out), dtype=np.int8)
        return out

    depth = depth.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0).astype(np.float32)
    if 'mbp_bar_coverage' in out.columns:
        cov = pd.to_numeric(out['mbp_bar_coverage'], errors='coerce').fillna(0.0).astype(np.float64)
        has_depth = cov > 0.0
    else:
        has_depth = pd.Series(np.abs(depth.to_numpy(dtype=np.float32)) > 1e-12, index=out.index)

    out['lob_imbalance'] = np.where(has_depth.to_numpy(), depth.to_numpy(), proxy.to_numpy()).astype(np.float32)
    out['lob_imbalance_is_depth'] = has_depth.astype(np.int8).to_numpy()
    return out

def aggregate_mbo_to_bars(df_mbo: pd.DataFrame, freq: str = '5min') -> pd.DataFrame:
    """
    يجمع التيكات في bars مع الحفاظ على order flow features.
    """
    df = df_mbo.copy()
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    df = df.set_index('ts_event').sort_index()

    # يدعم أكثر من اسم لحجم الصفقة في ملفات MBO الخام.
    size_col = 'size'
    if size_col not in df.columns:
        for alt in ('qty', 'volume'):
            if alt in df.columns:
                size_col = alt
                break
    if size_col not in df.columns:
        raise KeyError("aggregate_mbo_to_bars requires one of: size/qty/volume")

    action_s = (
        df['action'].astype(str).str.upper()
        if 'action' in df.columns
        else pd.Series('T', index=df.index, dtype='object')
    )
    side_s = (
        df['side'].astype(str).str.upper()
        if 'side' in df.columns
        else pd.Series('', index=df.index, dtype='object')
    )
    size_s = pd.to_numeric(df[size_col], errors='coerce').fillna(0.0).astype(np.float64)

    def _resample_num(col: str, how: str, default: float = 0.0) -> pd.Series:
        rs = _tick_resample_optional(df, freq, col, how)
        if rs is None:
            return pd.Series(default, index=bars.index, dtype=np.float64)
        return pd.to_numeric(rs, errors='coerce').reindex(bars.index).fillna(default).astype(np.float64)

    # OHLCV
    bars = df['price'].resample(freq).agg(
        open='first', high='max', low='min', close='last'
    )
    bars['volume'] = size_s.resample(freq).sum()

    # CVD
    has_tick_cvd = 'cvd' in df.columns
    if has_tick_cvd:
        tick_cvd = pd.to_numeric(df['cvd'], errors='coerce').ffill().fillna(0.0).astype(np.float64)
    else:
        is_trade = action_s.isin(TRADE_ACTIONS)
        # ملاحظة feed المشروع: side='A' يُعامل كـ buy aggressor و'B' كـ sell aggressor.
        is_buy = side_s.isin({'A', 'BUY', 'ASK'})
        is_sell = side_s.isin({'B', 'SELL', 'BID'})
        signed = np.zeros(len(df), dtype=np.float64)
        signed[(is_trade & is_buy).to_numpy()] = size_s[(is_trade & is_buy)].to_numpy(dtype=np.float64)
        signed[(is_trade & is_sell).to_numpy()] = -size_s[(is_trade & is_sell)].to_numpy(dtype=np.float64)
        tick_cvd = pd.Series(signed, index=df.index, dtype=np.float64).cumsum()
    bars['cvd'] = tick_cvd.resample(freq).last().reindex(bars.index).ffill().fillna(0.0)
    bars['bar_cvd_delta'] = tick_cvd.resample(freq).agg(
        lambda x: float(x.iloc[-1] - x.iloc[0]) if len(x) > 1 else 0.0,
    ).reindex(bars.index).fillna(0.0)
    if 'session_cvd' in df.columns:
        bars['session_cvd'] = _resample_num('session_cvd', 'last', 0.0)
    else:
        signed_step = tick_cvd.diff().fillna(tick_cvd)
        session_cvd = signed_step.groupby(signed_step.index.normalize()).cumsum()
        bars['session_cvd'] = session_cvd.resample(freq).last().reindex(bars.index).fillna(0.0)

    # Order Flow (peak + dispersion داخل الشمعة)
    bars['kyle_lambda'] = _resample_num('kyle_lambda', 'max', 0.0)
    bars['hawkes_intensity'] = _resample_num('hawkes_intensity', 'max', 0.0)
    bars['absorption_intensity'] = _resample_num('absorption_intensity', 'max', 0.0)
    bars['cancel_ratio'] = _resample_num('cancel_ratio', 'max', 0.0)
    bars['absorption_std'] = _resample_num('absorption_intensity', 'max', 0.0)
    bars['cancel_std'] = _resample_num('cancel_ratio', 'max', 0.0)
    bars['kyle_std'] = _resample_num('kyle_lambda', 'max', 0.0)
    bars['vnet'] = _resample_num('vnet', 'sum', 0.0)
    bars['volume_burst'] = _resample_num('volume_burst', 'max', 0.0)
    bars['liquidity_sweep'] = _resample_num('liquidity_sweep', 'max', 0.0)
    if 'absorption_intensity' in df.columns:
        bars['absorption_std'] = (
            pd.to_numeric(df['absorption_intensity'], errors='coerce')
            .resample(freq)
            .std()
            .reindex(bars.index)
            .fillna(0.0)
            .astype(np.float64)
        )
    if 'cancel_ratio' in df.columns:
        bars['cancel_std'] = (
            pd.to_numeric(df['cancel_ratio'], errors='coerce')
            .resample(freq)
            .std()
            .reindex(bars.index)
            .fillna(0.0)
            .astype(np.float64)
        )
    if 'kyle_lambda' in df.columns:
        bars['kyle_std'] = (
            pd.to_numeric(df['kyle_lambda'], errors='coerce')
            .resample(freq)
            .std()
            .reindex(bars.index)
            .fillna(0.0)
            .astype(np.float64)
        )

    # VWAP
    bars['vwap_z_score'] = _resample_num('vwap_z_score', 'last', 0.0)
    bars['current_vwap'] = _resample_num('current_vwap', 'last', np.nan)

    # Momentum
    bars['cvd_momentum'] = _resample_num('cvd_momentum', 'last', 0.0)
    bars['cvd_price_divergence'] = _resample_num('cvd_price_divergence', 'last', 0.0)
    bars['trend_strength'] = _resample_num('trend_strength', 'last', 0.0)
    bars['correction_depth'] = _resample_num('correction_depth', 'last', 0.0)

    # Microstructure
    bars['micro_atr'] = _resample_num('micro_atr', 'mean', 0.0)
    bars['micro_atr_max'] = _resample_num('micro_atr', 'max', 0.0)
    bars['fisher_signal'] = _resample_num('fisher_signal', 'last', 0.0)
    bars['anomaly'] = _resample_num('anomaly', 'max', 0.0)

    # Tick VWAP كنقطة أساس لـ micro_price (قبل الفلاتر؛ يكمّله apply_bar_level_catboost_parities لاحقًا)
    turnover = (
        pd.to_numeric(df['price'], errors='coerce').fillna(0)
        * size_s
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
    buy_vol = size_s[side_s.isin({'A', 'BUY', 'ASK'})].resample(freq).sum()
    sell_vol = size_s[side_s.isin({'B', 'SELL', 'BID'})].resample(freq).sum()
    bars['buy_volume']  = buy_vol
    bars['sell_volume'] = sell_vol
    total = (buy_vol + sell_vol).clip(lower=1)
    bars['buy_ratio']   = (buy_vol / total).fillna(0.5)

    # Tick count
    bars['tick_count'] = df['price'].resample(freq).count()

    # جسر السيولة الهجين: أسماء صريحة (num_trades, avg_trade_size, …)
    bars = attach_hybrid_liquidity_bridge(bars, df, freq)

    # ── Velocity / micro-dynamics داخل الشمعة (MBO ticks) ───────────────
    bars['cvd_velocity'] = tick_cvd.resample(freq).apply(
        lambda x: float(x.iloc[-1] - x.iloc[0]) / max(len(x), 1) if len(x) > 1 else 0.0,
    ).fillna(0.0)
    bars['cvd_velocity_signed'] = pd.to_numeric(bars.get('bar_cvd_delta', 0.0), errors='coerce').fillna(0.0)
    bars['cvd_net_direction'] = _sign_with_deadband(
        bars['cvd_velocity_signed'],
        min_abs=CVD_DIRECTION_MIN_ABS,
    )

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
    obi_base = pd.to_numeric(
        bars['obi'] if 'obi' in bars.columns else bars.get('order_flow_imbalance', 0.0),
        errors='coerce',
    ).fillna(0.0)
    bars['obi_net'] = obi_base.astype(np.float32)
    bars['obi_direction'] = _sign_with_deadband(obi_base, min_abs=OBI_DIRECTION_MIN_ABS)
    cr_sum = pd.to_numeric(df['cancel_ratio'], errors='coerce').resample(freq).sum() if 'cancel_ratio' in df.columns else pd.Series(
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

    # Fallback only. When MBP is present, apply_mbp_lob_imbalance replaces this
    # with signed book-depth imbalance after intrabar MBP enrichment.
    df = apply_mbp_lob_imbalance(df)

    df = apply_bar_level_catboost_parities(df, freq=freq)
    return df.fillna(0)


def assign_regime_label(
    df: pd.DataFrame,
    *,
    roll_window: int = 100,
    min_periods: int = 20,
    volatile_atr_ratio: float = 1.5,
    volatile_hawkes_z: float = 1.5,
    trending_atr_ratio: float = 0.8,
    trending_strength_min: float = 0.55,
    low_liquidity_cov: float = 0.30,
) -> pd.DataFrame:
    """
    يعين regime_label بشكل سببي على مستوى الشموع.
    مهم: volatile يتطلب BOTH (ATR مرتفع + Hawkes مرتفع) لتجنب false positives.
    """
    out = df.copy()
    if len(out) == 0:
        out['regime_label'] = pd.Series(dtype='object')
        out['regime_cluster'] = pd.Series(dtype=np.int8)
        return out

    atr = pd.to_numeric(out.get('atr_14', 0.0), errors='coerce').fillna(0.0).astype(np.float64)
    atr_med = atr.rolling(roll_window, min_periods=min_periods).median().replace(0.0, np.nan)
    atr_ratio = (atr / atr_med).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    hawkes = pd.to_numeric(out.get('hawkes_intensity', 0.0), errors='coerce').fillna(0.0).astype(np.float64)
    hk_roll = hawkes.rolling(roll_window, min_periods=min_periods)
    hawk_z = ((hawkes - hk_roll.mean()) / (hk_roll.std() + 1e-9)).fillna(0.0)

    cvd_signed = pd.to_numeric(
        out.get('cvd_velocity_signed', out.get('bar_cvd_delta', 0.0)),
        errors='coerce',
    ).fillna(0.0).astype(np.float64)
    cvd_ref = cvd_signed.abs().rolling(roll_window, min_periods=min_periods).median().fillna(0.0)
    trend_strength = pd.to_numeric(out.get('trend_strength', 0.0), errors='coerce').fillna(0.0).abs()

    volatile_mask = (atr_ratio > float(volatile_atr_ratio)) & (hawk_z > float(volatile_hawkes_z))
    trending_mask = (
        (~volatile_mask)
        & (atr_ratio > float(trending_atr_ratio))
        & ((cvd_signed.abs() > (cvd_ref + 1e-12)) | (trend_strength > float(trending_strength_min)))
    )

    if 'mbp_bar_coverage' in out.columns:
        cov = pd.to_numeric(out['mbp_bar_coverage'], errors='coerce').fillna(0.0).astype(np.float64)
        lowliq_mask = cov < float(low_liquidity_cov)
    else:
        lowliq_mask = pd.Series(False, index=out.index)

    regime_values = np.select(
        [lowliq_mask.to_numpy(), volatile_mask.to_numpy(), trending_mask.to_numpy()],
        ['low_liquidity', 'volatile', 'trending'],
        default='ranging',
    )
    out['regime_label'] = pd.Series(regime_values, index=out.index, dtype='object')
    out['regime_cluster'] = out['regime_label'].map(
        {'trending': 0, 'ranging': 1, 'volatile': 2, 'low_liquidity': 3},
    ).fillna(1).astype(np.int8)

    counts = out['regime_label'].value_counts().to_dict()
    total = max(len(out), 1)
    print("  ✅ Regime labels assigned:")
    for reg in ('trending', 'ranging', 'volatile', 'low_liquidity'):
        n_reg = int(counts.get(reg, 0))
        print(f"     {reg:12s}: {n_reg:,} ({n_reg/total:.1%})")
    return out


def add_event_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    يصنع event_direction من تصويت 3 مصادر اتجاه:
      1) cvd_velocity_signed
      2) obi_direction
      3) kalman_direction
    +1 LONG-bias | -1 SHORT-bias | 0 ambiguous
    """
    out = df.copy()
    cvd_signed = pd.to_numeric(
        out.get('cvd_velocity_signed', out.get('bar_cvd_delta', 0.0)),
        errors='coerce',
    ).fillna(0.0).astype(np.float64)

    if 'obi_direction' in out.columns:
        obi_dir = pd.to_numeric(out['obi_direction'], errors='coerce').fillna(0).astype(np.int8)
    else:
        obi_raw = pd.to_numeric(out.get('obi', out.get('order_flow_imbalance', 0.0)), errors='coerce').fillna(0.0)
        obi_dir = _sign_with_deadband(obi_raw, min_abs=OBI_DIRECTION_MIN_ABS)
        obi_dir = pd.Series(obi_dir, index=out.index, dtype=np.int8)

    kalman_dir = pd.to_numeric(out.get('kalman_direction', 0), errors='coerce').fillna(0).astype(np.int8)
    vote = (
        _sign_with_deadband(cvd_signed, min_abs=CVD_DIRECTION_MIN_ABS)
        + np.sign(obi_dir.to_numpy(dtype=np.int8)).astype(np.int8)
        + np.sign(kalman_dir.to_numpy(dtype=np.int8)).astype(np.int8)
    )
    direction = np.where(vote >= 2, 1, np.where(vote <= -2, -1, 0)).astype(np.int8)
    out['event_direction'] = direction

    if 'is_event' in out.columns:
        ev_mask = pd.to_numeric(out['is_event'], errors='coerce').fillna(0).astype(np.int8) == 1
        if bool(ev_mask.any()):
            ev_dir = out.loc[ev_mask, 'event_direction']
            vc = ev_dir.value_counts().to_dict()
            n = int(ev_mask.sum())
            print(
                "  🧭 Event direction votes: "
                f"long={int(vc.get(1, 0)):,} ({int(vc.get(1, 0))/max(n,1):.1%}) | "
                f"short={int(vc.get(-1, 0)):,} ({int(vc.get(-1, 0))/max(n,1):.1%}) | "
                f"amb={int(vc.get(0, 0)):,} ({int(vc.get(0, 0))/max(n,1):.1%})"
            )
    return out


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


def _coverage_stats(series: pd.Series | np.ndarray, *, low_threshold: float) -> dict:
    vals = pd.to_numeric(pd.Series(series), errors='coerce').fillna(0.0).astype(np.float64).to_numpy()
    if vals.size == 0:
        return {
            'mean': 0.0,
            'median': 0.0,
            'p25': 0.0,
            'p75': 0.0,
            'low_ratio': 0.0,
            'threshold': float(low_threshold),
        }
    return {
        'mean': float(np.mean(vals)),
        'median': float(np.median(vals)),
        'p25': float(np.percentile(vals, 25)),
        'p75': float(np.percentile(vals, 75)),
        'low_ratio': float(np.mean(vals < float(low_threshold))),
        'threshold': float(low_threshold),
    }


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
        if mbo_side[k] in ('A', 'ASK', 'BUY', 'BOT'):
            dist = abs((px - ask0) / tick)
            lvl = max(0, min(levels - 1, int(round(dist))))
            buy_fp[lvl] += np.float32(sz / tick_med)
        elif mbo_side[k] in ('B', 'BID', 'S', 'SELL'):
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

    def _col_or_zeros(frame: pd.DataFrame, col: str) -> pd.Series:
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors='coerce').fillna(0.0)
        return pd.Series(np.zeros(len(frame), dtype=np.float64), index=frame.index, dtype=np.float64)

    mbp_ts = df_mbp['ts_event'].to_numpy(dtype='datetime64[ns]', copy=False)
    bid_px = np.vstack([
        _col_or_zeros(df_mbp, c).to_numpy(dtype=np.float64, copy=False)
        for c in bid_px_cols]).T
    ask_px = np.vstack([
        _col_or_zeros(df_mbp, c).to_numpy(dtype=np.float64, copy=False)
        for c in ask_px_cols]).T
    bid_sz = np.vstack([
        _col_or_zeros(df_mbp, c).to_numpy(dtype=np.float64, copy=False)
        for c in bid_sz_cols]).T
    ask_sz = np.vstack([
        _col_or_zeros(df_mbp, c).to_numpy(dtype=np.float64, copy=False)
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

    bars_with_snapshot = 0
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
        bars_with_snapshot += 1

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

    snap_ratio = float(bars_with_snapshot) / float(max(n_bars, 1))
    mean_roll_cov = float(np.mean(roll_cov)) if len(roll_cov) else 0.0
    low_roll_ratio = float(np.mean(roll_cov < 0.50)) if len(roll_cov) else 1.0
    print(
        "  📊 LOB coverage telemetry: "
        f"bar_snapshots={bars_with_snapshot:,}/{n_bars:,} ({snap_ratio:.1%}) | "
        f"roll_mean={mean_roll_cov:.1%} | roll_low(<50%)={low_roll_ratio:.1%}"
    )
    if mean_roll_cov < 0.5 or low_roll_ratio > 0.4:
        print(
            "  ⚠️ rolling LOB coverage weak — CNN quality may degrade unless MBP density improves."
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

def detect_microstructure_events(
    df: pd.DataFrame,
    *,
    threshold_scale: float = 1.0,
    threshold_shift: float = 0.0,
) -> pd.DataFrame:
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
        base_threshold = regime_s.map(REGIME_EVENT_THRESHOLD).fillna(0.60).astype(np.float32)
    else:
        base_threshold = pd.Series(0.60, index=df.index, dtype=np.float32)

    scale = float(max(threshold_scale, 0.01))
    shift = float(threshold_shift)
    threshold = (base_threshold.astype(np.float64) * scale + shift).clip(0.05, 0.95).astype(np.float32)

    df['event_score'] = event_score
    df['is_event'] = (event_score >= threshold.values).astype(np.int8)

    # ── إحصاءات التشخيص ──────────────────────────────────────────────────────
    n_total = len(df)
    n_event = int(df['is_event'].sum())
    print(
        f"  📊 Event Detection: {n_event:,}/{n_total:,} bars = {n_event/max(n_total,1):.1%} events "
        f"(threshold_scale={scale:.2f}, shift={shift:+.2f})"
    )
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
_KALMAN_EVENT_FLOOR: float = 0.70

# neutral_reason parity with labels_v19
NEUTRAL_REASON_NONE       = 0
NEUTRAL_REASON_TIMEOUT    = 1
NEUTRAL_REASON_LONG_SL    = 2
NEUTRAL_REASON_SHORT_SL   = 3
NEUTRAL_REASON_KALMAN     = 4
NEUTRAL_REASON_WEAK_EVENT = 5


def add_kalman_trend(
    df: pd.DataFrame,
    *,
    obs_noise: float = 1e-3,
    trans_noise: float = 1e-5,
) -> pd.DataFrame:
    """
    يضيف kalman_trend / kalman_direction لفلترة الاتجاهات الضعيفة عكس الترند.
    إذا pykalman غير متاح، نستخدم EWMA fallback للحفاظ على نفس العقد.
    """
    out = df.copy()
    close_src = out['close'] if 'close' in out.columns else pd.Series(np.zeros(len(out), dtype=np.float64), index=out.index)
    close = pd.to_numeric(close_src, errors='coerce').ffill().fillna(0.0).astype(np.float64)
    if len(close) == 0:
        out['kalman_trend'] = np.float32(0.0)
        out['kalman_direction'] = np.int8(0)
        return out
    if _KALMAN_OK and KalmanFilter is not None:
        kf = KalmanFilter(
            transition_matrices=[[1.0]],
            observation_matrices=[[1.0]],
            initial_state_mean=[float(close.iloc[0])],
            observation_covariance=[[float(max(obs_noise, 1e-9))]],
            transition_covariance=[[float(max(trans_noise, 1e-12))]],
        )
        state_means, _ = kf.filter(close.to_numpy(dtype=np.float64).reshape(-1, 1))
        trend = pd.Series(state_means[:, 0], index=out.index, dtype=np.float64)
    else:
        trend = close.ewm(span=12, adjust=False).mean()
        print("  ⚠️ pykalman غير متاح — using EWMA trend fallback for kalman_direction.")
    direction = np.sign(trend.diff().fillna(0.0)).astype(np.int8)
    out['kalman_trend'] = trend.astype(np.float32)
    out['kalman_direction'] = direction
    return out


def label_by_outcome(
    df: pd.DataFrame,
    *,
    default_tp_mult: float = 1.5,
    default_sl_mult: float = 1.0,
    default_max_bars: int = 6,
    min_atr: float = 0.0003,
    kalman_event_floor: float = _KALMAN_EVENT_FLOOR,
    weak_event_to_directional: bool = False,
    weak_event_min_move_atr: float = 0.35,
    sl_to_opposite: bool = False,
    include_weak_directional_in_train: bool = False,
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
        path_outcome   (int8):   0=long_tp, 1=short_tp, 2=long_sl, 3=short_sl, 4=timeout, 5=weak_long, 6=weak_short
        neutral_reason (int8):   0=none, 1=timeout, 2=long_sl, 3=short_sl, 4=kalman, 5=weak_event
        trade_duration (int16):  عدد bars لنهاية الحدث
        signal_quality (int8):   0=ضعيف, 1=جيد, 2=ممتاز (جلسة لندن/overlap)
        forward_return (float32): عائد نهاية الأفق (للتشخيص فقط)
    """
    df = df.copy().sort_values('ts_event').reset_index(drop=True)
    n = len(df)

    # ── arrays الخروج ─────────────────────────────────────────────────────────
    bias_label     = np.full(n, 2, dtype=np.int8)
    path_outcome   = np.full(n, 4, dtype=np.int8)   # 4 = timeout
    neutral_reason = np.full(n, NEUTRAL_REASON_NONE, dtype=np.int8)
    trade_duration = np.zeros(n, dtype=np.int16)
    signal_quality = np.zeros(n, dtype=np.int8)
    forward_return = np.zeros(n, dtype=np.float32)

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
    ev_score_src = df['event_score'] if 'event_score' in df.columns else pd.Series(0.0, index=df.index)
    kalman_src = df['kalman_direction'] if 'kalman_direction' in df.columns else pd.Series(0, index=df.index)
    event_dir_src = df['event_direction'] if 'event_direction' in df.columns else pd.Series(0, index=df.index)
    event_score_arr = pd.to_numeric(ev_score_src, errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
    kalman_dir_arr = pd.to_numeric(kalman_src, errors='coerce').fillna(0).to_numpy(dtype=np.int8)
    event_dir_arr = pd.to_numeric(event_dir_src, errors='coerce').fillna(0).to_numpy(dtype=np.int8)

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
        regime_i = str(regime_arr[i]) if has_regime else 'ranging'
        max_bars_r = REGIME_MAX_BARS.get(regime_i, default_max_bars)
        atr_i = max(float(atr[i]), min_atr)
        entry = float(close[i])
        sess_end = _session_end(i, max_bars_r)
        fwd_idx = min(i + max_bars_r, n - 1)
        forward_return[i] = float((close[fwd_idx] - entry) / max(entry, 1e-8))

        if not is_ev_arr[i]:
            # ليس حدثاً: افتراضياً NEUTRAL، ويمكن (اختياريًا) تحويله لاتجاه ضعيف إذا الحركة واضحة.
            if weak_event_to_directional:
                move_px = float(close[sess_end] - entry) if sess_end >= i else 0.0
                move_thr = float(max(weak_event_min_move_atr, 0.0)) * atr_i
                trade_duration[i] = max(0, sess_end - i)
                if move_px >= move_thr:
                    bias_label[i] = 0
                    path_outcome[i] = 5  # weak_long
                    signal_quality[i] = 1 if london_ok[i] else 0
                    neutral_reason[i] = NEUTRAL_REASON_NONE
                elif move_px <= -move_thr:
                    bias_label[i] = 1
                    path_outcome[i] = 6  # weak_short
                    signal_quality[i] = 1 if london_ok[i] else 0
                    neutral_reason[i] = NEUTRAL_REASON_NONE
                else:
                    neutral_reason[i] = NEUTRAL_REASON_WEAK_EVENT
            else:
                neutral_reason[i] = NEUTRAL_REASON_WEAK_EVENT
            continue

        # Regime-aware TP/SL
        event_score_i = float(event_score_arr[i])
        kalman_dir_i = int(kalman_dir_arr[i])
        event_dir_i = int(event_dir_arr[i])
        tp_mult, sl_mult = REGIME_TP_SL.get(regime_i, (default_tp_mult, default_sl_mult))
        allow_long = True
        allow_short = True
        if event_dir_i > 0:
            allow_short = False
        elif event_dir_i < 0:
            allow_long = False
        else:
            if kalman_dir_i == 1 and event_score_i < float(kalman_event_floor):
                allow_short = False
            elif kalman_dir_i == -1 and event_score_i < float(kalman_event_floor):
                allow_long = False

        tp_dist = tp_mult * atr_i
        sl_dist = sl_mult * atr_i

        tp_long  = entry + tp_dist
        sl_long  = entry - sl_dist
        tp_short = entry - tp_dist
        sl_short = entry + sl_dist

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

            if not allow_long:
                lt = False
                ls = False
            if not allow_short:
                st = False
                ss = False

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
                break

        # ── تعيين الليبل ─────────────────────────────────────────────────────
        if first_ev is None:
            # Timeout = نهاية الجلسة بدون حسم
            bias_label[i]     = 2
            path_outcome[i]   = 4
            trade_duration[i] = max(0, sess_end - i)
            if (not allow_long) or (not allow_short):
                neutral_reason[i] = NEUTRAL_REASON_KALMAN
            else:
                neutral_reason[i] = NEUTRAL_REASON_TIMEOUT
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
                if sl_to_opposite:
                    bias_label[i] = 1  # map loss to opposite direction (optional aggressive mode)
                    signal_quality[i] = 1
                else:
                    bias_label[i] = 2   # NEUTRAL (SL = إشارة خاطئة)
                path_outcome[i] = 2
                if bias_label[i] == 2:
                    neutral_reason[i] = NEUTRAL_REASON_LONG_SL
            elif ev_type == 'short_sl':
                if sl_to_opposite:
                    bias_label[i] = 0
                    signal_quality[i] = 1
                else:
                    bias_label[i] = 2   # NEUTRAL
                path_outcome[i] = 3
                if bias_label[i] == 2:
                    neutral_reason[i] = NEUTRAL_REASON_SHORT_SL

    # ── كتابة النتائج ─────────────────────────────────────────────────────────
    df['bias_label']     = bias_label
    df['path_outcome']   = path_outcome
    df['neutral_reason'] = neutral_reason
    df['trade_duration'] = trade_duration
    df['signal_quality'] = signal_quality
    df['forward_return'] = forward_return

    # event_flag: فاز بـ TP فقط (للتدريب الفعلي)
    df['event_flag'] = (
        (df['is_event'] == 1) &
        (df['bias_label'].isin([0, 1])) &
        (df['signal_quality'] > 0)
    ).astype(np.int8)

    # train_event_flag: افتراضيًا directional داخل events فقط.
    if include_weak_directional_in_train:
        df['train_event_flag'] = (df['bias_label'] != 2).astype(np.int8)
    else:
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
    out['label_horizon_steps'] = horizon_bars
    out['effective_horizon'] = np.full(n, int(horizon_bars), dtype=np.int32)
    out['timeout_move_exceeded_band'] = np.zeros(n, dtype=np.int8)

    out['event_flag'] = ((out['bias_label'] != 2) & (out['signal_quality'] > 0)).astype(np.int8)
    out = _apply_train_event_pool(out, strict_session_atr=strict_train_pool)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Soft Labels (Regime-aware + engine)
# ══════════════════════════════════════════════════════════════════════════════

def add_soft_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Soft labels day-trading-aware:
      LONG  -> [0.55 .. 1.00]
      SHORT -> [0.00 .. 0.45]
      NEUTRAL -> 0.50
    القوة = 0.6*event_score + 0.4*speed (مدة أقصر = أقوى).
    """
    out = df.copy()
    n = len(out)
    soft = np.full(n, 0.5, dtype=np.float32)
    labels_src = out['bias_label'] if 'bias_label' in out.columns else pd.Series(2, index=out.index)
    events_src = out['is_event'] if 'is_event' in out.columns else pd.Series(0, index=out.index)
    score_src = out['event_score'] if 'event_score' in out.columns else pd.Series(0.0, index=out.index)
    duration_src = out['trade_duration'] if 'trade_duration' in out.columns else pd.Series(0, index=out.index)
    labels = pd.to_numeric(labels_src, errors='coerce').fillna(2).to_numpy(dtype=np.int8)
    events = pd.to_numeric(events_src, errors='coerce').fillna(0).to_numpy(dtype=np.int8)
    ev_score = pd.to_numeric(score_src, errors='coerce').fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=np.float64)
    duration = pd.to_numeric(duration_src, errors='coerce').fillna(0).to_numpy(dtype=np.float64)
    regime_s = out.get('regime_label', pd.Series('ranging', index=out.index)).astype(str).to_numpy()
    for i in np.where(events == 1)[0]:
        label_i = int(labels[i])
        if label_i == 2:
            continue
        regime_i = str(regime_s[i]) if i < len(regime_s) else 'ranging'
        max_dur = float(max(int(REGIME_MAX_BARS.get(regime_i, 6)), 1))
        speed = float(np.clip(1.0 - (duration[i] / max_dur), 0.0, 1.0))
        strength = float(np.clip(ev_score[i] * 0.6 + speed * 0.4, 0.0, 1.0))
        if label_i == 0:
            soft[i] = np.float32(0.55 + strength * 0.45)
        elif label_i == 1:
            soft[i] = np.float32(0.45 - strength * 0.45)
    conf = np.clip(np.abs(soft - 0.5) * 2.0, 0.0, 1.0).astype(np.float32)
    out['soft_label'] = soft
    out['label_confidence'] = conf
    out['soft_sample_weight'] = (1.0 + conf).astype(np.float32)
    return out


def attach_soft_labels_dt(df: pd.DataFrame) -> pd.DataFrame:
    """
    يطبق soft labels على مستوى الـ bars:
      1) day-trading regime-aware soft labels (مضمون دائماً)
      2) enrich اختياري عبر soft_label_engine (إذا متاح) دون فقد soft_label الأساسي
    """
    base = add_soft_labels(df)
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from modules.soft_label_engine import SoftLabelEngine, SoftLabelConfig
        config = SoftLabelConfig(mode='analytical')
        engine = SoftLabelEngine(config=config)
        enriched = engine.attach_soft_labels(base.copy())
        enriched['soft_label'] = base['soft_label'].astype(np.float32)
        base_conf = pd.to_numeric(base.get('label_confidence', 0.5), errors='coerce').fillna(0.5).astype(np.float32)
        if 'label_confidence' in enriched.columns:
            eng_conf = pd.to_numeric(enriched['label_confidence'], errors='coerce').fillna(0.5).astype(np.float32)
            enriched['label_confidence'] = np.maximum(eng_conf, base_conf).astype(np.float32)
        else:
            enriched['label_confidence'] = base_conf
        if 'soft_sample_weight' in enriched.columns:
            eng_w = pd.to_numeric(enriched['soft_sample_weight'], errors='coerce').fillna(1.0).astype(np.float32)
            scale = (0.75 + 0.5 * base_conf).astype(np.float32)
            enriched['soft_sample_weight'] = (eng_w * scale).astype(np.float32)
        else:
            enriched['soft_sample_weight'] = base['soft_sample_weight'].astype(np.float32)
        print("  ✅ Soft Labels مُرفقة (day-trading regime-aware + analytical enrich)")
        return enriched
    except Exception as e:
        print(f"  ⚠️ Soft label engine غير متاح: {e} — using day-trading regime-aware soft labels فقط.")
        return base


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
    event_threshold_scale: float = 1.0,
    event_threshold_shift: float = 0.0,
    kalman_event_floor: float = _KALMAN_EVENT_FLOOR,
    weak_event_to_directional: bool = False,
    weak_event_min_move_atr: float = 0.35,
    sl_to_opposite: bool = False,
    include_weak_directional_in_train: bool = False,
) -> str:
    """
    Pipeline كاملة: MBO → Day Trading Dataset
    المخرج جاهز مباشرة لـ train_v19.py
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*65)
    print("📊 Day Trading Refinery — QuantSystem V19")
    print(f"   Timeframe: {freq} | Horizon: {horizon_bars} bars")
    print(
        "   EventGate tuning: "
        f"threshold_scale={float(event_threshold_scale):.2f}, "
        f"threshold_shift={float(event_threshold_shift):+.2f}, "
        f"kalman_floor={float(kalman_event_floor):.2f}"
    )
    print(
        "   Label policy: "
        f"weak_to_dir={bool(weak_event_to_directional)} "
        f"(min_move_atr={float(weak_event_min_move_atr):.2f}) | "
        f"sl_to_opposite={bool(sl_to_opposite)} | "
        f"include_weak_train={bool(include_weak_directional_in_train)}"
    )
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
    mbo_dup_ts = int(df_mbo['ts_event'].duplicated().sum())
    mbo_key_cols = [c for c in ('ts_event', 'action', 'side', 'price', 'size', 'order_id') if c in df_mbo.columns]
    mbo_dup_key = int(df_mbo.duplicated(subset=mbo_key_cols).sum()) if mbo_key_cols else 0
    if mbo_dup_ts:
        print(
            f"  ⚠️ MBO duplicate ts_event rows: {mbo_dup_ts:,} "
            f"(duplicate_key_rows={mbo_dup_key:,}; not dropped because tick feeds can share timestamps)"
        )
    df_mbo = enrich_mbo_with_core_microstructure(df_mbo)
    print(f"  ✅ {len(df_mbo):,} تيك | {df_mbo['ts_event'].min()} → {df_mbo['ts_event'].max()}")

    # ── 1b. تحميل MBP (اختياري) ─────────────────────────────────────────────
    df_mbp = None
    mbp_dup_ts = 0
    mbp_dup_key = 0
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
            mbp_dup_ts = int(df_mbp['ts_event'].duplicated().sum())
            mbp_key_cols = [c for c in ('ts_event', 'bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00') if c in df_mbp.columns]
            mbp_dup_key = int(df_mbp.duplicated(subset=mbp_key_cols).sum()) if mbp_key_cols else 0
            if mbp_dup_ts:
                print(
                    f"  ⚠️ MBP duplicate ts_event rows: {mbp_dup_ts:,} "
                    f"(duplicate_key_rows={mbp_dup_key:,}; snapshots kept for intrabar aggregation)"
                )
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
    df_bars = apply_mbp_lob_imbalance(df_bars)
    print("\n📈 Kalman Trend Filter...")
    df_bars = add_kalman_trend(df_bars)
    print("\n🧭 Regime assignment...")
    df_bars = assign_regime_label(df_bars)
    print(f"  ✅ {len(df_bars.columns)} feature")

    # ── 4. Event Gate + Labels ────────────────────────────────────
    print(f"\n🔍 كشف Microstructure Events (Event Gate — المشكلة F+I)...")
    df_bars = detect_microstructure_events(
        df_bars,
        threshold_scale=event_threshold_scale,
        threshold_shift=event_threshold_shift,
    )
    print("\n🧭 Directional voting (CVD + OBI + Kalman)...")
    df_bars = add_event_direction(df_bars)

    print(f"\n🏷️  بناء Labels — First Barrier Hit — Regime-Aware (المشكلة G+H+J+K)...")
    print(f"   TP/SL per regime: {REGIME_TP_SL}")
    print(f"   Max bars per regime: {REGIME_MAX_BARS}")
    df_labeled = label_by_outcome(
        df_bars,
        default_tp_mult=tp_atr_mult,
        default_sl_mult=sl_atr_mult,
        default_max_bars=horizon_bars,
        kalman_event_floor=kalman_event_floor,
        weak_event_to_directional=weak_event_to_directional,
        weak_event_min_move_atr=weak_event_min_move_atr,
        sl_to_opposite=sl_to_opposite,
        include_weak_directional_in_train=include_weak_directional_in_train,
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
                     'is_event', 'event_score', 'event_direction', 'path_outcome', 'trade_duration',
                     'neutral_reason', 'kalman_direction', 'regime_label', 'regime_cluster']
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

    mbp_bar_cov_stats = _coverage_stats(df_out.get('mbp_bar_coverage', 0.0), low_threshold=0.30)
    mbp_roll_cov_stats = _coverage_stats(df_out.get('mbp_roll_lob_coverage', 0.0), low_threshold=0.50)
    print(
        "   📊 MBP coverage: "
        f"bar_mean={mbp_bar_cov_stats['mean']:.1%}, bar_low(<30%)={mbp_bar_cov_stats['low_ratio']:.1%} | "
        f"roll_mean={mbp_roll_cov_stats['mean']:.1%}, roll_low(<50%)={mbp_roll_cov_stats['low_ratio']:.1%}"
    )
    lob_depth_ratio = float(
        pd.to_numeric(df_out.get('lob_imbalance_is_depth', 0), errors='coerce').fillna(0).astype(np.float64).mean()
    )
    lob_flow_corr = None
    if {'lob_imbalance', 'order_flow_imbalance'}.issubset(df_out.columns) and len(df_out) >= 3:
        lob_flow_corr_raw = df_out[['lob_imbalance', 'order_flow_imbalance']].corr(method='spearman').iloc[0, 1]
        if pd.notna(lob_flow_corr_raw):
            lob_flow_corr = float(lob_flow_corr_raw)
    print(
        "   📊 LOB imbalance source: "
        f"depth_rows={lob_depth_ratio:.1%} | "
        f"spearman_vs_order_flow={lob_flow_corr if lob_flow_corr is not None else 'n/a'}"
    )

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
    neutral_counts = {
        str(k): int(v) for k, v in pd.to_numeric(df_out.get('neutral_reason', 0), errors='coerce')
        .fillna(0).astype(int).value_counts().to_dict().items()
    }
    event_direction_counts = {
        str(k): int(v) for k, v in pd.to_numeric(df_out.get('event_direction', 0), errors='coerce')
        .fillna(0).astype(int).value_counts().to_dict().items()
    }

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
        'event_threshold_scale'       : float(event_threshold_scale),
        'event_threshold_shift'       : float(event_threshold_shift),
        'kalman_event_floor'          : float(kalman_event_floor),
        'weak_event_to_directional'   : bool(weak_event_to_directional),
        'weak_event_min_move_atr'     : float(weak_event_min_move_atr),
        'sl_to_opposite'              : bool(sl_to_opposite),
        'include_weak_directional_in_train': bool(include_weak_directional_in_train),
        'rows'                        : len(df_out),
        'mbo_duplicate_ts_rows'       : mbo_dup_ts,
        'mbo_duplicate_key_rows'      : mbo_dup_key,
        'mbp_duplicate_ts_rows'       : mbp_dup_ts,
        'mbp_duplicate_key_rows'      : mbp_dup_key,
        'label_distribution'          : {str(k): int(v) for k, v in label_dist.items()},
        'event_rate_tp_wins'          : float(event_strict),
        'train_event_pool_rate'       : float(train_pool),
        'train_event_count'           : n_train_events,
        'bias_long_short_counts'      : {'LONG': nc0, 'SHORT': nc1},
        'long_short_imbalance'        : round(imb, 3),
        'regime_train_event_counts'   : regime_ev_counts,
        'neutral_reason_counts'       : neutral_counts,
        'event_direction_counts'      : event_direction_counts,
        'catboost_surface_n'          : len(CATBOOST_ADVISOR_FEATURES_DT),
        'catboost_advisor_features'   : CATBOOST_ADVISOR_FEATURES_DT,
        'day_trading_context_features': DAY_TRADING_FEATURES,
        'lob_tensors_built'           : bool(build_lob_tensors),
        'lob_tensor_normalization'    : 'raw_unscaled_train_pipeline_must_fit_normalizer_on_train_only',
        'lob_roll_lookback_bars'      : DEEPLOB_TIME_STEPS_DEFAULT,
        'lob_tensor_layout'           : 'rolling_bars_time_x_20_levels_x_3ch_peak_mbp_when_available',
        'mbp_bar_coverage_stats'      : mbp_bar_cov_stats,
        'mbp_roll_lob_coverage_stats' : mbp_roll_cov_stats,
        'lob_imbalance_source'        : 'mbp_depth_when_available_else_trade_flow_proxy',
        'lob_imbalance_depth_row_rate': lob_depth_ratio,
        'lob_imbalance_vs_order_flow_spearman': lob_flow_corr,
        'ts_min'                      : str(df_out['ts_event'].min()),
        'ts_max'                      : str(df_out['ts_event'].max()),
        'label_logic'                 : (
            'Event Gate (hawkes+absorption+kyle+cvd zscore) → '
            'Kalman Trend Filter (weak counter-trend veto) → '
            'First Barrier Hit (Regime-Aware TP/SL + Session Timeout) → '
            'SL outcomes → NEUTRAL'
        ),
    }
    with open(os.path.join(output_dir, 'day_trading_manifest.json'), 'w', encoding='utf-8') as f:
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
        '--event_threshold_scale',
        type=float,
        default=1.0,
        help='scale لعَتبات Event Gate (أقل من 1.0 = إشارات أكثر، default=1.0)',
    )
    p.add_argument(
        '--event_threshold_shift',
        type=float,
        default=0.0,
        help='إزاحة لعَتبات Event Gate بعد الـ scale (قيمة سالبة = إشارات أكثر، default=0.0)',
    )
    p.add_argument(
        '--kalman_event_floor',
        type=float,
        default=float(_KALMAN_EVENT_FLOOR),
        help='عتبة event_score قبل Kalman counter-trend veto (أقل = veto أقل، default=0.70)',
    )
    p.add_argument(
        '--weak_event_to_directional',
        action='store_true',
        help='حوّل weak-event rows إلى LONG/SHORT إذا حركة الأفق تخطت حد ATR (تقليل قوي للـ NEUTRAL).',
    )
    p.add_argument(
        '--weak_event_min_move_atr',
        type=float,
        default=0.35,
        help='الحد الأدنى لحركة weak-event (بوحدة ATR) قبل تحويلها لاتجاهي، default=0.35',
    )
    p.add_argument(
        '--sl_to_opposite',
        action='store_true',
        help='حوّل long_sl→SHORT و short_sl→LONG بدل NEUTRAL (وضع aggressive).',
    )
    p.add_argument(
        '--include_weak_directional_in_train',
        action='store_true',
        help='أدخل الاتجاهات الناتجة من weak-event ضمن train_event_flag.',
    )
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
        event_threshold_scale=args.event_threshold_scale,
        event_threshold_shift=args.event_threshold_shift,
        kalman_event_floor=args.kalman_event_floor,
        weak_event_to_directional=args.weak_event_to_directional,
        weak_event_min_move_atr=args.weak_event_min_move_atr,
        sl_to_opposite=args.sl_to_opposite,
        include_weak_directional_in_train=args.include_weak_directional_in_train,
    )
