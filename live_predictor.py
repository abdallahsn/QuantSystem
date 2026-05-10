"""
modules/live_predictor.py — Live Prediction Pipeline
══════════════════════════════════════════════════════
QuantSystem V19 | Section 10 من المرجع التقني

كل bar يمر بـ 5 خطوات قبل أي قرار:
    1. Event Gate     — هل هذا bar يستحق التقييم؟ (event_score >= threshold)
    2. Regime Gate    — هل يوجد موديل لهذا الـ regime؟
    3. Feature Select — اختيار الـ features المناسبة للـ regime
    4. Predict        — استخراج الاحتمالية
    5. Confidence     — هل تتجاوز العتبة المخصصة للـ regime؟

الاستخدام في التداول الحي:
    from modules.live_predictor import run_bar_pipeline

    signal, confidence, debug = run_bar_pipeline(
        df_bar_row = bar,          # pd.Series لـ bar مكتمل
        models     = regime_models,
        ensemble   = online_ensemble,  # اختياري
    )
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

try:
    from modules.feature_factory_v19 import apply_scaler_params_to_frame
except ImportError:
    apply_scaler_params_to_frame = None

try:
    from regime_config import REGIME_EXTRA_FEATURES, REGIME_PRED_THRESHOLD, REGIME_EVENT_THRESHOLD
except ImportError:
    REGIME_EXTRA_FEATURES = {
        'trending': ['cvd_velocity', 'hawkes_intensity', 'kyle_lambda', 'cvd_direction_pct'],
        'ranging' : ['absorption_std', 'imb_reversals', 'cancel_std',
                     'mbp_bid_slope_intrabar', 'mbp_ask_slope_intrabar'],
        'volatile': ['kyle_lambda', 'cancel_volume_ratio', 'micro_atr_max', 'absorption_intensity'],
    }
    REGIME_PRED_THRESHOLD  = {'trending': 0.60, 'ranging': 0.60, 'volatile': 0.65}
    REGIME_EVENT_THRESHOLD = {'trending': 0.60, 'ranging': 0.60, 'volatile': 0.75}

# ── Base Features — تُوحَّد مع train_v19.py FEATURE_COLS ──────────────────────
# يجب أن تُطابق ما استُخدم في التدريب الأولي
BASE_FEATURES: list[str] = [
    'cvd', 'obi', 'absorption_intensity', 'cancel_ratio',
    'spoofing_ratio', 'spoofing_duration', 'liquidity_trap',
    'micro_atr', 'volume_burst', 'inter_event_time',
    'micro_price', 'bid_wall_strength', 'ask_wall_strength',
    'distance_to_wall', 'gap_size', 'liquidity_density',
    'fisher_signal', 'anomaly',
    'cvd_momentum', 'cvd_price_divergence',
    'trend_strength', 'correction_depth', 'liquidity_sweep',
    'kyle_lambda', 'hawkes_intensity', 'vnet', 'vwap_z_score',
    'atr_14', 'rsi_14', 'bar_range', 'body_ratio',
    'buy_ratio', 'bar_cvd_delta', 'lob_imbalance',
    'cvd_velocity', 'imb_reversals', 'cancel_volume_ratio',
    'absorption_std', 'cancel_std', 'kyle_std', 'micro_atr_max',
]

_REGIMES = ('trending', 'ranging', 'volatile')
_SIGNAL_NEUTRAL = 'NEUTRAL'
_SIGNAL_LONG    = 'LONG'
_SIGNAL_SHORT   = 'SHORT'


# ══════════════════════════════════════════════════════════════════════════════
# Feature Selector
# ══════════════════════════════════════════════════════════════════════════════

def get_regime_features(regime: str, available_cols: list[str]) -> list[str]:
    """
    يرجع قائمة features مناسبة للـ regime، مع فلترة ما هو موجود فعلاً في الـ bar.

    = BASE_FEATURES + REGIME_EXTRA_FEATURES[regime]
    """
    wanted = list(BASE_FEATURES) + REGIME_EXTRA_FEATURES.get(regime, [])
    # deduplicate مع الحفاظ على الترتيب
    seen: set[str] = set()
    ordered: list[str] = []
    for f in wanted:
        if f not in seen:
            seen.add(f)
            ordered.append(f)
    return [f for f in ordered if f in available_cols]


def load_live_feature_contract(models_dir: str) -> tuple[dict, dict]:
    """Load the exact training feature schema and inference scaler."""
    schema_path = os.path.join(str(models_dir), 'feature_schema_v19.json')
    scaler_path = os.path.join(str(models_dir), 'scaler_params.json')
    with open(schema_path, 'r', encoding='utf-8') as f:
        feature_schema = json.load(f)
    scaler_params: dict = {}
    if os.path.exists(scaler_path):
        with open(scaler_path, 'r', encoding='utf-8') as f:
            scaler_params = json.load(f)
    return feature_schema, scaler_params


def build_live_vector(
    bar: pd.Series,
    feature_schema: dict,
    scaler_params: dict | None = None,
    *,
    already_scaled: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Build a production inference vector from the saved training schema."""
    features = list((feature_schema or {}).get('stat_features') or [])
    if not features:
        raise ValueError('Live feature schema is missing stat_features.')
    missing = [c for c in features if c not in bar.index]
    if missing:
        raise ValueError(f'Live bar missing trained features: {missing[:10]}')

    frame = pd.DataFrame([{c: bar[c] for c in features}])
    for c in features:
        frame[c] = pd.to_numeric(frame[c], errors='coerce')
    if not already_scaled and scaler_params:
        if apply_scaler_params_to_frame is None:
            raise RuntimeError('Scaler params supplied but modules.feature_factory_v19 is unavailable.')
        frame = apply_scaler_params_to_frame(frame, scaler_params)
    x = frame[features].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.reshape(1, -1), features


# ══════════════════════════════════════════════════════════════════════════════
# Core Prediction Function
# ══════════════════════════════════════════════════════════════════════════════

def predict_live(
    bar_features   : pd.Series,
    current_regime : str,
    event_score    : float,
    models         : dict,
    ensemble       : object | None = None,
    feature_schema : dict | None = None,
    scaler_params  : dict | None = None,
    already_scaled : bool = False,
    allow_dynamic_features: bool = False,
) -> tuple[str, float, dict]:
    """
    Pipeline كامل للتنبؤ الحي — 5 خطوات.

    Parameters
    ----------
    bar_features   : pd.Series  — كل features الـ bar الحالي
    current_regime : str        — 'trending' / 'ranging' / 'volatile'
    event_score    : float      — من detect_microstructure_events [0,1]
    models         : dict       — {regime: CatBoostClassifier}
    ensemble       : RegimeConditionalEnsemble | None — Online Learning (اختياري)

    Returns
    -------
    (signal, confidence, debug_info)
    signal     : 'LONG' | 'SHORT' | 'NEUTRAL'
    confidence : احتمالية الـ signal (0–1)
    debug_info : dict للتشخيص والـ logging
    """
    debug: dict = {
        'regime'      : current_regime,
        'event_score' : round(float(event_score), 4),
        'step_blocked': None,
    }

    # ── Step 1: Event Gate ──────────────────────────────────────────────────
    threshold = REGIME_EVENT_THRESHOLD.get(current_regime, 0.60)
    debug['event_threshold'] = threshold

    if float(event_score) < threshold:
        debug['step_blocked'] = 'event_gate'
        return _SIGNAL_NEUTRAL, 0.0, debug

    # ── Step 2: Regime Model موجود؟ ────────────────────────────────────────
    effective_models = {
        k: v for k, v in dict(models or {}).items()
        if k not in {'feature_schema', 'scaler_params'}
    }
    if feature_schema is None and isinstance(models, dict):
        maybe_schema = models.get('feature_schema')
        if isinstance(maybe_schema, dict):
            feature_schema = maybe_schema
    if scaler_params is None and isinstance(models, dict):
        maybe_scaler = models.get('scaler_params')
        if isinstance(maybe_scaler, dict):
            scaler_params = maybe_scaler
    if ensemble is not None:
        # Ensemble يُكمِّل الـ models الأساسية
        for r in _REGIMES:
            if r not in effective_models and r in getattr(ensemble, 'models', {}):
                effective_models[r] = ensemble.models[r]

    if current_regime not in effective_models:
        debug['step_blocked'] = f'no_model_for_regime:{current_regime}'
        return _SIGNAL_NEUTRAL, 0.0, debug

    # ── Step 3: Feature Contract ────────────────────────────────────────────
    try:
        if feature_schema is not None:
            X, feats = build_live_vector(
                bar_features,
                feature_schema,
                scaler_params,
                already_scaled=already_scaled,
            )
            debug['schema_contract'] = 'feature_schema_v19'
        elif allow_dynamic_features:
            available_cols = list(bar_features.index)
            feats = get_regime_features(current_regime, available_cols)
            if not feats:
                debug['step_blocked'] = 'no_features_available'
                return _SIGNAL_NEUTRAL, 0.0, debug
            X = bar_features[feats].to_numpy(dtype=np.float32).reshape(1, -1)
            X = np.where(np.isfinite(X), X, 0.0)
            debug['schema_contract'] = 'legacy_dynamic_features'
        else:
            debug['step_blocked'] = 'missing_feature_schema'
            return _SIGNAL_NEUTRAL, 0.0, debug
    except Exception as e:
        debug['step_blocked'] = f'feature_contract_error:{e}'
        return _SIGNAL_NEUTRAL, 0.0, debug
    debug['n_features'] = len(feats)

    # ── Step 4: Predict ─────────────────────────────────────────────────────
    try:
        proba = effective_models[current_regime].predict_proba(X)[0]
    except Exception as e:
        debug['step_blocked'] = f'predict_error:{e}'
        return _SIGNAL_NEUTRAL, 0.0, debug

    # نتوقع [P(LONG), P(SHORT)] — تأكد من الحجم
    if len(proba) >= 2:
        long_p  = float(proba[0])
        short_p = float(proba[1])
    elif len(proba) == 1:
        long_p, short_p = float(proba[0]), 1.0 - float(proba[0])
    else:
        debug['step_blocked'] = 'unexpected_proba_shape'
        return _SIGNAL_NEUTRAL, 0.0, debug

    debug['long_prob']  = round(long_p,  4)
    debug['short_prob'] = round(short_p, 4)

    # ── Step 5: Confidence Threshold (Regime-Aware) ─────────────────────────
    min_conf = REGIME_PRED_THRESHOLD.get(current_regime, 0.60)
    debug['min_confidence'] = min_conf

    if long_p >= min_conf and long_p > short_p:
        debug['features_used'] = feats
        return _SIGNAL_LONG, long_p, debug

    if short_p >= min_conf and short_p > long_p:
        debug['features_used'] = feats
        return _SIGNAL_SHORT, short_p, debug

    debug['step_blocked'] = 'below_confidence_threshold'
    return _SIGNAL_NEUTRAL, float(max(long_p, short_p)), debug


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point — يُستدعى لكل bar في التداول الحي
# ══════════════════════════════════════════════════════════════════════════════

def run_bar_pipeline(
    df_bar_row: pd.Series,
    models    : dict,
    ensemble  : object | None = None,
    *,
    feature_schema: dict | None = None,
    scaler_params: dict | None = None,
    models_dir: str | None = None,
    already_scaled: bool = False,
    allow_dynamic_features: bool = False,
) -> tuple[str, float, dict]:
    """
    Entry point رئيسي — يُستدعى لكل bar جديد في التداول الحي.

    يستخرج تلقائياً:
        - regime       من 'regime_label'
        - event_score  من 'event_score'

    ويُشغِّل pipeline التنبؤ الكامل في دالة واحدة.

    Example
    -------
    >>> for bar in live_feed:
    ...     signal, conf, dbg = run_bar_pipeline(bar, models, ensemble)
    ...     if signal == 'LONG' and conf >= 0.62:
    ...         execute_long(bar['close'])
    """
    regime      = str(df_bar_row.get('regime_label', 'ranging'))
    event_score = float(df_bar_row.get('event_score', 0.0))
    if models_dir is not None and (feature_schema is None or scaler_params is None):
        loaded_schema, loaded_scaler = load_live_feature_contract(models_dir)
        feature_schema = loaded_schema if feature_schema is None else feature_schema
        scaler_params = loaded_scaler if scaler_params is None else scaler_params

    return predict_live(
        bar_features   = df_bar_row,
        current_regime = regime,
        event_score    = event_score,
        models         = models,
        ensemble       = ensemble,
        feature_schema = feature_schema,
        scaler_params  = scaler_params,
        already_scaled = already_scaled,
        allow_dynamic_features = allow_dynamic_features,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Batch Evaluation (للتشخيص وفحص الـ pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_on_test_set(
    df_test: pd.DataFrame,
    models : dict,
    ensemble: object | None = None,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    يُشغِّل run_bar_pipeline على كل صف في df_test ويرجع DataFrame بالنتائج.

    يُستخدم للتحقق من توزيع الإشارات (هدف: NEUTRAL < 70%).
    """
    rows = []
    for _, row in df_test.iterrows():
        signal, conf, dbg = run_bar_pipeline(row, models, ensemble)
        rows.append({
            'ts_event'    : row.get('ts_event'),
            'regime'      : dbg.get('regime'),
            'event_score' : dbg.get('event_score'),
            'signal'      : signal,
            'confidence'  : conf,
            'step_blocked': dbg.get('step_blocked'),
        })

    df_res = pd.DataFrame(rows)

    if verbose and len(df_res):
        print("\n📊 Live Pipeline — توزيع الإشارات:")
        counts = df_res['signal'].value_counts(normalize=True)
        for sig, pct in counts.items():
            icon = '✅' if sig == 'NEUTRAL' and pct < 0.70 else ('⚠️' if sig == 'NEUTRAL' else '🔵')
            print(f"   {icon} {sig}: {pct:.1%}")

        if 'regime' in df_res.columns:
            print("\n   По Regime:")
            for reg in _REGIMES:
                sub = df_res[df_res['regime'] == reg]
                if len(sub):
                    n_sig = int((sub['signal'] != 'NEUTRAL').sum())
                    print(f"     {reg:8s}: {len(sub):,} bars → {n_sig:,} signals ({n_sig/len(sub):.1%})")

        n_neutral = int((df_res['signal'] == 'NEUTRAL').sum())
        pct_neutral = n_neutral / max(len(df_res), 1)
        if pct_neutral > 0.70:
            print(f"   ⚠️ NEUTRAL = {pct_neutral:.1%} > 70% — راجع event_threshold أو بيانات الـ regime")

    return df_res
