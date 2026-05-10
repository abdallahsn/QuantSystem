"""
regime_config.py — المرجع المركزي لإعدادات كل Regime
═══════════════════════════════════════════════════════
QuantSystem V19 | Section 9 من المرجع التقني

يُستورد من:
    prepare_day_trading.py  → REGIME_TP_SL, REGIME_MAX_BARS, REGIME_EVENT_THRESHOLD
    initial_training_6_years.py → REGIME_MODEL_CONFIGS, REGIME_EXTRA_FEATURES
    modules/online_learning.py  → REGIME_DRIFT_CONFIG, REGIME_PRED_THRESHOLD
    modules/live_predictor.py   → REGIME_EXTRA_FEATURES, REGIME_PRED_THRESHOLD

قاعدة: لا تُعرِّف هذه الثوابت في أي ملف آخر — استورد دائماً من هنا.
"""

from __future__ import annotations

# ── تعريف الـ Regimes المدعومة ────────────────────────────────────────────────
REGIMES: tuple[str, ...] = ('trending', 'ranging', 'volatile')

# ──────────────────────────────────────────────────────────────────────────────
# 1. Event Detection — عتبة كشف الأحداث الحقيقية
# ──────────────────────────────────────────────────────────────────────────────
# Volatile: عتبة أعلى — فقط إشارات قوية جداً تمر (السوق فوضوي والإشارات الضعيفة مضللة)
REGIME_EVENT_THRESHOLD: dict[str, float] = {
    'trending': 0.60,
    'ranging' : 0.60,
    'volatile': 0.75,
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. Labels: TP/SL Multipliers
# ──────────────────────────────────────────────────────────────────────────────
# (tp_mult, sl_mult) — مضروبان في ATR
# Trending:  حركة طويلة → TP = 2× ATR كافٍ، SL = 1× ATR عادي
# Ranging:   حركة ضيقة وسريعة العكس → TP/SL صغيران لتحقيق نسبة ربح
# Volatile:  حركة حادة وخطيرة → SL مساوٍ للـ TP للحماية من الانعكاس المباغت
REGIME_TP_SL: dict[str, tuple[float, float]] = {
    'trending': (2.0, 1.0),
    'ranging' : (0.8, 0.6),
    'volatile': (1.5, 1.5),
}

# ──────────────────────────────────────────────────────────────────────────────
# 3. Labels: Max Horizon (بالـ bars)
# ──────────────────────────────────────────────────────────────────────────────
# كل bar = 5 دقائق (افتراضياً)
# Trending:  الحركة تستمر طويلاً → 12 bar = 60 دقيقة
# Ranging:   الحركة تعكس بسرعة → 6 bars = 30 دقيقة
# Volatile:  الحركة تنتهي بسرعة → 3 bars = 15 دقيقة
REGIME_MAX_BARS: dict[str, int] = {
    'trending': 12,
    'ranging' :  6,
    'volatile':  3,
}

# ──────────────────────────────────────────────────────────────────────────────
# 4. Training: CatBoost Model Config
# ──────────────────────────────────────────────────────────────────────────────
# Trending:  عمق أوسط + iterations كثيرة → يتعلم تسلسل اتجاهي طويل
# Ranging:   عمق أكبر → يتعلم أنماط انعكاس دقيقة
# Volatile:  عمق ضحل + learning_rate أسرع → يتعلم بسرعة ولا يُفرط في التخصيص
REGIME_MODEL_CONFIGS: dict[str, dict] = {
    'trending': {'iterations': 800,  'learning_rate': 0.02,  'depth': 7},
    'ranging' : {'iterations': 1000, 'learning_rate': 0.015, 'depth': 8},
    'volatile': {'iterations': 500,  'learning_rate': 0.05,  'depth': 5},
}

# ──────────────────────────────────────────────────────────────────────────────
# 5. Training: Features الإضافية لكل Regime
# ──────────────────────────────────────────────────────────────────────────────
# تُضاف فوق BASE_FEATURES المشتركة
# Trending:  يحتاج لقياس التسارع والزخم الاتجاهي
# Ranging:   يحتاج لقياس الاضطراب والانعكاسات وعمق السيولة
# Volatile:  يحتاج للحظات الذروة والضغط الشديد
REGIME_EXTRA_FEATURES: dict[str, list[str]] = {
    'trending': [
        'cvd_velocity',           # تسارع الضغط الاتجاهي
        'hawkes_intensity',       # كثافة وصول الأوردرات
        'kyle_lambda',            # استجابة السعر للـ flow
        'cvd_direction_pct',      # نسبة الـ slices في اتجاه واحد
    ],
    'ranging': [
        'absorption_std',         # تقلب صانع السوق
        'imb_reversals',          # عدد انعكاسات الـ OBI
        'cancel_std',             # تقلب الإلغاءات
        'mbp_bid_slope_intrabar', # ميل عمق الـ bid
        'mbp_ask_slope_intrabar', # ميل عمق الـ ask
    ],
    'volatile': [
        'kyle_lambda',            # تأثير flow (max يُحسب في التجميع)
        'cancel_volume_ratio',    # نسبة السبوفينغ الكلي
        'micro_atr_max',          # أعلى تقلب intrabar
        'absorption_intensity',   # شدة الامتصاص
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
# 6. Online Learning: Drift Detection
# ──────────────────────────────────────────────────────────────────────────────
# Volatile: delta أكبر + lambda أصغر = يكتشف drift بسرعة أكبر
#           لأن السوق الـ volatile يتغير أسرع من الـ trending/ranging
REGIME_DRIFT_CONFIG: dict[str, dict] = {
    'trending': {'delta': 0.005, 'lambda_': 50},
    'ranging' : {'delta': 0.005, 'lambda_': 50},
    'volatile': {'delta': 0.008, 'lambda_': 30},
}

# ──────────────────────────────────────────────────────────────────────────────
# 7. Live Prediction: Confidence Threshold
# ──────────────────────────────────────────────────────────────────────────────
# Volatile: ثقة أعلى مطلوبة لأن الإشارات الخاطئة في volatile أكثر تكلفةً
REGIME_PRED_THRESHOLD: dict[str, float] = {
    'trending': 0.60,
    'ranging' : 0.60,
    'volatile': 0.65,
}

# ──────────────────────────────────────────────────────────────────────────────
# 8. Event Score Weights — أوزان مكونات event_score
# ──────────────────────────────────────────────────────────────────────────────
# المجموع = 1.0
# hawkes + absorption هما الأقوى (0.30 + 0.30 = 0.60) → informed flow
# kyle يعزز (0.20) → استجابة السعر
# cvd_align يؤكد (0.20) → اتساق الاتجاه
EVENT_SCORE_WEIGHTS: dict[str, float] = {
    'hawkes_z_above_1'  : 0.30,
    'absorb_z_above_1'  : 0.30,
    'kyle_z_above_05'   : 0.20,
    'cvd_align_above_06': 0.20,
}

# Z-score window للـ rolling baseline
EVENT_ZSCORE_WINDOW: int = 100
EVENT_ZSCORE_MIN_PERIODS: int = 20

# ──────────────────────────────────────────────────────────────────────────────
# جدول ملخص سريع للمراجعة
# ──────────────────────────────────────────────────────────────────────────────
def print_regime_summary() -> None:
    """يطبع جدول مقارنة الإعدادات — للتشخيص فقط."""
    print("\n" + "═" * 70)
    print("  Regime Configuration Summary — QuantSystem V19")
    print("═" * 70)
    header = f"  {'Setting':<28} {'Trending':>12} {'Ranging':>12} {'Volatile':>12}"
    print(header)
    print("─" * 70)
    rows = [
        ("Event threshold",
         str(REGIME_EVENT_THRESHOLD['trending']),
         str(REGIME_EVENT_THRESHOLD['ranging']),
         str(REGIME_EVENT_THRESHOLD['volatile'])),
        ("TP multiplier",
         f"{REGIME_TP_SL['trending'][0]}× ATR",
         f"{REGIME_TP_SL['ranging'][0]}× ATR",
         f"{REGIME_TP_SL['volatile'][0]}× ATR"),
        ("SL multiplier",
         f"{REGIME_TP_SL['trending'][1]}× ATR",
         f"{REGIME_TP_SL['ranging'][1]}× ATR",
         f"{REGIME_TP_SL['volatile'][1]}× ATR"),
        ("Max horizon (bars)",
         str(REGIME_MAX_BARS['trending']),
         str(REGIME_MAX_BARS['ranging']),
         str(REGIME_MAX_BARS['volatile'])),
        ("CatBoost depth",
         str(REGIME_MODEL_CONFIGS['trending']['depth']),
         str(REGIME_MODEL_CONFIGS['ranging']['depth']),
         str(REGIME_MODEL_CONFIGS['volatile']['depth'])),
        ("CatBoost iterations",
         str(REGIME_MODEL_CONFIGS['trending']['iterations']),
         str(REGIME_MODEL_CONFIGS['ranging']['iterations']),
         str(REGIME_MODEL_CONFIGS['volatile']['iterations'])),
        ("Learning rate",
         str(REGIME_MODEL_CONFIGS['trending']['learning_rate']),
         str(REGIME_MODEL_CONFIGS['ranging']['learning_rate']),
         str(REGIME_MODEL_CONFIGS['volatile']['learning_rate'])),
        ("Drift lambda_",
         str(REGIME_DRIFT_CONFIG['trending']['lambda_']),
         str(REGIME_DRIFT_CONFIG['ranging']['lambda_']),
         str(REGIME_DRIFT_CONFIG['volatile']['lambda_'])),
        ("Pred threshold",
         str(REGIME_PRED_THRESHOLD['trending']),
         str(REGIME_PRED_THRESHOLD['ranging']),
         str(REGIME_PRED_THRESHOLD['volatile'])),
    ]
    for label, t, r, v in rows:
        print(f"  {label:<28} {t:>12} {r:>12} {v:>12}")
    print("═" * 70 + "\n")


if __name__ == '__main__':
    print_regime_summary()
