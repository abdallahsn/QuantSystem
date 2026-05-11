"""
modules/online_learning.py — نظام Online Learning مع Drift Detection
══════════════════════════════════════════════════════════════════════
QuantSystem V19 | Section 8 من المرجع التقني

المكونات الثلاثة:
    1. DriftDetector     — يكشف تغيُّر توزيع السوق (Page-Hinkley)
    2. SlidingWindowTrainer — إعادة تدريب أسبوعية على نافذة متحركة
    3. RegimeConditionalEnsemble — كل regime = موديل + detector + buffer مستقل

الاستخدام في التداول الحي:
    from modules.online_learning import RegimeConditionalEnsemble

    ensemble = RegimeConditionalEnsemble()
    # عند كل إشارة جديدة:
    ensemble.update(features, label, regime, prediction_error)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

try:
    from catboost import CatBoostClassifier
    _CATBOOST_OK = True
except ImportError:
    _CATBOOST_OK = False
    CatBoostClassifier = None  # type: ignore

try:
    from regime_config import REGIME_DRIFT_CONFIG, REGIME_PRED_THRESHOLD
except ImportError:
    REGIME_DRIFT_CONFIG = {
        'trending': {'delta': 0.005, 'lambda_': 50},
        'ranging' : {'delta': 0.005, 'lambda_': 50},
        'volatile': {'delta': 0.008, 'lambda_': 30},
    }
    REGIME_PRED_THRESHOLD = {'trending': 0.60, 'ranging': 0.60, 'volatile': 0.65}

_REGIMES: tuple[str, ...] = ('trending', 'ranging', 'volatile')


# ══════════════════════════════════════════════════════════════════════════════
# 1. Drift Detector — Page-Hinkley Test
# ══════════════════════════════════════════════════════════════════════════════

class DriftDetector:
    """
    يكشف متى يتغير توزيع أخطاء التنبؤ بشكل دائم (Concept Drift).

    خوارزمية Page-Hinkley:
        PH_t = Σ(x_i - μ_t - δ)
        drift إذا: PH_t - min(PH) > λ

    Parameters
    ----------
    delta    : حساسية لتغيرات صغيرة (0.005 = يتجاهل ضجيج < 0.5%)
    lambda_  : عتبة الـ drift (50 = يحتاج تراكماً كبيراً قبل الإعلان)

    Volatile: delta=0.008, lambda_=30 → يكتشف drift أسرع
    """

    def __init__(
        self,
        delta: float = 0.005,
        lambda_: float = 50.0,
    ) -> None:
        self.delta    = float(delta)
        self.lambda_  = float(lambda_)
        self._sum     = 0.0
        self._min_sum = 0.0
        self._mean    = 0.0
        self._n       = 0
        self.drift_count = 0

    def update(self, error: float) -> bool:
        """
        أدخل prediction error (0-1)، ارجع True عند اكتشاف drift.
        بعد الـ drift → reset تلقائي.
        """
        self._n    += 1
        self._mean += (error - self._mean) / self._n
        self._sum  += error - self._mean - self.delta
        self._min_sum = min(self._min_sum, self._sum)

        drift = (self._sum - self._min_sum) > self.lambda_
        if drift:
            self.drift_count += 1
            self.reset()
        return drift

    def reset(self) -> None:
        self._sum = self._min_sum = self._mean = 0.0
        self._n = 0

    @property
    def statistic(self) -> float:
        """قيمة الـ statistic الحالية — تزداد عند تدهور الأداء."""
        return self._sum - self._min_sum


# ══════════════════════════════════════════════════════════════════════════════
# 2. Sliding Window Trainer
# ══════════════════════════════════════════════════════════════════════════════

class SlidingWindowTrainer:
    """
    إعادة تدريب أسبوعية على نافذة متحركة من أحدث البيانات.

    الفلسفة:
        - وزن زمني: أحدث بيانات → أهم
        - عند Drift: يضيّق النافذة على آخر 60% (يُهمل البيانات القديمة)

    Parameters
    ----------
    window_days      : حجم النافذة (افتراضي 30 يوم)
    retrain_every_days: تكرار التدريب (افتراضي 7 أيام)
    """

    def __init__(
        self,
        window_days: int = 30,
        retrain_every_days: int = 7,
    ) -> None:
        self.window_days        = int(window_days)
        self.retrain_every_days = int(retrain_every_days)
        self.last_retrain: pd.Timestamp | None = None
        self.model: Any = None
        self._retrain_count = 0

    def should_retrain(self, current_date: pd.Timestamp) -> bool:
        if self.last_retrain is None:
            return True
        return (current_date - self.last_retrain).days >= self.retrain_every_days

    def build_sample_weights(
        self,
        df: pd.DataFrame,
        half_life_days: float = 7.0,
    ) -> np.ndarray:
        """أوزان تناسبية: أحدث بيانات = وزن أعلى (half-life تحكم التناقص)."""
        now  = df['ts_event'].max()
        ages = (now - df['ts_event']).dt.days.values.astype(float)
        ages = np.maximum(ages, 0.0)
        decay = np.exp(-np.log(2) * ages / max(half_life_days, 1.0))
        total = decay.sum()
        return decay / total * len(decay) if total > 1e-9 else np.ones(len(decay))

    def retrain(
        self,
        df_new: pd.DataFrame,
        feature_cols: list[str],
        drift_detected: bool = False,
    ) -> bool:
        """
        يُعيد التدريب على events الـ DataFrame المُمرَّر.

        Returns True لو التدريب نجح، False لو بيانات قليلة.
        """
        if not _CATBOOST_OK:
            return False

        if drift_detected:
            # Drift → ركز على آخر 60% من البيانات
            df_new = df_new.tail(max(int(len(df_new) * 0.60), 10)).copy()

        # فلتر events حقيقية فقط
        mask = (df_new.get('is_event', pd.Series(1)) == 1) & \
               (df_new.get('bias_label', pd.Series(0)) != 2)
        df_ev = df_new[mask].copy()
        if len(df_ev) < 50:
            return False

        weights  = self.build_sample_weights(df_ev)
        feats    = [f for f in feature_cols if f in df_ev.columns]
        if not feats:
            return False

        self.model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.03,
            depth=6,
            verbose=0,
            allow_writing_files=False,
        ).fit(
            X=df_ev[feats],
            y=df_ev['bias_label'],
            sample_weight=weights,
        )

        self.last_retrain = df_new['ts_event'].max()
        self._retrain_count += 1
        return True

    def predict(self, X: np.ndarray) -> np.ndarray | None:
        """يرجع احتمالية [LONG, SHORT] أو None لو لا موديل."""
        if self.model is None:
            return None
        return self.model.predict_proba(X.reshape(1, -1))[0]


# ══════════════════════════════════════════════════════════════════════════════
# 3. Regime-Conditional Ensemble (Online)
# ══════════════════════════════════════════════════════════════════════════════

class RegimeConditionalEnsemble:
    """
    كل regime له موديل + drift detector + buffer مستقل.

    التصميم:
        - 3 موديلات CatBoost (trending / ranging / volatile)
        - 3 DriftDetectors مضبوطة من REGIME_DRIFT_CONFIG
        - Buffer: يحتفظ بآخر 500 عينة لكل regime

    Buffer auto-retrain:
        - لما buffer يصل 200 عينة → يُعيد التدريب
        - لما drift يُكتشف → يُعيد التدريب فوراً على آخر 50%

    الاستخدام:
        ensemble = RegimeConditionalEnsemble()
        proba = ensemble.predict(bar_features, regime='trending')
        ensemble.update(features, label=0, regime='trending', error=0.35)
    """

    _MIN_SAMPLES_RETRAIN = 200    # retrain عند هذا الحجم
    _BUFFER_MAX          = 500    # احتفظ بآخر N عينة فقط
    _MIN_SAMPLES_INITIAL = 50     # أقل عدد يسمح بالتدريب

    def __init__(self) -> None:
        cfg = REGIME_DRIFT_CONFIG

        self.detectors: dict[str, DriftDetector] = {
            r: DriftDetector(**cfg.get(r, {'delta': 0.005, 'lambda_': 50}))
            for r in _REGIMES
        }
        self.models: dict[str, Any] = {}
        self.buffers: dict[str, list[dict]] = {r: [] for r in _REGIMES}
        self._retrain_counts: dict[str, int] = {r: 0 for r in _REGIMES}

    # ── Predict ─────────────────────────────────────────────────────────────

    def predict(
        self,
        features: np.ndarray,
        regime: str,
    ) -> np.ndarray:
        """
        يرجع [P(LONG), P(SHORT)] للـ bar الحالي.
        إذا لا يوجد موديل للـ regime → [0.5, 0.5]
        """
        if regime not in self.models:
            return np.array([0.5, 0.5], dtype=np.float32)
        try:
            return self.models[regime].predict_proba(
                features.reshape(1, -1)
            )[0].astype(np.float32)
        except Exception:
            return np.array([0.5, 0.5], dtype=np.float32)

    # ── Update ───────────────────────────────────────────────────────────────

    def update(
        self,
        features: np.ndarray,
        label: int,
        regime: str,
        error: float,
    ) -> None:
        """
        يُحدِّث الـ regime المحدد بعينة جديدة.

        Parameters
        ----------
        features : np.ndarray — feature vector للـ bar
        label    : int        — 0=LONG, 1=SHORT
        regime   : str        — 'trending' / 'ranging' / 'volatile'
        error    : float      — prediction error (0-1)، عادةً 1 - confidence
        """
        if regime not in _REGIMES:
            return

        # buffer
        self.buffers[regime].append({
            'features': np.asarray(features, dtype=np.float32),
            'label'   : int(label),
            'error'   : float(error),
        })

        # drift detection
        drift = self.detectors[regime].update(error)

        # retrain conditions
        should = (
            len(self.buffers[regime]) >= self._MIN_SAMPLES_RETRAIN
            or drift
        )
        if should:
            self._retrain_regime(regime, force_drift=drift)

    # ── Internal Retrain ─────────────────────────────────────────────────────

    def _retrain_regime(self, regime: str, *, force_drift: bool = False) -> None:
        """يُعيد تدريب موديل الـ regime على buffer الحالي."""
        if not _CATBOOST_OK:
            return

        buf = self.buffers[regime]
        if len(buf) < self._MIN_SAMPLES_INITIAL:
            return

        X = np.vstack([b['features'] for b in buf])
        y = np.array([b['label']    for b in buf], dtype=np.int32)

        if force_drift:
            # Drift: ركز على آخر 50% لتجاهل بيانات من الـ regime القديم
            keep = max(int(len(X) * 0.50), self._MIN_SAMPLES_INITIAL)
            X, y = X[-keep:], y[-keep:]

        try:
            from regime_config import REGIME_MODEL_CONFIGS
            cfg = REGIME_MODEL_CONFIGS.get(regime, {
                'iterations': 300, 'learning_rate': 0.03, 'depth': 6
            })
        except ImportError:
            cfg = {'iterations': 300, 'learning_rate': 0.03, 'depth': 6}

        try:
            self.models[regime] = CatBoostClassifier(
                **cfg,
                loss_function='MultiClass',
                verbose=0,
                allow_writing_files=False,
            ).fit(X, y)

            self._retrain_counts[regime] += 1

            # trim buffer: احتفظ بآخر N عينة
            self.buffers[regime] = buf[-self._BUFFER_MAX:]

        except Exception as e:
            print(f"  ⚠️ RegimeConditionalEnsemble: فشل تدريب {regime}: {e}")

    # ── Initialize from Pretrained Models ───────────────────────────────────

    def load_pretrained(self, models: dict[str, Any]) -> None:
        """يُحمِّل موديلات مُدرَّبة مسبقاً (من initial_training_6_years)."""
        for regime, model in models.items():
            if regime in _REGIMES:
                self.models[regime] = model

    # ── Status ───────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            'models_loaded' : list(self.models.keys()),
            'buffer_sizes'  : {r: len(b) for r, b in self.buffers.items()},
            'drift_counts'  : {r: d.drift_count for r, d in self.detectors.items()},
            'retrain_counts': dict(self._retrain_counts),
        }

    def __repr__(self) -> str:
        s = self.status()
        return (
            f"RegimeConditionalEnsemble("
            f"models={s['models_loaded']}, "
            f"buffers={s['buffer_sizes']}, "
            f"drifts={s['drift_counts']})"
        )
