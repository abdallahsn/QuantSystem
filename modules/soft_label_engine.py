"""
soft_label_engine.py — Soft Labels for V19 QuantSystem
=======================================================
يحل مشكلة الـ Hidden-State Label Noise:

  Hard Label:  نفس X → 0 أو 1 (يتناقض مع نفسه عندما تتغير الحالة الخفية)
  Soft Label:  نفس X → احتمالية ∈ [0, 1] (تعكس عدم اليقين الحقيقي)

الهيكل المعماري:
  build_causal_event_labels()
    ↓
  attach_soft_labels()          ← نقطة التكامل الرئيسية
    ↓
  labeled_df مع أعمدة جديدة:
    soft_label         : P(ربح | الاتجاه الحالي) ∈ [0, 1]
    label_confidence   : مدى اليقين في الـ soft_label ∈ [0, 1]
    soft_label_long    : P(ربح | اتجاه LONG)
    soft_label_short   : P(ربح | اتجاه SHORT)
    soft_sample_weight : وزن التدريب النهائي = confidence × clarity × quality

وضعان:
  ANALYTICAL (سريع، للإنتاج):
    يستخدم path_outcome + forward_return الموجودة بالفعل
    زمن التشغيل: O(n) — بدون forward scans إضافية
    
  MONTE_CARLO (بطيء، دقة عالية، للبحث):
    يُشغّل n_scenarios سيناريو بمعاملات مزعزعة لتقدير P(ربح)
    زمن التشغيل: O(n × n_scenarios) — استخدم offline فقط

ملاحظة علمية:
  الوضع الأول (ANALYTICAL) مناسب للإنتاج لأنه:
  - لا يُعيد الـ forward scan المكلف
  - يستخدم المعلومات المتاحة بشكل رياضياً سليم
  - يُنتج soft labels ذات معنى حقيقي بناءً على نظرية الاحتمالات
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

# ─── ثوابت مستوردة من labels_v19 ─────────────────────────────────────────────
# PATH_OUTCOME
PATH_LONG_TP_FIRST  = 0
PATH_SHORT_TP_FIRST = 1
PATH_LONG_SL_FIRST  = 2
PATH_SHORT_SL_FIRST = 3
PATH_TIMEOUT        = 4

# BIAS DIRECTION
DIR_LONG    = 0
DIR_SHORT   = 1
DIR_NEUTRAL = 2

# SIGNAL QUALITY
QUALITY_NONE   = 0
QUALITY_WEAK   = 1
QUALITY_STRONG = 2

# ─── مصفوفة نقطة البداية (Analytical) ────────────────────────────────────────
# soft_label_long: P(win | direction=LONG) بناءً على path_outcome + quality
# المنطق الرياضي:
#   - TP_FIRST     : TP تم الوصول إليه → فوز مؤكد، ولكن TP المزعزع قد لا يُصاب
#                    → P ≈ 0.78-0.90 بحسب الجودة
#   - SL_FIRST     : SL تم الضرب → خسارة واضحة
#                    → P ≈ 0.08-0.12
#   - SHORT_TP     : الاتجاه المعاكس فاز → هذه إشارة LONG ستكون خاسرة
#                    → P ≈ 0.10
#   - SHORT_SL     : SHORT ضُرب SL يعني السعر ارتفع → مفيد للـ LONG
#                    → P ≈ 0.68-0.72
#   - TIMEOUT      : غير محدد → sigmoid(forward_return / dynamic_threshold)
_BASE_SOFT_LONG: dict[int, dict[int, float]] = {
    PATH_LONG_TP_FIRST:  {QUALITY_WEAK: 0.78, QUALITY_STRONG: 0.90},
    PATH_LONG_SL_FIRST:  {QUALITY_WEAK: 0.10, QUALITY_STRONG: 0.08},
    PATH_SHORT_TP_FIRST: {QUALITY_WEAK: 0.10, QUALITY_STRONG: 0.10},
    PATH_SHORT_SL_FIRST: {QUALITY_WEAK: 0.68, QUALITY_STRONG: 0.72},
}

# soft_label_short: متماثل مع soft_label_long بعكس المسارات
_BASE_SOFT_SHORT: dict[int, dict[int, float]] = {
    PATH_SHORT_TP_FIRST: {QUALITY_WEAK: 0.78, QUALITY_STRONG: 0.90},
    PATH_SHORT_SL_FIRST: {QUALITY_WEAK: 0.10, QUALITY_STRONG: 0.08},
    PATH_LONG_TP_FIRST:  {QUALITY_WEAK: 0.10, QUALITY_STRONG: 0.10},
    PATH_LONG_SL_FIRST:  {QUALITY_WEAK: 0.68, QUALITY_STRONG: 0.72},
}


# ─── إعدادات المحرك ──────────────────────────────────────────────────────────

@dataclass
class SoftLabelConfig:
    """إعدادات محرك Soft Labels"""
    mode: Literal["analytical", "monte_carlo"] = "monte_carlo"
    # ── Monte Carlo (افتراضي — تمايز أفضل من analytical عند الغموض / الرينج) ──
    n_scenarios: int = 200
    horizon_std: float = 0.30     # تباين أفقي أوسع للسيناريوهات
    tp_std: float = 0.20          # تباين أهداف TP
    sl_std: float = 0.20          # تباين أهداف SL
    random_seed: int = 42
    # ── أوزان التدريب النهائية ───────────────────────────────────────────────
    strong_quality_boost: float = 1.30   # مضاعف الجودة القوية
    weak_quality_boost:   float = 1.00   # مضاعف الجودة الضعيفة
    min_sample_weight: float = 0.10      # حد أدنى للوزن
    confidence_weight_power: float = 1.0 # أس الثقة في الوزن (1 = خطي)
    # ── فلتر الثقة ──────────────────────────────────────────────────────────
    min_confidence_threshold: float = 0.30   # استثناء الصفوف ضعيفة الثقة جداً
    # ── sigmoid لـ TIMEOUT ──────────────────────────────────────────────────
    timeout_sigmoid_scale: float = 1.5  # انحدار sigmoid عند TIMEOUT
    timeout_soft_label_range: float = 0.40  # نطاق soft label عند TIMEOUT: 0.5 ± range


# ─── المحرك الرئيسي ──────────────────────────────────────────────────────────

class SoftLabelEngine:
    """
    محرك Soft Labels لنظام V19.
    
    يُنتج:
    - soft_label         : P(ربح | اتجاه bias_label)
    - soft_label_long    : P(ربح | LONG)
    - soft_label_short   : P(ربح | SHORT)
    - label_confidence   : يقين soft_label
    - soft_sample_weight : وزن التدريب النهائي
    """

    def __init__(self, config: Optional[SoftLabelConfig] = None):
        self.config = config or SoftLabelConfig()

    # ── واجهة رئيسية ─────────────────────────────────────────────────────────

    def attach_soft_labels(self, labeled_df: pd.DataFrame) -> pd.DataFrame:
        """
        أضف أعمدة Soft Labels إلى DataFrame المُصنَّف.
        
        يتوقع الأعمدة التالية من build_causal_event_labels():
          path_outcome, bias_label, signal_quality,
          forward_return, label_dynamic_threshold,
          effective_horizon, label_horizon_steps,
          timeout_move_exceeded_band
        
        يُرجع نسخة معدّلة من labeled_df.
        """
        df = labeled_df.copy()
        n = len(df)
        if n == 0:
            return df

        if self.config.mode == "monte_carlo":
            return self._attach_monte_carlo(df)
        else:
            return self._attach_analytical(df)

    # ── الوضع التحليلي (سريع) ────────────────────────────────────────────────

    def _attach_analytical(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        cfg = self.config

        path    = df["path_outcome"].to_numpy(dtype=np.int8, copy=False)
        bias    = df["bias_label"].to_numpy(dtype=np.int8, copy=False)
        quality = df["signal_quality"].to_numpy(dtype=np.int8, copy=False)
        fwd_ret = df["forward_return"].to_numpy(dtype=np.float32, copy=False)

        # dynamic_threshold: إما مخزون أو نُعيد بناؤه من micro_atr
        dyn_thr = self._resolve_dynamic_threshold(df, n)

        eff_hor = df["effective_horizon"].to_numpy(dtype=np.int32, copy=False)
        hor_steps = df["label_horizon_steps"].to_numpy(dtype=np.int32, copy=False)
        timeout_exceeded = df.get(
            "timeout_move_exceeded_band",
            pd.Series(np.zeros(n, dtype=np.int8), index=df.index)
        ).to_numpy(dtype=np.int8, copy=False)

        # ── احسب soft_label_long و soft_label_short ─────────────────────────
        soft_long  = np.full(n, 0.50, dtype=np.float32)
        soft_short = np.full(n, 0.50, dtype=np.float32)

        # جودة: STRONG أو WEAK (نُعالج NONE كـ WEAK)
        is_strong = (quality == QUALITY_STRONG)

        for path_val, table_long, table_short in [
            (PATH_LONG_TP_FIRST,  _BASE_SOFT_LONG, _BASE_SOFT_SHORT),
            (PATH_SHORT_TP_FIRST, _BASE_SOFT_LONG, _BASE_SOFT_SHORT),
            (PATH_LONG_SL_FIRST,  _BASE_SOFT_LONG, _BASE_SOFT_SHORT),
            (PATH_SHORT_SL_FIRST, _BASE_SOFT_LONG, _BASE_SOFT_SHORT),
        ]:
            mask = (path == path_val)
            if not mask.any():
                continue
            if path_val in table_long:
                soft_long[mask & ~is_strong]  = table_long[path_val][QUALITY_WEAK]
                soft_long[mask &  is_strong]  = table_long[path_val][QUALITY_STRONG]
            if path_val in table_short:
                soft_short[mask & ~is_strong] = table_short[path_val][QUALITY_WEAK]
                soft_short[mask &  is_strong] = table_short[path_val][QUALITY_STRONG]

        # TIMEOUT: soft label بناءً على forward_return / dynamic_threshold
        timeout_mask = (path == PATH_TIMEOUT)
        if timeout_mask.any():
            safe_dyn = np.where(dyn_thr > 1e-10, dyn_thr, 1e-10)
            norm_ret = fwd_ret / safe_dyn   # ∈ [-sl_mult, tp_mult] تقريباً
            # sigmoid: 1/(1+e^{-scale*x}), ثم نُخفف النطاق
            sigmoid_val = 1.0 / (1.0 + np.exp(-cfg.timeout_sigmoid_scale * norm_ret))
            ranged = 0.5 + cfg.timeout_soft_label_range * (sigmoid_val - 0.5)
            ranged_f = ranged.astype(np.float32)
            soft_long[timeout_mask]  = ranged_f[timeout_mask]
            soft_short[timeout_mask] = (1.0 - ranged_f[timeout_mask])  # متماثل

        # ── soft_label المحوري (باتجاه bias_label) ──────────────────────────
        soft_label = np.full(n, 0.5, dtype=np.float32)
        soft_label[bias == DIR_LONG]    = soft_long[bias == DIR_LONG]
        soft_label[bias == DIR_SHORT]   = soft_short[bias == DIR_SHORT]
        # NEUTRAL → يبقى 0.5

        # ── label_confidence ─────────────────────────────────────────────────
        confidence = self._compute_confidence(
            path=path, quality=quality,
            eff_hor=eff_hor, hor_steps=hor_steps,
            timeout_exceeded=timeout_exceeded,
        )

        # ── soft_sample_weight ───────────────────────────────────────────────
        clarity = np.abs(soft_label - 0.5) * 2.0     # 0=غامض, 1=واضح
        quality_mult = np.where(
            is_strong, cfg.strong_quality_boost, cfg.weak_quality_boost
        ).astype(np.float32)

        raw_weight = (
            (confidence ** cfg.confidence_weight_power)
            * clarity
            * quality_mult
        )
        raw_weight = np.clip(raw_weight, cfg.min_sample_weight, None)
        # نُرمّل حول المتوسط=1 للحفاظ على scale الـ learning rate
        mean_w = float(np.mean(raw_weight))
        if mean_w > 1e-8:
            soft_sample_weight = (raw_weight / mean_w).astype(np.float32)
        else:
            soft_sample_weight = np.ones(n, dtype=np.float32)

        # ── أضف الأعمدة ───────────────────────────────────────────────────────
        df["soft_label"]         = soft_label
        df["label_confidence"]   = confidence.astype(np.float32)
        df["soft_label_long"]    = soft_long
        df["soft_label_short"]   = soft_short
        df["soft_sample_weight"] = soft_sample_weight

        self._print_diagnostics(df)
        return df

    # ── الوضع Monte Carlo (افتراضي للتدريب عندما تريد soft_label غنيًّا) ─────

    def _attach_monte_carlo(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Monte Carlo: n_scenarios اختبارات مع perturbations على horizon / TP / SL.
        تعقيد O(n × n_scenarios) — أبطأ من analytical، لكنه يحدّ كثافة soft_label≈0.5.
        للسرعة الخالصة أو الـ smoke فقط استخدم الوضع التحليلي صراحةً (--soft_label_mode analytical).
        """
        cfg = self.config
        rng = np.random.default_rng(cfg.random_seed)
        n   = len(df)

        prices  = df["close"].to_numpy(dtype=np.float64, copy=False)
        bias    = df["bias_label"].to_numpy(dtype=np.int8, copy=False)
        quality = df["signal_quality"].to_numpy(dtype=np.int8, copy=False)
        dyn_thr = self._resolve_dynamic_threshold(df, n)

        # قراءة من config v19
        base_horizon = df["effective_horizon"].to_numpy(dtype=np.int32, copy=False)
        base_tp      = dyn_thr * 1.5   # tp_mult الافتراضي
        base_sl      = dyn_thr * 1.0   # sl_mult الافتراضي

        print(f"[SoftLabel-MC] Running {cfg.n_scenarios} Monte Carlo scenarios on {n:,} rows...")

        win_count  = np.zeros(n, dtype=np.int32)
        lose_count = np.zeros(n, dtype=np.int32)

        for s in range(cfg.n_scenarios):
            # Perturb parameters
            h  = np.clip(
                (base_horizon * (1 + rng.normal(0, cfg.horizon_std, n))).astype(int),
                1, len(prices) - 1
            )
            tp = np.clip(base_tp * (1 + rng.normal(0, cfg.tp_std, n)), 1e-5, None)
            sl = np.clip(base_sl * (1 + rng.normal(0, cfg.sl_std, n)), 1e-5, None)

            for i in range(n):
                end = min(i + h[i], n - 1)
                if end <= i:
                    continue
                future = prices[i+1 : end+1]
                ep = prices[i]

                if bias[i] == DIR_LONG:
                    if np.any(future >= ep + tp[i]):
                        win_count[i]  += 1
                    elif np.any(future <= ep - sl[i]):
                        lose_count[i] += 1

                elif bias[i] == DIR_SHORT:
                    if np.any(future <= ep - tp[i]):
                        win_count[i]  += 1
                    elif np.any(future >= ep + sl[i]):
                        lose_count[i] += 1

        win_prob  = win_count  / cfg.n_scenarios
        lose_prob = lose_count / cfg.n_scenarios
        timeout_prob = 1.0 - win_prob - lose_prob

        soft_label = win_prob.astype(np.float32)
        # NEUTRAL rows → force to 0.5
        soft_label[bias == DIR_NEUTRAL] = 0.5

        # Confidence: 1 - entropy
        eps = 1e-10
        p3  = np.stack([win_prob, lose_prob, np.maximum(timeout_prob, 0)], axis=1)
        p3  = np.clip(p3, eps, 1)
        p3  = p3 / p3.sum(axis=1, keepdims=True)
        entropy = -(p3 * np.log(p3)).sum(axis=1)
        max_entropy = np.log(3)
        confidence = (1.0 - entropy / max_entropy).astype(np.float32)

        # Sample weight
        is_strong = (quality == QUALITY_STRONG).astype(np.float32)
        quality_mult = (
            1.0 + (cfg.strong_quality_boost - 1.0) * is_strong
        )
        clarity    = np.abs(soft_label - 0.5) * 2.0
        raw_weight = confidence * clarity * quality_mult
        raw_weight = np.clip(raw_weight, cfg.min_sample_weight, None)
        mean_w = float(np.mean(raw_weight))
        soft_sample_weight = (raw_weight / max(mean_w, 1e-8)).astype(np.float32)

        df["soft_label"]         = soft_label
        df["label_confidence"]   = confidence
        df["soft_label_long"]    = np.where(bias == DIR_LONG,  soft_label, 1.0 - soft_label).astype(np.float32)
        df["soft_label_short"]   = np.where(bias == DIR_SHORT, soft_label, 1.0 - soft_label).astype(np.float32)
        df["soft_sample_weight"] = soft_sample_weight

        self._print_diagnostics(df)
        return df

    # ── دوال مساعدة ─────────────────────────────────────────────────────────

    def _resolve_dynamic_threshold(self, df: pd.DataFrame, n: int) -> np.ndarray:
        """
        استرجع dynamic_threshold من الأعمدة المخزونة.
        أولوية: label_dynamic_threshold → 0.5 * micro_atr → fallback
        """
        if "label_dynamic_threshold" in df.columns:
            arr = pd.to_numeric(df["label_dynamic_threshold"], errors="coerce").fillna(np.nan).to_numpy(dtype=np.float32)
            if not np.isnan(arr).all():
                return np.where(np.isnan(arr), np.nanmedian(arr) if not np.isnan(arr).all() else 1e-4, arr)

        if "micro_atr" in df.columns:
            atr = pd.to_numeric(df["micro_atr"], errors="coerce").fillna(np.nan).to_numpy(dtype=np.float32)
            thr = 0.5 * atr
            med = float(np.nanmedian(thr)) if not np.isnan(thr).all() else 1e-4
            return np.where(np.isnan(thr), med, thr)

        # fallback: estimate from forward_return at TP rows
        if "forward_return" in df.columns and "path_outcome" in df.columns:
            fwd = df["forward_return"].to_numpy(dtype=np.float32, copy=False)
            path = df["path_outcome"].to_numpy(dtype=np.int8, copy=False)
            tp_mask = (path == PATH_LONG_TP_FIRST) | (path == PATH_SHORT_TP_FIRST)
            if tp_mask.any():
                median_tp_ret = float(np.nanmedian(np.abs(fwd[tp_mask])))
                # TP = dyn_thr * tp_mult(≈1.5) → dyn_thr ≈ median_tp_ret / 1.5
                fallback_thr = max(median_tp_ret / 1.5, 1e-5)
                return np.full(n, fallback_thr, dtype=np.float32)

        warnings.warn(
            "[SoftLabel] Cannot resolve dynamic_threshold — using 1e-4 fallback. "
            "Add 'label_dynamic_threshold' column in labels_v19.py for accuracy.",
            RuntimeWarning,
            stacklevel=3,
        )
        return np.full(n, 1e-4, dtype=np.float32)

    def _compute_confidence(
        self,
        path: np.ndarray,
        quality: np.ndarray,
        eff_hor: np.ndarray,
        hor_steps: np.ndarray,
        timeout_exceeded: np.ndarray,
    ) -> np.ndarray:
        """
        حساب label_confidence لكل صف.
        
        المنطق:
        - PATH_TP أو PATH_SL (نتيجة واضحة): ثقة عالية (0.75)
          + جودة STRONG: +0.10 → 0.85
        - PATH_TIMEOUT (نتيجة غير محددة): ثقة منخفضة (0.40)
          + نسبة الأفق المستهلك: حتى +0.15
          + جودة STRONG: +0.10
          + تجاوز النطاق المحايد: +0.05
        """
        n = len(path)
        confidence = np.zeros(n, dtype=np.float32)

        is_strong  = (quality == QUALITY_STRONG)
        is_timeout = (path == PATH_TIMEOUT)

        # نسبة الأفق المستهلك (للـ timeout)
        safe_eff = np.where(eff_hor > 0, eff_hor, 1).astype(np.float32)
        horizon_usage = np.clip(hor_steps.astype(np.float32) / safe_eff, 0.0, 1.0)

        # مسارات واضحة (TP أو SL)
        clear_mask = ~is_timeout
        confidence[clear_mask]  = 0.75
        confidence[clear_mask & is_strong] += 0.10

        # TIMEOUT
        confidence[is_timeout]  = 0.40
        confidence[is_timeout] += 0.10 * is_strong[is_timeout].astype(float)
        confidence[is_timeout] += 0.15 * horizon_usage[is_timeout]
        confidence[is_timeout] += 0.05 * timeout_exceeded[is_timeout].astype(float)

        return np.clip(confidence, 0.0, 1.0).astype(np.float32)

    def _print_diagnostics(self, df: pd.DataFrame) -> None:
        """طباعة ملخص تشخيصي للـ Soft Labels."""
        sl     = df["soft_label"].to_numpy()
        conf   = df["label_confidence"].to_numpy()
        sw     = df["soft_sample_weight"].to_numpy()

        near_half = float(((sl > 0.45) & (sl < 0.55)).mean())
        avg_conf  = float(np.nanmean(conf))
        clarity   = float(np.nanmean(np.abs(sl - 0.5) * 2.0))
        bias      = df["bias_label"].to_numpy(dtype=np.int8)

        print("─" * 60)
        print("📊  Soft Labels — تقرير التشخيص")
        print("─" * 60)
        print(f"  الوضع          : {self.config.mode.upper()}")
        print(f"  قرب 0.5        : {near_half:.1%}  (يجب < 30%)")
        print(f"  متوسط الثقة    : {avg_conf:.3f}  (يجب > 0.50)")
        print(f"  متوسط الوضوح   : {clarity:.3f}  (كلما ارتفع كلما كانت الإشارات أوضح)")
        print(f"  LONG share     : {float((bias == DIR_LONG).mean()):.1%}")
        print(f"  SHORT share    : {float((bias == DIR_SHORT).mean()):.1%}")
        print(f"  NEUTRAL share  : {float((bias == DIR_NEUTRAL).mean()):.1%}")
        print(f"  وزن max/mean   : {float(sw.max()):.2f}/{float(sw.mean()):.2f}")

        # تحذيرات
        if near_half > 0.30:
            hint = ""
            if str(self.config.mode).lower() == "analytical":
                hint = (
                    " اعتمد monte_carlo في المصفاة (مثال: "
                    "--soft_label_mode monte_carlo --soft_label_n_scenarios 200)."
                )
            print(f"  ⚠️  نسبة عالية من الصفوف قرب 0.5 ({near_half:.1%}) — "
                  f"قد تعني غموضًا في الإشارات.{hint}")
        elif str(self.config.mode).lower() == "analytical" and clarity < 0.20:
            print(
                "  ⚠️  وضوح منخفض مع الوضع التحليلي — للتمايز أنصح monte_carlo "
                "(انظر configs/v19/defaults.yaml أو --soft_label_mode monte_carlo)."
            )
        if avg_conf < 0.50:
            print(f"  ⚠️  متوسط الثقة منخفض ({avg_conf:.3f}) — "
                  "فكّر في رفع حدود event_score")
        print("─" * 60)


# ─── دوال التكامل المباشرة (للاستخدام من labels_v19.py) ─────────────────────

def attach_soft_labels(
    labeled_df: pd.DataFrame,
    config: Optional[SoftLabelConfig] = None,
) -> pd.DataFrame:
    """
    واجهة بسيطة لإضافة Soft Labels إلى مخرجات build_causal_event_labels().
    
    مثال الاستخدام:
        labeled = build_causal_event_labels(df, ...)
        labeled = attach_soft_labels(labeled)
    """
    engine = SoftLabelEngine(config)
    return engine.attach_soft_labels(labeled_df)


def validate_soft_labels(labeled_df: pd.DataFrame) -> bool:
    """
    تحقق من جودة الـ Soft Labels قبل التدريب.
    يُرجع True إذا كانت الـ Soft Labels جاهزة للتدريب.
    """
    required = ["soft_label", "label_confidence", "soft_sample_weight"]
    missing  = [c for c in required if c not in labeled_df.columns]
    if missing:
        print(f"❌ أعمدة مفقودة: {missing}")
        return False

    sl   = labeled_df["soft_label"].to_numpy()
    conf = labeled_df["label_confidence"].to_numpy()

    near_half   = float(((sl > 0.45) & (sl < 0.55)).mean())
    avg_conf    = float(np.nanmean(conf))
    clarity     = np.abs(sl - 0.5)
    correlation = float(np.corrcoef(conf, clarity)[0, 1]) if len(conf) > 2 else 0.0
    long_bias   = float((sl > 0.5).mean())

    print("=" * 55)
    print("🔍  التحقق من صحة Soft Labels")
    print("=" * 55)
    print(f"  قرب 0.5         : {near_half:.1%}  {'✅' if near_half < 0.30 else '❌'} (يجب < 30%)")
    print(f"  متوسط الثقة     : {avg_conf:.3f}   {'✅' if avg_conf > 0.50 else '❌'} (يجب > 0.50)")
    print(f"  ارتباط ثقة/وضوح : {correlation:.3f}  {'✅' if correlation > 0.30 else '⚠️'} (يجب > 0.30)")
    print(f"  نسبة LONG       : {long_bias:.1%}  (معلومة فقط)")

    is_valid = (near_half < 0.30) and (avg_conf > 0.50)
    print(f"\n{'✅ Soft Labels جاهزة للتدريب' if is_valid else '❌ Soft Labels تحتاج مراجعة'}")
    print("=" * 55)
    return is_valid


def compute_soft_sample_weights(
    labeled_df: pd.DataFrame,
    quality_weight_strong: float = 2.0,
    quality_weight_weak:   float = 1.0,
    min_weight: float = 0.10,
    confidence_power: float = 1.0,
) -> np.ndarray:
    """
    احسب أوزان التدريب النهائية بدمج:
      - الأوزان الموجودة (quality-based من النظام القديم)
      - ثقة الـ Soft Labels
      - وضوح الـ Label

    يُرجع array من الأوزان بنفس طول labeled_df.
    يُستخدم مباشرةً في catboost_brain.fit() كـ sample_weight.
    """
    n = len(labeled_df)
    quality = labeled_df.get("signal_quality", pd.Series(np.ones(n, dtype=np.int8)))
    quality = quality.to_numpy(dtype=np.int8)

    # أوزان الجودة الأساسية (النظام القديم)
    q_weight = np.where(
        quality == QUALITY_STRONG,
        float(quality_weight_strong),
        float(quality_weight_weak)
    ).astype(np.float32)

    # إذا لم تكن Soft Labels موجودة، أرجع الأوزان الأساسية
    if "soft_label" not in labeled_df.columns or "label_confidence" not in labeled_df.columns:
        return q_weight / float(q_weight.mean() + 1e-8)

    soft_label = labeled_df["soft_label"].to_numpy(dtype=np.float32)
    confidence = labeled_df["label_confidence"].to_numpy(dtype=np.float32)

    clarity = np.abs(soft_label - 0.5) * 2.0

    combined = q_weight * (confidence ** confidence_power) * clarity
    combined = np.clip(combined, min_weight, None)

    mean_w = float(np.mean(combined))
    if mean_w < 1e-8:
        return np.ones(n, dtype=np.float32)

    return (combined / mean_w).astype(np.float32)
