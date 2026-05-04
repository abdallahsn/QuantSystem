"""
catboost_brain.py — CatBoost Quantitative Brain
═══════════════════════════════════════════════════════════════════════
البديل السريع والقوي للـ Transformer

المزايا:
  ✅ يتدرب في دقائق بدل ساعات
  ✅ مناعة قوية ضد Overfitting
  ✅ يعمل على 2D tabular (مش محتاج sequences)
  ✅ SHAP Values مدمج
  ✅ يتجاهل الضجيج تلقائياً

المخرجات:
  bias: LONG/SHORT/NEUTRAL
  confidence: قوة الإشارة
  shap_values: أهمية كل feature
═══════════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import pandas as pd
import json
from typing import Optional
import pickle
from collections import deque

try:
    from modules.range_state_machine import RangeStateMachine
    RSM_AVAILABLE = True
except ImportError:
    try:
        from range_state_machine import RangeStateMachine
        RSM_AVAILABLE = True
    except ImportError:
        RSM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, Pool
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False
    print("  ⚠️  CatBoost غير مثبّت — pip install catboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

BIAS_LABELS = {0: 'LONG', 1: 'SHORT', 2: 'NEUTRAL'}


class CatBoostQuantBrain:
    """
    CatBoost Classifier للتنبؤ بـ LONG/SHORT/NEUTRAL
    على بيانات 2D Tabular (آخر تيك × features + embeddings)
    """

    def __init__(self,
                 feature_cols: list,
                 brain_filename: str = 'outputs/catboost_brain.cbm',
                 depth: int = 6,
                 iterations: int = 500,
                 learning_rate: float = 0.05,
                 l2_leaf_reg: float = 3.0,
                 confidence_threshold: float = 0.60,
                 embeddings_dim: int = 8,
                 # ── FIX: Rolling Context Window ──────────────────────────
                 use_rolling_context: bool = True,
                 rolling_window: int = 12,
                 # 12 شمعة × 5 دقائق = 60 دقيقة من السياق الزمني
                 # CatBoost يرى الآن mean/std آخر 12 شمعة لكل feature
                 # بدلاً من لقطة واحدة فقط → يعرف أين هو في الترند
                 ):

        # ندمج أسماء الميزات الأصلية مع أسماء الـ Embeddings
        self.base_feature_cols = feature_cols
        self.embeddings_dim = embeddings_dim
        self.feature_cols = feature_cols + [f'emb_{i}' for i in range(embeddings_dim)]

        self.brain_file           = brain_filename
        self.depth                = depth
        self.iterations           = iterations
        self.learning_rate        = learning_rate
        self.l2_leaf_reg          = l2_leaf_reg
        self.confidence_threshold = confidence_threshold

        # Rolling context
        self.use_rolling_context = use_rolling_context
        self.rolling_window      = rolling_window
        self._ctx_feature_cols   = list(self.feature_cols)  # يُحدَّث بعد fit
        self._predict_buffer     = deque(maxlen=rolling_window)  # بافر للـ predict اللحظي

        # ── RangeStateMachine — يمنع التقلب في الرينج ─────────────────
        # يراكم إشارات CatBoost ولا يُصدر قرار إلا بعد N تأكيدات
        # من الحافة الصحيحة (قاع الرينج للشراء، قمته للبيع)
        self._rsm: Optional[RangeStateMachine] = None
        if RSM_AVAILABLE:
            self._rsm = RangeStateMachine(
                range_window        = rolling_window,
                min_confirmations   = 3,
                confirmation_window = 6,
                min_confidence      = confidence_threshold * 0.85,
                min_candles_between = 4,
            )

        self.model       = None
        self._fitted     = False

        if not CB_AVAILABLE:
            return

        if os.path.exists(brain_filename):
            try:
                print(f"[CatBoost] 🧠 تحميل: {brain_filename}")
                self.model   = CatBoostClassifier()
                self.model.load_model(brain_filename)
                self._fitted = True
            except Exception as e:
                print(f"[CatBoost] ⚠️ مكسور ({e}) — بنبني جديد")
                self.model = None

    # ── Rolling Context ────────────────────────────────────────────────────────

    def _build_rolling_features(self, X: np.ndarray, cols: list) -> tuple:
        """
        FIX-Context: يُضيف rolling mean + std آخر rolling_window صف لكل feature.

        المشكلة:
          CatBoost كان يرى لقطة واحدة (تيك واحد) بلا ذاكرة زمنية.
          النتيجة: يُغيّر رأيه مع كل شمعة لأنه لا يعرف إذا كنا في ترند.

        الحل:
          لكل صف i نُضيف:
            - mean(X[i-w+1 : i+1]) → الاتجاه العام في النافذة
            - std(X[i-w+1 : i+1])  → مستوى التذبذب في النافذة

          هذا يُعطي CatBoost "ذاكرة" بسيطة دون الحاجة لـ LSTM/Transformer.

        Returns: (X_ctx, ctx_cols)
        """
        if not self.use_rolling_context or len(X) == 0:
            return X, cols

        w = self.rolling_window
        df_X = pd.DataFrame(X, columns=cols)

        roll_mean = df_X.rolling(w, min_periods=1).mean()
        roll_std  = df_X.rolling(w, min_periods=1).std().fillna(0.0)

        roll_mean.columns = [f'{c}__rm{w}' for c in cols]
        roll_std.columns  = [f'{c}__rs{w}' for c in cols]

        X_ctx    = np.hstack([X, roll_mean.values, roll_std.values])
        ctx_cols = cols + list(roll_mean.columns) + list(roll_std.columns)
        return X_ctx.astype(np.float32), ctx_cols

    def _rolling_from_buffer(self, x_combined: np.ndarray) -> np.ndarray:
        """
        يحسب rolling context من بافر الـ predict اللحظي.
        يُستخدم في predict() عند التداول الحي — بافر يحفظ آخر rolling_window صف.
        """
        if not self.use_rolling_context:
            return x_combined

        self._predict_buffer.append(x_combined.copy())
        buf = np.array(self._predict_buffer)  # (k, features), k <= rolling_window

        ctx_mean = buf.mean(axis=0)
        ctx_std  = buf.std(axis=0) if len(buf) > 1 else np.zeros_like(ctx_mean)
        return np.concatenate([x_combined, ctx_mean, ctx_std]).astype(np.float32)

    # ── Training ────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y_bias: np.ndarray, embeddings: np.ndarray = None,
            output_dir: str = 'outputs') -> dict:
        """
        يدرّب CatBoost
        X: (n_samples, n_features) — آخر تيك من كل sequence
        y_bias: (n_samples,) — LONG/SHORT/NEUTRAL
        embeddings: (n_samples, embeddings_dim) — مخرجات الـ Autoencoder
        """
        if not CB_AVAILABLE:
            return {}

        os.makedirs(output_dir, exist_ok=True)
        print("\n🐱 CatBoost Training...")

        # دمج X مع embeddings إذا توفرت
        if embeddings is not None:
            if X.shape[0] != embeddings.shape[0]:
                print(f"  ⚠️ عدم تطابق في الأبعاد بين X ({X.shape[0]}) و embeddings ({embeddings.shape[0]})")
                return {}
            # دمج الأعمدة
            X_combined = np.hstack((X, embeddings))
        else:
             # إذا لم تُمرر Embeddings، نملأها بأصفار لتجنب الخطأ
             zeros_emb = np.zeros((X.shape[0], self.embeddings_dim))
             X_combined = np.hstack((X, zeros_emb))
             print("  ⚠️ لم يتم تمرير Embeddings لـ CatBoost، تم استخدام قيم صفرية مؤقتاً.")


        # ── FIX: تطبيق Rolling Context (mean/std آخر rolling_window شمعة) ──
        # يُعطي CatBoost ذاكرة زمنية لـ 60 دقيقة بدلاً من لقطة واحدة
        current_base_cols = self.feature_cols
        X_combined, ctx_cols = self._build_rolling_features(X_combined, current_base_cols)
        self._ctx_feature_cols = ctx_cols  # نحفظها للـ predict

        # 🔴 تصحيح الجراحة: فلترة القيم السالبة لمنع Crash الـ bincount
        valid_mask = (y_bias >= 0) & (y_bias <= 2)
        X_clean = X_combined[valid_mask]
        y_clean = y_bias[valid_mask]

        if len(X_clean) == 0:
            print("  ⚠️ لا توجد بيانات صالحة للتدريب بعد الفلترة.")
            return {}

        # تحضير الداتا
        split    = int(len(X_clean) * 0.8)
        X_tr, X_val = X_clean[:split], X_clean[split:]
        y_tr, y_val = y_clean[:split], y_clean[split:]

        # class weights تلقائي
        total  = len(y_tr)
        counts = np.bincount(y_tr, minlength=3)
        cw     = {k: total / (3 * max(c,1)) for k, c in enumerate(counts)}
        sample_w = np.array([cw[y] for y in y_tr], dtype=np.float32)

        print(f"  Train: {len(X_tr):,} | Val: {len(X_val):,}")
        print(f"  Weights: LONG={cw[0]:.2f} SHORT={cw[1]:.2f} NEUTRAL={cw[2]:.2f}")

        # التحقق من تطابق عدد الميزات
        if X_tr.shape[1] != len(self._ctx_feature_cols):
             print(f"  ⚠️ خطأ في الأبعاد: X_tr={X_tr.shape[1]}, ctx_feature_cols={len(self._ctx_feature_cols)}")
             return {}

        train_pool = Pool(X_tr, y_tr, sample_weight=sample_w,
                          feature_names=self._ctx_feature_cols)
        val_pool   = Pool(X_val, y_val,
                          feature_names=self._ctx_feature_cols)

        from modules.gpu_config import GPU_AVAILABLE
        self.model = CatBoostClassifier(
            iterations      = self.iterations,
            depth           = self.depth,
            learning_rate   = self.learning_rate,
            l2_leaf_reg     = self.l2_leaf_reg,
            loss_function   = 'MultiClass',
            eval_metric     = 'Accuracy',
            early_stopping_rounds = 50,
            use_best_model  = True,
            verbose         = 50,
            random_seed     = 42,
            task_type       = 'GPU' if GPU_AVAILABLE else 'CPU',
            devices         = '0'   if GPU_AVAILABLE else None,
        )

        self.model.fit(
            train_pool,
            eval_set=val_pool,
            plot=False,
        )

        # حفظ الموديل
        model_path = os.path.join(output_dir, 'catboost_brain.cbm')
        self.model.save_model(model_path)
        self._fitted = True

        # تقرير الدقة
        val_pred = self.model.predict(X_val).flatten().astype(int)
        accuracy = float(np.mean(val_pred == y_val))
        print(f"\n  ✅ Val Accuracy: {accuracy:.2%}")

        # SHAP Values
        shap_result = self._compute_shap(X_val, y_val, output_dir)

        metrics = {
            'val_accuracy': accuracy,
            'best_iteration': self.model.get_best_iteration(),
            'feature_importance': self._get_feature_importance(),
        }
        metrics.update(shap_result)

        # حفظ metrics
        metrics_path = os.path.join(output_dir, 'catboost_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({k: v for k, v in metrics.items()
                      if isinstance(v, (int, float, str, list, dict))}, f, indent=2)

        return metrics

    def _get_feature_importance(self) -> dict:
        """أهمية كل feature"""
        if not self._fitted:
            return {}
        imp = self.model.get_feature_importance()
        names = self.feature_cols
        ranked = sorted(zip(names, imp), key=lambda x: -x[1])
        return {name: round(float(val), 4) for name, val in ranked}

    def _compute_shap(self, X_val: np.ndarray,
                       y_val: np.ndarray,
                       output_dir: str) -> dict:
        """
        يحسب SHAP Values ويرسم Dashboard
        """
        if not SHAP_AVAILABLE or not self._fitted:
            print("  ⚠️  SHAP غير متاح — pip install shap")
            return {}

        print("\n📊 SHAP Analysis...")
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            explainer   = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(
                Pool(X_val, feature_names=self.feature_cols))

            # ── Plot 1: Feature Importance (SHAP) ────────────────
            fig, axes = plt.subplots(1, 3, figsize=(20, 8),
                                      facecolor='#0d1117')
            fig.suptitle('SHAP Analysis — V16Pro-1',
                          color='white', fontsize=14, fontweight='bold')

            class_names = ['LONG', 'SHORT', 'NEUTRAL']
            colors      = ['#00ff88', '#ff4444', '#ffaa00']

            for idx, (cls_name, color) in enumerate(zip(class_names, colors)):
                ax  = axes[idx]
                ax.set_facecolor('#161b22')

                sv  = shap_values[idx]  # (n, features)
                imp = np.abs(sv).mean(axis=0)
                ranked = sorted(zip(self.feature_cols, imp),
                                key=lambda x: -x[1])[:15]

                names_r = [r[0] for r in ranked]
                vals_r  = [r[1] for r in ranked]

                bars = ax.barh(range(len(names_r)), vals_r,
                               color=color, alpha=0.8)

                ax.set_yticks(range(len(names_r)))
                ax.set_yticklabels(names_r, color='white', fontsize=9)
                ax.set_xlabel('Mean |SHAP|', color='gray')
                ax.set_title(f'{cls_name}', color=color, fontweight='bold')
                ax.tick_params(colors='gray')

                for spine in ax.spines.values():
                    spine.set_edgecolor('#333')

            plt.tight_layout()
            shap_path = os.path.join(output_dir, 'shap_dashboard.png')
            plt.savefig(shap_path, dpi=150, bbox_inches='tight',
                        facecolor='#0d1117')
            plt.close()
            print(f"  ✅ SHAP Dashboard: {shap_path}")

            # ── أهم 5 features لكل class ─────────────────────────
            shap_summary = {}
            for idx, cls_name in enumerate(class_names):
                sv   = shap_values[idx]
                imp  = np.abs(sv).mean(axis=0)
                top5 = sorted(zip(self.feature_cols, imp),
                              key=lambda x: -x[1])[:5]
                shap_summary[cls_name] = [(n, round(float(v),4)) for n,v in top5]
                print(f"  {cls_name} top features: {[n for n,_ in top5]}")

            # حفظ SHAP
            shap_pkl = os.path.join(output_dir, 'shap_values.pkl')
            with open(shap_pkl, 'wb') as f:
                pickle.dump({'shap_values': shap_values,
                             'feature_names': self.feature_cols,
                             'summary': shap_summary}, f)

            return {'shap_summary': shap_summary}

        except Exception as e:
            print(f"  ⚠️ SHAP error: {e}")
            return {}

    def predict(self, x: np.ndarray, embeddings: np.ndarray = None) -> dict:
        """
        يتنبأ لـ row واحد
        x: (n_features,) — آخر تيك
        embeddings: (embeddings_dim,) — مخرجات الـ Autoencoder
        """
        if not self._fitted or self.model is None:
            return {'bias': 'NEUTRAL', 'bias_idx': 2,
                    'confidence': 0.0, 'tradeable': False}

        # دمج X مع Embeddings
        if embeddings is not None:
             x_combined = np.concatenate((x, embeddings))
        else:
             zeros_emb = np.zeros(self.embeddings_dim)
             x_combined = np.concatenate((x, zeros_emb))

        # ── FIX: Rolling Context (ذاكرة زمنية) ──
        # نضيف لـ x_combined mean/std من بافر آخر rolling_window صف
        # CatBoost يرى الآن السياق الزمني لآخر 12 شمعة (60 دقيقة)
        x_ctx = self._rolling_from_buffer(x_combined)

        x2d   = x_ctx.reshape(1, -1)

        # التحقق من عدد الميزات قبل التنبؤ
        expected_cols = len(self._ctx_feature_cols)
        if x2d.shape[1] != expected_cols:
             print(f"  ⚠️ خطأ في الأبعاد للتنبؤ: المدخلات={x2d.shape[1]}, المطلوب={expected_cols}")
             return {'bias': 'NEUTRAL', 'bias_idx': 2, 'confidence': 0.0, 'tradeable': False}

        probs = self.model.predict_proba(x2d)[0]

        bias_idx   = int(np.argmax(probs))
        confidence = float(probs[bias_idx])
        bias       = BIAS_LABELS[bias_idx]
        tradeable  = (confidence >= self.confidence_threshold
                      and bias != 'NEUTRAL')

        return {
            'bias':        bias,
            'bias_idx':    bias_idx,
            'confidence':  round(confidence, 4),
            'tradeable':   tradeable,
            'probs':       {BIAS_LABELS[i]: round(float(p), 4)
                           for i, p in enumerate(probs)},
        }

    def should_trade(
        self,
        x: np.ndarray,
        embeddings: np.ndarray = None,
        price: float = 0.0,
        regime: str = 'Ranging',
    ) -> tuple:
        """
        Interface متوافق مع TransformerBrain — مع RangeStateMachine.

        Parameters
        ----------
        x          : features
        embeddings : autoencoder embeddings
        price      : السعر الحالي (للـ RSM)
        regime     : حالة السوق من regime_classifier
        """
        result    = self.predict(x, embeddings)
        tradeable = result.pop('tradeable', False)

        # ── تطبيق RangeStateMachine ────────────────────────────────────
        # بدلاً من إصدار كل إشارة CatBoost مباشرة:
        # نمررها للـ RSM يراكمها ويُصدر قرار مؤكد فقط
        if self._rsm is not None and price > 0:
            raw_bias   = result.get('bias', 'NEUTRAL')
            confidence = result.get('confidence', 0.0)

            decision = self._rsm.process(
                price      = price,
                raw_signal = raw_bias,
                confidence = confidence,
                regime     = regime,
            )

            if decision['action'] == 'ENTER':
                # إشارة مؤكدة — أصدرها
                result['bias']       = decision['direction']
                result['rsm_reason'] = decision['reason']
                result['rsm_state']  = decision['market_state']
                result['rsm_confs']  = decision['confirmations']
                tradeable = (decision['direction'] in ('LONG', 'SHORT'))
            else:
                # لا تزال في مرحلة التراكم — لا تدخل
                result['bias']       = 'NEUTRAL'
                result['rsm_reason'] = decision['reason']
                result['rsm_state']  = decision['market_state']
                result['rsm_confs']  = decision['confirmations']
                tradeable = False

        return tradeable, result

    def notify_trade_closed(self):
        """
        استدعِ هذا عند إغلاق الصفقة لفتح RSM للإشارة التالية.
        """
        if self._rsm is not None:
            self._rsm.unlock()

    def get_rsm_summary(self) -> dict:
        """تقرير حالة الـ RangeStateMachine."""
        if self._rsm is None:
            return {'rsm': 'unavailable'}
        return self._rsm.summary()

    def get_feature_report(self) -> str:
        """تقرير أهمية الـ features"""
        if not self._fitted:
            return "CatBoost: غير مدرّب"

        imp  = self._get_feature_importance()
        lines = ["\n🐱 CatBoost Feature Importance:"]
        for i, (name, val) in enumerate(list(imp.items())[:15], 1):
            bar = '█' * int(val / 2)
            lines.append(f"  {i:2}. {name:<28} {val:5.2f}% {bar}")
        return '\n'.join(lines)
