"""
catboost_brain.py — CatBoost Quantitative Brain
═══════════════════════════════════════════════════════════════════════
LEGACY NOTE:
  This module is currently NOT the training path used by stage2_catboost.py.
  The active V19 Stage 2 pipeline trains CatBoost/XGBoost via train_v19.py
  on directional event rows only. This file remains as a legacy/experimental
  3-class CatBoost brain and should not be assumed to affect current Stage 2
  results unless it is explicitly wired into the pipeline.

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
import pickle
from collections import deque

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
        print(
            "[CatBoostQuantBrain] ⚠️ Legacy module loaded. "
            "stage2_catboost.py currently trains via train_v19.py, not via modules/catboost_brain.py."
        )

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

        self.model       = None
        self._fitted     = False

        if not CB_AVAILABLE:
            return

        if os.path.exists(brain_filename):
            try:
                print(f"[CatBoost] 🧠 تحميل: {brain_filename}")
                self.model   = CatBoostClassifier()
                self.model.load_model(brain_filename)
                self._restore_feature_metadata(brain_filename)
                self._fitted = True
            except Exception as e:
                print(f"[CatBoost] ⚠️ مكسور ({e}) — بنبني جديد")
                self.model = None

    def _resolved_model_path(self, output_dir: str) -> str:
        if os.path.isabs(self.brain_file):
            return self.brain_file
        return os.path.join(output_dir, os.path.basename(self.brain_file))

    @staticmethod
    def _metadata_path(model_path: str) -> str:
        root, _ = os.path.splitext(model_path)
        return f"{root}.meta.json"

    def _infer_ctx_feature_cols(self) -> list:
        base_cols = list(self.feature_cols)
        if not self.use_rolling_context:
            return base_cols
        w = int(self.rolling_window)
        return (
            base_cols
            + [f'{c}__rm{w}' for c in base_cols]
            + [f'{c}__rs{w}' for c in base_cols]
        )

    def _restore_feature_metadata(self, model_path: str) -> None:
        feature_names = None
        meta_path = self._metadata_path(model_path)
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    payload = json.load(f)
                self.use_rolling_context = bool(payload.get('use_rolling_context', self.use_rolling_context))
                self.rolling_window = max(int(payload.get('rolling_window', self.rolling_window)), 1)
                self._predict_buffer = deque(maxlen=self.rolling_window)
                feature_names = payload.get('ctx_feature_cols')
            except Exception as e:
                print(f"[CatBoost] ⚠️ تعذر قراءة metadata ({e}) — سيتم استخدام fallback")

        if not feature_names and self.model is not None:
            feature_names = getattr(self.model, 'feature_names_', None)

        if isinstance(feature_names, (list, tuple)) and len(feature_names) > 0:
            self._ctx_feature_cols = [str(col) for col in feature_names]
            return

        self._ctx_feature_cols = self._infer_ctx_feature_cols()

    def _save_feature_metadata(self, model_path: str) -> None:
        payload = {
            'base_feature_cols': list(self.base_feature_cols),
            'feature_cols': list(self.feature_cols),
            'ctx_feature_cols': list(self._ctx_feature_cols),
            'embeddings_dim': int(self.embeddings_dim),
            'use_rolling_context': bool(self.use_rolling_context),
            'rolling_window': int(self.rolling_window),
            'confidence_threshold': float(self.confidence_threshold),
        }
        with open(self._metadata_path(model_path), 'w') as f:
            json.dump(payload, f, indent=2)

    def _effective_feature_names(self, n_features: int | None = None) -> list:
        names = list(self._ctx_feature_cols) if self._ctx_feature_cols else self._infer_ctx_feature_cols()
        if n_features is None or len(names) == int(n_features):
            return names
        if len(self.feature_cols) == int(n_features):
            return list(self.feature_cols)
        return [f'feature_{i}' for i in range(int(n_features))]

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
            output_dir: str = 'outputs',
            soft_label_weights: np.ndarray = None) -> dict:
        """
        يدرّب CatBoost
        X: (n_samples, n_features) — آخر تيك من كل sequence
        y_bias: (n_samples,) — LONG/SHORT/NEUTRAL
        embeddings: (n_samples, embeddings_dim) — مخرجات الـ Autoencoder
        soft_label_weights: (n_samples,) — أوزان Soft Labels الاختيارية
            إذا مُرّرت: تُضرب في class-balance weights بدلاً من استخدام quality weights فقط
            احصل عليها من: soft_label_engine.compute_soft_sample_weights(labeled_df)
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
        if len(X_clean) <= 1:
            print("  ⚠️ عدد العينات غير كافٍ لتقسيم train/val آمن.")
            return {}

        split = int(len(X_clean) * 0.8)
        split = min(max(split, 1), len(X_clean) - 1)
        X_tr, X_val = X_clean[:split], X_clean[split:]
        y_tr, y_val = y_clean[:split], y_clean[split:]

        # class weights تلقائي
        total  = len(y_tr)
        counts = np.bincount(y_tr, minlength=3)
        cw     = {k: total / (3 * max(c,1)) for k, c in enumerate(counts)}
        sample_w = np.array([cw[y] for y in y_tr], dtype=np.float32)

        # ── FIX-12: دمج Soft Label Weights ──────────────────────────────────
        # إذا مُرّرت soft_label_weights، نضربها في class-balance weights
        # هذا يجعل النموذج يُركّز على الإشارات الواضحة عالية الثقة
        # بينما يُهمّش الصفوف الغامضة (soft_label ≈ 0.5) تلقائياً
        if soft_label_weights is not None:
            try:
                sl_w = np.asarray(soft_label_weights, dtype=np.float32)
                # فلتر نفس valid_mask المُطبّق على X و y
                sl_w_clean = sl_w[valid_mask] if len(sl_w) == (len(valid_mask) if hasattr(valid_mask, '__len__') else len(y_bias)) else sl_w
                if len(sl_w_clean) == len(sample_w):
                    sl_w_tr = sl_w_clean[:split]
                    # حوّل الأوزان إلى نفس scale
                    sl_w_tr = np.clip(sl_w_tr, 0.05, None)
                    sl_w_tr = sl_w_tr / float(np.mean(sl_w_tr) + 1e-8)
                    sample_w = sample_w * sl_w_tr
                    # أعد التطبيع
                    sample_w = sample_w / float(np.mean(sample_w) + 1e-8)
                    print(f"  ✅ Soft Label Weights مُطبّقة: mean={float(np.mean(sample_w)):.2f} "
                          f"max={float(np.max(sample_w)):.2f}")
                else:
                    print(f"  ⚠️ soft_label_weights طولها {len(sl_w_clean)} ≠ {len(sample_w)} — تجاهل")
            except Exception as _sw_err:
                print(f"  ⚠️ خطأ في تطبيق soft_label_weights: {_sw_err!r} — تجاهل")
        # ── نهاية FIX-12 ────────────────────────────────────────────────────

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
        model_path = self._resolved_model_path(output_dir)
        self.model.save_model(model_path)
        self.brain_file = model_path
        self._save_feature_metadata(model_path)
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
        names = self._effective_feature_names(len(imp))
        ranked = sorted(zip(names, imp), key=lambda x: -x[1])
        return {name: round(float(val), 4) for name, val in ranked}

    # ── FIX-12: Binary Soft Label Head ───────────────────────────────────────

    def fit_soft_binary(
        self,
        X: np.ndarray,
        soft_label: np.ndarray,
        label_confidence: np.ndarray = None,
        output_dir: str = 'outputs',
        model_suffix: str = '_soft_binary',
    ) -> dict:
        """
        FIX-12 — تدريب رأس ثنائي بـ CrossEntropy مع Soft Labels.

        يُكمّل الـ MultiClass الأساسي بنموذج ثنائي يُقدّر:
          P(win | direction, features) ∈ [0, 1]

        الفرق عن fit() الأساسي:
          - loss_function = 'CrossEntropy'  (يقبل احتمالية مستمرة، ليس 0/1 فقط)
          - target = soft_label ∈ [0, 1]   (بدلاً من class index 0/1/2)
          - sample_weight = confidence × |soft_label - 0.5| × 2
          - eval_metric = 'AUC'             (أفضل للتمييز)

        المدخلات:
          X              : (n_samples, n_features) — نفس ميزات fit() الأساسي
          soft_label     : (n_samples,) — P(win) ∈ [0, 1] من soft_label_engine
          label_confidence: (n_samples,) — وزن الثقة ∈ [0, 1] (اختياري)
          output_dir     : مجلد الحفظ
          model_suffix   : لاحقة اسم ملف النموذج

        المخرجات:
          dict مع: val_auc, val_logloss, best_iteration
        """
        if not CB_AVAILABLE:
            return {}

        import os
        os.makedirs(output_dir, exist_ok=True)
        print("\n🎯 CatBoost Soft Binary Training (CrossEntropy)...")

        # تحضير الأوزان
        n = len(soft_label)
        soft_label = np.clip(np.asarray(soft_label, dtype=np.float32), 0.01, 0.99)
        clarity = np.abs(soft_label - 0.5) * 2.0

        if label_confidence is not None:
            conf = np.clip(np.asarray(label_confidence, dtype=np.float32), 0.01, 1.0)
        else:
            conf = np.ones(n, dtype=np.float32)

        sample_w = conf * clarity
        sample_w = np.clip(sample_w, 0.05, None)
        sample_w = sample_w / float(np.mean(sample_w) + 1e-8)

        # تقسيم بدون خلط (بيانات مالية → ترتيب زمني ضروري)
        split = int(n * 0.8)
        split = min(max(split, 1), n - 1)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = soft_label[:split], soft_label[split:]
        sw_tr       = sample_w[:split]

        print(f"  Train: {len(X_tr):,} | Val: {len(X_val):,}")
        print(f"  soft_label mean(train)={float(y_tr.mean()):.3f} | "
              f"clarity mean={float(clarity[:split].mean()):.3f}")

        feature_names = getattr(self, '_ctx_feature_cols', None)

        train_pool = Pool(
            X_tr, y_tr,
            sample_weight=sw_tr,
            feature_names=feature_names,
        )
        val_pool = Pool(
            X_val, y_val,
            feature_names=feature_names,
        )

        from modules.gpu_config import GPU_AVAILABLE

        soft_model = CatBoostClassifier(
            iterations            = self.iterations,
            depth                 = self.depth,
            learning_rate         = self.learning_rate,
            l2_leaf_reg           = self.l2_leaf_reg,
            loss_function         = 'CrossEntropy',   # يقبل soft labels [0,1]
            eval_metric           = 'AUC',             # AUC أفضل من Accuracy للتمييز
            early_stopping_rounds = 50,
            use_best_model        = True,
            verbose               = 50,
            random_seed           = 42,
            task_type             = 'GPU' if GPU_AVAILABLE else 'CPU',
            devices               = '0'   if GPU_AVAILABLE else None,
        )

        soft_model.fit(train_pool, eval_set=val_pool, plot=False)

        # حفظ النموذج الثنائي بجانب النموذج الأساسي
        base_path  = self._resolved_model_path(output_dir)
        soft_path  = base_path.replace('.cbm', f'{model_suffix}.cbm')
        soft_model.save_model(soft_path)
        print(f"\n  ✅ Soft Binary Model → {soft_path}")

        # تقييم
        val_proba = soft_model.predict_proba(X_val)[:, 1]
        from sklearn.metrics import roc_auc_score, log_loss
        try:
            val_auc = float(roc_auc_score(y_val > 0.5, val_proba))
        except Exception:
            val_auc = float('nan')
        try:
            val_logloss = float(log_loss(y_val, val_proba))
        except Exception:
            val_logloss = float('nan')

        print(f"  Val AUC = {val_auc:.4f} | LogLoss = {val_logloss:.4f}")

        # حفظ مرجع للنموذج الثنائي
        self._soft_binary_model   = soft_model
        self._soft_binary_path    = soft_path

        return {
            'val_auc':       val_auc,
            'val_logloss':   val_logloss,
            'best_iteration': soft_model.get_best_iteration(),
            'model_path':    soft_path,
        }

    def predict_soft_binary(self, X: np.ndarray) -> np.ndarray:
        """
        FIX-12 — تنبؤ بـ P(win) من الرأس الثنائي.
        يُرجع array من الاحتماليات ∈ [0, 1].
        يتطلب استدعاء fit_soft_binary() أولاً.
        """
        if not hasattr(self, '_soft_binary_model') or self._soft_binary_model is None:
            raise RuntimeError("fit_soft_binary() لم يُستدعَ بعد.")
        proba = self._soft_binary_model.predict_proba(X)[:, 1]
        return proba.astype(np.float32)

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

            feature_names = self._effective_feature_names(X_val.shape[1])
            explainer   = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(
                Pool(X_val, feature_names=feature_names))

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
                ranked = sorted(zip(feature_names, imp),
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
                top5 = sorted(zip(feature_names, imp),
                              key=lambda x: -x[1])[:5]
                shap_summary[cls_name] = [(n, round(float(v),4)) for n,v in top5]
                print(f"  {cls_name} top features: {[n for n,_ in top5]}")

            # حفظ SHAP
            shap_pkl = os.path.join(output_dir, 'shap_values.pkl')
            with open(shap_pkl, 'wb') as f:
                pickle.dump({'shap_values': shap_values,
                             'feature_names': feature_names,
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

    def should_trade(self, x: np.ndarray, embeddings: np.ndarray = None) -> tuple:
        """Interface متوافق مع TransformerBrain"""
        result    = self.predict(x, embeddings)
        tradeable = result.pop('tradeable')
        return tradeable, result

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
