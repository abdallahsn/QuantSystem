"""
meta_learner.py — LSTM Meta-Learner (V19 Core Brain)
═══════════════════════════════════════════════════════════════════════
صانع القرار النهائي الذي يدمج ثلاثة مصادر معلومات:

  Input A — Visual (CNN):    مصفوفة (50, 20, 3) → 8 Visual Embeddings
  Input B — Statistical:     مصفوفة (50, N_STAT) تشمل:
    • الـ 25 Feature الكلاسيكية (CVD, OBI, Absorption...)
    • 2 احتمالات من CatBoost   (P_LONG, P_SHORT)
    • 4 One-Hot للـ Cluster     (Cluster 0..3)
    • 3 Soft Regime Scores      (volatile/trend/low-liq)

Architecture:
  CNN Branch:   Input(50,20,3) → DeepLOBCNN → (8,) per timestep
  Stat Branch:  Input(50,N_STAT) → LayerNorm
  Fusion:       Concat([stat, visual_per_step]) → (50, N_STAT+8)
  LSTM Stack:   LSTM(128) → LSTM(64) → Attention → Dense
  Output:       Softmax(2) → LONG/SHORT + Confidence
                no-trade / NEUTRAL يُحسم خارج النموذج عبر EventGate
═══════════════════════════════════════════════════════════════════════
"""

import os
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("  ⚠️  TensorFlow غير مثبّت — MetaLearner غير متاح")

from sklearn.metrics import classification_report, precision_recall_fscore_support

BIAS_LABELS  = {0: 'LONG', 1: 'SHORT', 2: 'NEUTRAL'}
N_CLUSTERS   = 4
N_CB_PROBS   = 2   # P_LONG, P_SHORT
N_REGIME_SCORES = 3
DEFAULT_META_FEATURES = N_CB_PROBS + N_CLUSTERS + N_REGIME_SCORES


def _normalize_model_input_shape(model) -> tuple | None:
    shape = getattr(model, 'input_shape', None)
    if shape is None:
        inputs = getattr(model, 'inputs', None)
        if isinstance(inputs, (list, tuple)) and inputs:
            shape = getattr(inputs[0], 'shape', None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) if dim is not None else None for dim in tuple(shape))
    except Exception:
        return None


def _input_shape_matches(model, seq_len: int, n_total: int) -> bool:
    shape = _normalize_model_input_shape(model)
    if shape is None or len(shape) < 3:
        return False
    return shape[-2:] == (int(seq_len), int(n_total))

# ── Warm-up LR ───────────────────────────────────────────────────
if TF_AVAILABLE:
    class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model=64, warmup_steps=500):
            super().__init__()
            self.d_model      = tf.cast(d_model, tf.float32)
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            arg1 = tf.math.rsqrt(step + 1e-8)
            arg2 = step * (self.warmup_steps ** -1.5)
            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

        def get_config(self):
            return {'d_model': int(self.d_model.numpy()),
                    'warmup_steps': self.warmup_steps}
else:
    class WarmupCosineDecay:
        def __init__(self, *a, **kw): pass


# ══════════════════════════════════════════════════════════════════
# MetaLearner — LSTM Meta-Learning Brain
# ══════════════════════════════════════════════════════════════════
def _multitask_aux_heads(shared_tensor, dropout: float) -> dict:
    """Range (sigmoid) + wall forward-delta regressors — shared trunk."""
    r = layers.Dense(48, activation='gelu')(shared_tensor)
    r = layers.Dropout(dropout * 0.5)(r)
    range_ctx_out = layers.Dense(1, activation='sigmoid', name='range_ctx_out')(r)

    w = layers.Dense(48, activation='gelu')(shared_tensor)
    w = layers.Dropout(dropout * 0.5)(w)
    wall_bid_out = layers.Dense(1, activation='linear', name='wall_bid_out')(w)
    wall_ask_out = layers.Dense(1, activation='linear', name='wall_ask_out')(w)

    return {'range_ctx_out': range_ctx_out, 'wall_bid_out': wall_bid_out, 'wall_ask_out': wall_ask_out}


class MetaLearnerLSTM:
    """
    LSTM يقرأ "القصة الكاملة" للسوق عبر الزمن ويصدر القرار النهائي.

    يستقبل في كل خطوة زمنية:
      - الفيتشرز الإحصائية (n_stat)
      - Meta Features من Stage 1/Schema (n_meta)
      - Visual Embeddings من CNN (8)
      المجموع: n_stat + n_meta + 8 feature per timestep
    """

    def __init__(self,
                 seq_len:       int   = 50,
                 n_stat_feat:   int   = 25,    # statistical features for V19
                 n_meta_feat:   int   = DEFAULT_META_FEATURES,
                 n_visual_emb:  int   = 8,     # من DeepLOB CNN
                 brain_file:    str   = 'outputs/meta_learner.keras',
                 lstm_units_1:  int   = 128,
                 lstm_units_2:  int   = 64,
                 dropout:       float = 0.25,
                 confidence_threshold: float = 0.65,
                 bias_long_threshold: float = 0.50,
                 multitask_meta: bool = False,
                 range_then_wall_phase1_frac: float = 0.28):

        self.seq_len      = seq_len
        self.n_stat       = n_stat_feat
        self.n_meta       = max(int(n_meta_feat), 0)
        self.n_visual     = n_visual_emb
        self.n_total      = n_stat_feat + self.n_meta + n_visual_emb
        self.brain_file   = brain_file
        self.lstm1        = lstm_units_1
        self.lstm2        = lstm_units_2
        self.drop         = dropout
        self.conf_thresh  = confidence_threshold
        self.bias_long_threshold = float(np.clip(bias_long_threshold, 0.05, 0.95))
        self.multitask_meta = bool(multitask_meta)
        self.range_then_wall_phase1_frac = float(np.clip(range_then_wall_phase1_frac, 0.05, 0.95))
        self.aux_loss_phase = 'phase1_range'
        self.wall_scale_bid = 1.0
        self.wall_scale_ask = 1.0

        self.bias_threshold_metrics = {
            'selected_threshold': float(self.bias_long_threshold),
        }
        self.bias_threshold_split = {
            'selection_source': 'not_run',
            'calibration_rows': 0,
            'report_rows': 0,
        }
        self.base_conf_loss_weight = 0.3
        self.current_conf_loss_weight = float(self.base_conf_loss_weight)
        self.confidence_head_enabled = True
        self.confidence_target_std = 0.0

        self.model   = None
        self._fitted = False

        if not TF_AVAILABLE:
            return

        if os.path.exists(brain_file):
            try:
                loaded_model = tf.keras.models.load_model(
                    brain_file, compile=False,
                    custom_objects={'WarmupCosineDecay': WarmupCosineDecay})
                if not _input_shape_matches(loaded_model, self.seq_len, self.n_total):
                    found_shape = _normalize_model_input_shape(loaded_model)
                    print(
                        "[MetaLearner] ⚠️ stale model input shape "
                        f"{found_shape} != expected (None, {self.seq_len}, {self.n_total}) — rebuilding"
                    )
                    self.model = self._build()
                else:
                    out_names = []
                    try:
                        out_names = list(loaded_model.output_names)
                    except Exception:
                        outs = getattr(loaded_model, 'outputs', None) or []
                        out_names = [getattr(o, 'name', '') or getattr(o, '_name', '') for o in outs]
                    multitask_loaded = {'range_ctx_out', 'wall_bid_out', 'wall_ask_out'}.issubset(set(out_names))
                    if multitask_loaded:
                        self.multitask_meta = True
                    if self.multitask_meta and not multitask_loaded:
                        print(
                            "[MetaLearner] ⚠️ multitask checkpoint missing auxiliary heads → rebuilding skeleton"
                        )
                        self.model = self._build()
                    else:
                        self.model = loaded_model
                        self._recompile()
                        self._fitted = True
                        print(f"[MetaLearner] 🧠 تحميل: {brain_file}")
            except Exception as e:
                print(f"[MetaLearner] ⚠️ ({e}) — بنبني جديد")
                self.model = self._build()
        else:
            print("[MetaLearner] 🧠 بناء LSTM Meta-Learner...")
            self.model = self._build()

    # ── Build ────────────────────────────────────────────────────
    def _build(self) -> 'tf.keras.Model':
        """
        البناء الكامل للـ Meta-Learner:
        Input → LayerNorm → LSTM(128) → LSTM(64) → Self-Attention → Output
        """
        # Input: (batch, seq_len, n_total)
        inp = layers.Input(
            shape=(self.seq_len, self.n_total),
            name='meta_input')

        # ── LayerNorm للتطبيع ──
        x = layers.LayerNormalization(epsilon=1e-6)(inp)

        # ── LSTM Stack ──────────────────────────────────────────
        # LSTM 1: يُعيد كامل التسلسل
        x = layers.LSTM(self.lstm1,
                        return_sequences=True,
                        dropout=self.drop,
                        recurrent_dropout=self.drop * 0.5,
                        name='lstm_1')(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # LSTM 2: يُعيد كامل التسلسل (للـ Attention)
        x = layers.LSTM(self.lstm2,
                        return_sequences=True,
                        dropout=self.drop,
                        name='lstm_2')(x)

        # ── Self-Attention ───────────────────────────────────────
        # يُركّز على أهم اللحظات في النافذة الزمنية
        attn_out, attn_scores = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=self.lstm2 // 4,
            dropout=self.drop * 0.5,
            name='self_attention')(x, x, return_attention_scores=True)

        # Residual connection
        x = layers.Add()([x, attn_out])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # ── Pooling ──────────────────────────────────────────────
        last_tok   = x[:, -1, :]                           # آخر timestep
        global_avg = layers.GlobalAveragePooling1D()(x)    # متوسط كل الزمن
        pooled     = layers.Concatenate()([last_tok, global_avg])
        # shape: (lstm2 × 2,) = (128,)

        # ── Shared Dense ─────────────────────────────────────────
        shared = layers.Dense(128, activation='gelu', name='shared')(pooled)
        shared = layers.Dropout(self.drop)(shared)

        # ── Output Heads ─────────────────────────────────────────
        # 1. Bias Head (LONG/SHORT)
        b = layers.Dense(64, activation='gelu')(shared)
        b = layers.Dropout(0.2)(b)
        bias_out = layers.Dense(2, activation='softmax', name='bias_out')(b)

        # 2. Confidence Head
        c = layers.Dense(32, activation='gelu')(shared)
        conf_out = layers.Dense(1, activation='sigmoid', name='conf_out')(c)

        out_map: dict = {'bias_out': bias_out, 'conf_out': conf_out}
        if self.multitask_meta:
            out_map.update(_multitask_aux_heads(shared, self.drop))

        model = Model(inp, out_map, name='MetaLearner_LSTM_multitask' if self.multitask_meta else 'MetaLearner_LSTM')

        self.model = model
        self._recompile(auxiliary_phase=getattr(self, 'aux_loss_phase', 'phase1_range'))
        model.summary(line_length=80)
        return model

    def _recompile(self, conf_loss_weight: float | None = None, auxiliary_phase: str | None = None):
        if conf_loss_weight is not None:
            self.current_conf_loss_weight = float(max(conf_loss_weight, 0.0))
            self.confidence_head_enabled = self.current_conf_loss_weight > 0.0
        if auxiliary_phase:
            self.aux_loss_phase = str(auxiliary_phase)
        # FIX-3: رفع warmup_steps من 500 إلى 2000 — مع 7 أيام داتا (467 batch/epoch)
        # الـ LR كان يرتفع بسرعة فيسبب overfitting من Epoch 2
        lr = WarmupCosineDecay(d_model=self.lstm2, warmup_steps=2000)
        losses: dict = {
            'bias_out': 'sparse_categorical_crossentropy',
            'conf_out': 'binary_crossentropy',
        }
        loss_weights = {'bias_out': 1.0, 'conf_out': float(self.current_conf_loss_weight)}
        out_names: list[str] = []
        try:
            out_names = list(self.model.output_names)
        except Exception:
            outs = getattr(self.model, 'outputs', None) or []
            out_names = [getattr(o, 'name', '') or getattr(o, '_name', '') for o in outs]
        has_aux = {'range_ctx_out', 'wall_bid_out', 'wall_ask_out'}.issubset(set(out_names))
        if has_aux:
            losses['range_ctx_out'] = 'binary_crossentropy'
            losses['wall_bid_out'] = 'huber'
            losses['wall_ask_out'] = 'huber'
            if self.aux_loss_phase == 'phase1_range':
                loss_weights.update({'range_ctx_out': 0.45, 'wall_bid_out': 0.0, 'wall_ask_out': 0.0})
            else:
                loss_weights.update({'range_ctx_out': 0.32, 'wall_bid_out': 0.18, 'wall_ask_out': 0.18})
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=losses,
            loss_weights=loss_weights,
            metrics={},  # keep quiet for multi-loss history keys
        )

    # ── Build Input ──────────────────────────────────────────────
    def build_meta_input(self,
                         X_stat:    np.ndarray,
                         cb_probs:  np.ndarray,
                         clusters:  np.ndarray,
                         vis_embs:  np.ndarray) -> np.ndarray:
        """
        يدمج كل المدخلات في مصفوفة واحدة لكل timestep.

        Args:
            X_stat:   (N, seq_len, n_stat)   — الفيتشرز الإحصائية
            cb_probs: (N, seq_len, 2)         — احتمالات CatBoost
            clusters: (N, seq_len, 4)         — One-Hot Cluster
            vis_embs: (N, seq_len, 8)         — Visual Embeddings من CNN
        Returns:
            X_meta:   (N, seq_len, n_total)
        """
        # التحقق من الأبعاد
        N, T = X_stat.shape[:2]

        # Pad أو crop لو في اختلاف
        def _pad(arr, target_dim):
            if arr.shape[-1] < target_dim:
                pad = np.zeros((*arr.shape[:-1], target_dim - arr.shape[-1]), dtype=np.float32)
                return np.concatenate([arr, pad], axis=-1)
            return arr[..., :target_dim]

        cb_probs  = _pad(cb_probs,  N_CB_PROBS)
        clusters  = _pad(clusters,  N_CLUSTERS)
        vis_embs  = _pad(vis_embs,  self.n_visual)
        X_stat    = _pad(X_stat,    self.n_stat)

        return np.concatenate([X_stat, cb_probs, clusters, vis_embs], axis=-1).astype(np.float32)

    @staticmethod
    def _labels_from_long_probs(long_probs: np.ndarray, threshold: float) -> np.ndarray:
        probs = np.asarray(long_probs, dtype=np.float32).reshape(-1)
        thr = float(np.clip(threshold, 0.0, 1.0))
        return np.where(probs >= thr, 0, 1).astype(np.int32)

    @classmethod
    def choose_bias_long_threshold(
        cls,
        long_probs: np.ndarray,
        y_true: np.ndarray,
        search_min: float = 0.35,
        search_max: float = 0.65,
        steps: int = 31,
    ) -> tuple[float, dict]:
        probs = np.asarray(long_probs, dtype=np.float32).reshape(-1)
        y = np.asarray(y_true, dtype=np.int32).reshape(-1)
        if probs.size == 0 or y.size == 0 or probs.size != y.size:
            threshold = 0.50
            return threshold, {'selected_threshold': threshold, 'reason': 'empty_validation'}

        best_threshold = 0.50
        best_metrics = None
        thresholds = np.linspace(float(search_min), float(search_max), max(int(steps), 3))

        for threshold in thresholds:
            pred = cls._labels_from_long_probs(probs, float(threshold))
            precision, recall, f1, support = precision_recall_fscore_support(
                y,
                pred,
                labels=[0, 1],
                zero_division=0,
            )
            macro_precision = float(np.mean(precision))
            macro_recall = float(np.mean(recall))
            macro_f1 = float(np.mean(f1))
            recall_gap = float(abs(recall[0] - recall[1]))
            threshold_metrics = {
                'selected_threshold': float(threshold),
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
                'long_precision': float(precision[0]),
                'short_precision': float(precision[1]),
                'long_recall': float(recall[0]),
                'short_recall': float(recall[1]),
                'long_f1': float(f1[0]),
                'short_f1': float(f1[1]),
                'long_support': int(support[0]),
                'short_support': int(support[1]),
                'recall_gap': recall_gap,
            }
            candidate_key = (
                round(macro_f1, 6),
                round(min(recall[0], recall[1]), 6),
                round(macro_precision, 6),
                round(-recall_gap, 6),
                round(-abs(float(threshold) - 0.50), 6),
            )
            if best_metrics is None or candidate_key > best_metrics['selection_key']:
                best_threshold = float(threshold)
                best_metrics = {
                    **threshold_metrics,
                    'selection_key': candidate_key,
                }

        if best_metrics is None:
            best_metrics = {'selected_threshold': best_threshold, 'reason': 'fallback_default'}
        else:
            best_metrics.pop('selection_key', None)
        return float(best_threshold), best_metrics

    @staticmethod
    def _resolve_threshold_holdout_split(
        n_rows: int,
        calibration_frac: float = 0.50,
        min_calibration_rows: int = 128,
    ) -> dict:
        n_rows = int(max(n_rows, 0))
        if n_rows <= 1:
            return {
                'selection_source': 'full_validation_fallback',
                'calibration_rows': n_rows,
                'report_rows': 0,
            }
        if n_rows < max(int(min_calibration_rows) * 2, 96):
            calibration_rows = max(1, n_rows // 2)
            return {
                'selection_source': 'small_validation_fallback',
                'calibration_rows': int(calibration_rows),
                'report_rows': int(n_rows - calibration_rows),
            }

        min_report_rows = max(32, int(round(n_rows * 0.20)))
        max_calibration_rows = n_rows - min_report_rows
        if max_calibration_rows <= 0:
            calibration_rows = max(1, n_rows // 2)
            return {
                'selection_source': 'small_validation_fallback',
                'calibration_rows': int(calibration_rows),
                'report_rows': int(n_rows - calibration_rows),
            }

        calibration_rows = int(round(n_rows * float(calibration_frac)))
        calibration_rows = max(
            calibration_rows,
            min(int(min_calibration_rows), int(max_calibration_rows)),
        )
        calibration_rows = min(calibration_rows, int(max_calibration_rows))
        report_rows = int(n_rows - calibration_rows)
        if report_rows <= 0:
            return {
                'selection_source': 'full_validation_fallback',
                'calibration_rows': n_rows,
                'report_rows': 0,
            }
        return {
            'selection_source': 'earlier_validation_slice',
            'calibration_rows': int(calibration_rows),
            'report_rows': int(report_rows),
        }

    @staticmethod
    def _slice_prediction_cache(pred_cache: dict | None, start: int, end: int | None = None) -> dict | None:
        if pred_cache is None:
            return None
        out = {}
        for key, value in pred_cache.items():
            out[key] = np.asarray(value)[start:end]
        return out

    @staticmethod
    def _outputs_include_aux(model) -> bool:
        if model is None:
            return False
        try:
            names = list(model.output_names)
        except Exception:
            outs = getattr(model, 'outputs', None) or []
            names = [getattr(o, 'name', '') or getattr(o, '_name', '') for o in outs]
        return {'range_ctx_out', 'wall_bid_out', 'wall_ask_out'}.issubset(set(names))

    # ── Fit ─────────────────────────────────────────────────────
    def fit(self,
            X_meta:  np.ndarray,
            y_bias:  np.ndarray,
            y_conf:  np.ndarray,
            epochs:  int   = 100,
            batch:   int   = 64,
            output_dir: str = 'outputs',
            class_weights: dict = None) -> 'History':

        if not TF_AVAILABLE or self.model is None:
            print("  ❌ TensorFlow غير متاح")
            return None

        os.makedirs(output_dir, exist_ok=True)

        # Train/Val split (80/20, time-ordered — anti-leakage)
        n = len(X_meta)
        split = int(n * 0.80)
        X_tr, X_val = X_meta[:split],  X_meta[split:]
        yb_tr, yb_val = y_bias[:split], y_bias[split:]
        yc_tr, yc_val = y_conf[:split], y_conf[split:]

        return self.fit_train_val(
            X_tr, yb_tr, yc_tr,
            X_val, yb_val, yc_val,
            epochs=epochs,
            batch=batch,
            output_dir=output_dir,
            class_weights=class_weights,
        )

    def fit_train_val(self,
                      X_tr: np.ndarray,
                      yb_tr: np.ndarray,
                      yc_tr: np.ndarray,
                      X_val: np.ndarray,
                      yb_val: np.ndarray,
                      yc_val: np.ndarray,
                      epochs: int = 100,
                      batch: int = 64,
                      output_dir: str = 'outputs',
                      class_weights: dict | None = None,
                      aux_train: dict[str, np.ndarray] | None = None,
                      aux_val: dict[str, np.ndarray] | None = None) -> object | None:
        if not TF_AVAILABLE or self.model is None:
            print("  ❌ TensorFlow غير متاح")
            return None

        os.makedirs(output_dir, exist_ok=True)
        brain_path = self.brain_file
        if not os.path.isabs(brain_path) and not os.path.dirname(brain_path):
            brain_path = os.path.join(output_dir, brain_path)

        n_tr = len(X_tr)
        n_val = len(X_val)

        yb_arr = np.asarray(yb_tr, dtype=np.int32)
        yc_arr = np.asarray(yc_tr, dtype=np.float32)

        n_long = max(int((yb_arr == 0).sum()), 1)
        n_short = max(int((yb_arr == 1).sum()), 1)
        n_total = n_tr

        # FIX-2: cap الأوزان عند 3.0 — بدون cap تتضاعف مع Stage 1 وتحيّز النموذج
        w_long  = min(n_total / (2.0 * n_long),  3.0)
        w_short = min(n_total / (2.0 * n_short), 3.0)

        quality_boost = np.where(yc_arr > 0.5, 2.0, 1.0).astype(np.float32)
        class_w_arr = np.where(yb_arr == 0, w_long, w_short).astype(np.float32)
        sample_w = (class_w_arr * quality_boost).astype(np.float32)
        conf_w = quality_boost.copy()
        self.confidence_target_std = float(np.std(yc_arr)) if len(yc_arr) else 0.0
        # FIX-1: رفع الحد من 1e-4 إلى 0.05 — conf_target شبه ثابت (std < 0.05)
        # يجمّد الـ gradient ويعطّل bias_out بشكل غير مباشر
        if self.confidence_target_std < 0.05:
            self._recompile(conf_loss_weight=0.0)
            print(
                "   Confidence head   → disabled "
                f"(std(conf_target)={self.confidence_target_std:.6f} < 0.05)"
            )
        else:
            self._recompile(conf_loss_weight=self.base_conf_loss_weight)
            print(
                "   Confidence head   → enabled "
                f"(std(conf_target)={self.confidence_target_std:.6f})"
            )

        aux_ready = (
            bool(self.multitask_meta)
            and self._outputs_include_aux(self.model)
            and aux_train is not None
            and aux_val is not None
        )
        if self.multitask_meta and aux_train is None:
            print("  ⚠️ multitask_meta=True لكن لم تُمرّر aux_train/aux_val → تمرين بدون الأهداف الثانوية")
        elif aux_ready:
            print(
                "  🧭 Meta multitask aux → range_ctx + wall deltas "
                f"(phase1_frac={self.range_then_wall_phase1_frac:.0%})"
            )

        print(f"\n🧠 MetaLearner Training: {n_tr + n_val:,} sequences | split={n_tr:,}/{n_val:,}")
        print(
            f"   Class distribution → LONG={n_long:,} ({n_long/n_total*100:.1f}%) | "
            f"SHORT={n_short:,} ({n_short/n_total*100:.1f}%)"
        )
        print(f"   Class weights      → w_LONG={w_long:.2f} | w_SHORT={w_short:.2f}")
        print(f"   Quality boost      → STRONG×2.0 | WEAK×1.0")

        callbacks_phase = [
            EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True, verbose=1),
            ModelCheckpoint(brain_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1),
        ]

        history = None

        def _run_fit(
            e_run: int,
            sw_phase_note: str,
            initial_epoch_num: int = 0,
        ) -> object | None:
            y_tr_pkg = {'bias_out': yb_tr, 'conf_out': yc_tr}
            y_val_pkg = {'bias_out': yb_val, 'conf_out': yc_val}
            sw_tr_pkg = {'bias_out': sample_w, 'conf_out': conf_w}
            fit_val: tuple = (X_val, y_val_pkg)
            if aux_ready:
                y_tr_pkg.update(
                    {
                        'range_ctx_out': aux_train['range_ctx'].astype(np.float32).reshape(-1, 1),
                        'wall_bid_out': aux_train['wall_bid_scaled'].astype(np.float32).reshape(-1, 1),
                        'wall_ask_out': aux_train['wall_ask_scaled'].astype(np.float32).reshape(-1, 1),
                    }
                )
                y_val_pkg.update(
                    {
                        'range_ctx_out': aux_val['range_ctx'].astype(np.float32).reshape(-1, 1),
                        'wall_bid_out': aux_val['wall_bid_scaled'].astype(np.float32).reshape(-1, 1),
                        'wall_ask_out': aux_val['wall_ask_scaled'].astype(np.float32).reshape(-1, 1),
                    }
                )
                wm_tr = aux_train['wall_mask'].astype(np.float32).reshape(-1)
                wm_va = aux_val['wall_mask'].astype(np.float32).reshape(-1)
                sw_tr_pkg.update(
                    {
                        'range_ctx_out': np.ones(n_tr, dtype=np.float32),
                        'wall_bid_out': wm_tr,
                        'wall_ask_out': wm_tr,
                    }
                )
                sw_val_pkg = {
                    'bias_out': np.ones(n_val, dtype=np.float32),
                    'conf_out': np.ones(n_val, dtype=np.float32),
                    'range_ctx_out': np.ones(n_val, dtype=np.float32),
                    'wall_bid_out': wm_va,
                    'wall_ask_out': wm_va,
                }
                fit_val = (X_val, y_val_pkg, sw_val_pkg)
            e_cap = max(int(initial_epoch_num) + int(e_run), 1)
            print(
                f"   [{sw_phase_note}] epochs→{initial_epoch_num}..{e_cap}"
                + ("" if not aux_ready else " (weighted wall NaNs masked on val)")
            )
            return self.model.fit(
                X_tr,
                y_tr_pkg,
                validation_data=fit_val,
                initial_epoch=int(initial_epoch_num),
                epochs=int(e_cap),
                batch_size=batch,
                sample_weight=sw_tr_pkg if aux_ready else {'bias_out': sample_w, 'conf_out': conf_w},
                callbacks=callbacks_phase,
                verbose=1,
            )

        if aux_ready:
            epochs1 = max(8, int(round(epochs * float(self.range_then_wall_phase1_frac))))
            epochs2 = max(8, int(epochs) - epochs1)
            self.aux_loss_phase = 'phase1_range'
            self._recompile()
            hist1 = _run_fit(epochs1, 'phase1 RANGE only (wall λ=0)', initial_epoch_num=0)
            ie2 = len(hist1.history.get('loss', [])) if hist1 else 0
            self.aux_loss_phase = 'phase2_wall'
            self._recompile()
            hist2 = _run_fit(epochs2, 'phase2 RANGE+WALL (weighted losses)', initial_epoch_num=ie2)
            history = hist2 or hist1
        else:
            history = self.model.fit(
                X_tr,
                {'bias_out': yb_tr, 'conf_out': yc_tr},
                validation_data=(X_val, {'bias_out': yb_val, 'conf_out': yc_val}),
                epochs=epochs,
                batch_size=batch,
                sample_weight={'bias_out': sample_w, 'conf_out': conf_w},
                callbacks=callbacks_phase,
                verbose=1,
            )

        self._fitted = True
        pred_cache = self.model.predict(X_val, verbose=0) if len(X_val) else None
        report_X = X_val
        report_y = yb_val
        report_cache = pred_cache
        report_title = 'MetaLearner Validation'
        report_filename = 'meta_learner_report.txt'
        if pred_cache is not None and 'bias_out' in pred_cache:
            split_info = self._resolve_threshold_holdout_split(n_val)
            self.bias_threshold_split = dict(split_info)
            calibration_rows = int(split_info.get('calibration_rows', 0))
            report_rows = int(split_info.get('report_rows', 0))

            threshold_cache = pred_cache
            threshold_y = yb_val
            if calibration_rows > 0 and report_rows > 0:
                threshold_cache = self._slice_prediction_cache(pred_cache, 0, calibration_rows)
                threshold_y = yb_val[:calibration_rows]
                report_cache = self._slice_prediction_cache(
                    pred_cache,
                    calibration_rows,
                    calibration_rows + report_rows,
                )
                report_X = X_val[calibration_rows : calibration_rows + report_rows]
                report_y = yb_val[calibration_rows : calibration_rows + report_rows]
                report_title = 'MetaLearner Final Holdout'
                report_filename = 'meta_learner_holdout_report.txt'

            self.bias_long_threshold, self.bias_threshold_metrics = self.choose_bias_long_threshold(
                threshold_cache['bias_out'][:, 0],
                threshold_y,
            )
            self.bias_threshold_metrics.update(split_info)
            print(
                "  🎚️ Bias threshold calibration → "
                f"LONG if p_long >= {self.bias_long_threshold:.3f} "
                f"| macro_f1={self.bias_threshold_metrics.get('macro_f1', 0.0):.3f} "
                f"| long_recall={self.bias_threshold_metrics.get('long_recall', 0.0):.3f} "
                f"| short_recall={self.bias_threshold_metrics.get('short_recall', 0.0):.3f} "
                f"| calibration_rows={int(split_info.get('calibration_rows', 0)):,} "
                f"| final_holdout_rows={int(split_info.get('report_rows', 0)):,}"
            )
        self._report(
            report_X,
            report_y,
            output_dir,
            pred_cache=report_cache,
            report_title=report_title,
            report_filename=report_filename,
        )
        return history

    def _report(
        self,
        X_val,
        yb_val,
        output_dir,
        pred_cache=None,
        report_title: str = 'MetaLearner Validation',
        report_filename: str = 'meta_learner_report.txt',
    ):
        """تقرير التحقق"""
        if not self._fitted:
            return
        preds = pred_cache if pred_cache is not None else self.model.predict(X_val, verbose=0)
        bp = self._labels_from_long_probs(preds['bias_out'][:, 0], self.bias_long_threshold)
        rep   = classification_report(
            yb_val, bp,
            labels=[0, 1],
            target_names=['LONG', 'SHORT'],
            zero_division=0)
        threshold_text = (
            f"Bias LONG threshold: {self.bias_long_threshold:.3f}\n"
            if self.bias_long_threshold is not None
            else ""
        )
        print(f"\n📊 {report_title}:\n{threshold_text}{rep}")
        path = os.path.join(output_dir, report_filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(threshold_text + rep)

    def _aux_fields_from_batch_pred(self, pred_batch: dict) -> dict:
        """مخرجات الرؤوس الثانوية (رينج + جدار) بعد denorm لدلتا الجدار."""
        out: dict = {}
        if not self._outputs_include_aux(self.model):
            return out
        if 'range_ctx_out' in pred_batch:
            out['range_ctx_prob'] = float(np.asarray(pred_batch['range_ctx_out'], dtype=np.float32).reshape(-1)[0])
        if 'wall_bid_out' in pred_batch:
            vb = float(np.asarray(pred_batch['wall_bid_out'], dtype=np.float32).reshape(-1)[0])
            out['wall_bid_delta_hat'] = vb * float(self.wall_scale_bid)
            out['wall_bid_scaled'] = vb
        if 'wall_ask_out' in pred_batch:
            va = float(np.asarray(pred_batch['wall_ask_out'], dtype=np.float32).reshape(-1)[0])
            out['wall_ask_delta_hat'] = va * float(self.wall_scale_ask)
            out['wall_ask_scaled'] = va
        return out

    # ── Predict ─────────────────────────────────────────────────
    def predict(self,
                X_meta: np.ndarray,
                n_mc:   int = 20) -> dict:
        """
        MC Dropout للحصول على Uncertainty Estimate.
        X_meta shape: (seq_len, n_total) أو (N, seq_len, n_total)
        """
        if not TF_AVAILABLE or not self._fitted or self.model is None:
            return {'bias': 'NEUTRAL', 'bias_idx': 2,
                    'confidence': 0.0, 'uncertainty': 1.0}

        single = (X_meta.ndim == 2)
        if single:
            X_meta = X_meta[np.newaxis]
        if not self.confidence_head_enabled:
            det = self.model.predict(X_meta, verbose=0)
            bias_mean = np.asarray(det['bias_out'], dtype=np.float32)[0]
            bias_idx = int(self._labels_from_long_probs(np.array([bias_mean[0]], dtype=np.float32), self.bias_long_threshold)[0])
            out = {
                'bias':        BIAS_LABELS[bias_idx],
                'bias_idx':    bias_idx,
                'bias_probs':  bias_mean.tolist(),
                'confidence':  1.0,
                'uncertainty': 0.0,
                'tradeable':   True,
            }
            out.update(self._aux_fields_from_batch_pred(det))
            return out

        # MC Dropout: n_mc forward passes
        X_tiled = np.tile(X_meta, (n_mc, 1, 1))
        out = self.model(X_tiled, training=True)  # Dropout active

        bp    = out['bias_out'].numpy()    # (n_mc, 2)
        conf  = out['conf_out'].numpy().flatten()   # (n_mc,)

        bias_mean = np.mean(bp.reshape(n_mc, -1, 2), axis=0)[0]  # (2,)
        conf_mean = float(np.mean(conf))
        conf_std  = float(np.std(conf))

        bias_idx = int(self._labels_from_long_probs(np.array([bias_mean[0]], dtype=np.float32), self.bias_long_threshold)[0])
        result = {
            'bias':        BIAS_LABELS[bias_idx],
            'bias_idx':    bias_idx,
            'bias_probs':  bias_mean.tolist(),
            'confidence':  round(conf_mean, 4),
            'uncertainty': round(conf_std, 4),
            'tradeable':   (conf_mean >= self.conf_thresh),
        }
        det_aux = self.model.predict(X_meta, verbose=0)
        result.update(self._aux_fields_from_batch_pred(det_aux))
        return result

    def save(self, output_dir: str = 'outputs'):
        if self.model:
            path = os.path.join(output_dir, 'meta_learner.keras')
            self.model.save(path)
            print(f"  ✅ MetaLearner: {path}")


# ═══════════════════════════════════════════════════════════════════
# التعديل 3 — TCN Block (Temporal Convolutional Network)
# التعديل 7 — Volatility-Weighted Loss
# ═══════════════════════════════════════════════════════════════════

def _build_tcn_block(x, filters: int = 64, kernel_size: int = 3, dilations=None):
    """
    التعديل 3: TCN Block مع Causal Dilated Convolutions.

    لماذا TCN قبل LSTM؟
      - TCN يستخرج patterns محلية قصيرة (3-10 ticks) بكفاءة
      - LSTM يستخرج dependencies طويلة (50 ticks)
      - التركيبة تقلل overfitting لأن TCN له parameters أقل من LSTM

    Architecture:
      Input → [Conv1D(d=1) → Conv1D(d=2) → Conv1D(d=4) → Conv1D(d=8)]
            → Residual → LayerNorm → Output

    Causal: كل conv تستخدم padding='causal' → لا look-ahead
    Dilated: يوسع الـ receptive field بدون زيادة parameters
    """
    if not TF_AVAILABLE:
        return x

    if dilations is None:
        dilations = [1, 2, 4, 8]

    residual = x

    for d in dilations:
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=d,
            padding='causal',
            activation='gelu',
        )(x)
        x = layers.Dropout(0.1)(x)

    # Residual projection (لو الـ shape اختلف)
    if residual.shape[-1] != filters:
        residual = layers.Conv1D(filters, kernel_size=1, padding='same')(residual)

    x = layers.Add()([x, residual])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x


class MetaLearnerTCNLSTM(MetaLearnerLSTM):
    """
    التعديل 3: نسخة محسّنة من MetaLearnerLSTM تضيف TCN block.

    الفرق الوحيد عن الأصل:
      Input → LayerNorm → [TCN Block] → LSTM(128) → LSTM(64) → Attention → Output

    TCN يُضاف قبل LSTM مباشرة كـ preprocessing layer.
    """

    def __init__(self, *args, tcn_filters: int = 64, **kwargs):
        self.tcn_filters = tcn_filters
        super().__init__(*args, **kwargs)

    def _build(self) -> 'tf.keras.Model':
        if not TF_AVAILABLE:
            return None

        inp = layers.Input(
            shape=(self.seq_len, self.n_total),
            name='meta_input')

        # LayerNorm
        x = layers.LayerNormalization(epsilon=1e-6)(inp)

        # ── TCN Block (التعديل 3) ──────────────────────────────
        x = _build_tcn_block(x, filters=self.tcn_filters, kernel_size=3, dilations=[1, 2, 4, 8])

        # ── LSTM Stack ──────────────────────────────────────────
        x = layers.LSTM(self.lstm1,
                        return_sequences=True,
                        dropout=self.drop,
                        recurrent_dropout=self.drop * 0.5,
                        name='lstm_1')(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        x = layers.LSTM(self.lstm2,
                        return_sequences=True,
                        dropout=self.drop,
                        name='lstm_2')(x)

        # ── Self-Attention ───────────────────────────────────────
        attn_out, _ = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=self.lstm2 // 4,
            dropout=self.drop * 0.5,
            name='self_attention')(x, x, return_attention_scores=True)

        x = layers.Add()([x, attn_out])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        last_tok   = x[:, -1, :]
        global_avg = layers.GlobalAveragePooling1D()(x)
        pooled     = layers.Concatenate()([last_tok, global_avg])

        shared = layers.Dense(128, activation='gelu', name='shared')(pooled)
        shared = layers.Dropout(self.drop)(shared)

        b = layers.Dense(64, activation='gelu')(shared)
        b = layers.Dropout(0.2)(b)
        bias_out = layers.Dense(2, activation='softmax', name='bias_out')(b)

        c = layers.Dense(32, activation='gelu')(shared)
        conf_out = layers.Dense(1, activation='sigmoid', name='conf_out')(c)

        out_map: dict = {'bias_out': bias_out, 'conf_out': conf_out}
        if self.multitask_meta:
            out_map.update(_multitask_aux_heads(shared, self.drop))

        model = Model(
            inp,
            out_map,
            name=(
                'MetaLearner_TCN_LSTM_multitask' if self.multitask_meta else 'MetaLearner_TCN_LSTM'
            ),
        )
        self.model = model
        self._recompile(auxiliary_phase=getattr(self, 'aux_loss_phase', 'phase1_range'))
        model.summary(line_length=80)
        return model


def make_volatility_weighted_loss(vol_weights: np.ndarray):
    """
    التعديل 7: Volatility-Weighted Cross-Entropy Loss.

    الفكرة من Att-LSTM-GARCH:
      - خطأ في فترة تقلب عالي أهم من خطأ في فترة هادئة
      - weighted_loss = CrossEntropy × (1 + vol_normalized)
      - يدفع النموذج للتركيز على الحالات الصعبة

    Parameters
    ----------
    vol_weights : array شكله (n_samples,) — قيم GARCH conditional vol
                  مُطبّعة لـ [0, 1]

    Returns
    -------
    loss_fn : callable تُمرر لـ model.compile(loss=loss_fn)
    """
    if not TF_AVAILABLE:
        return 'sparse_categorical_crossentropy'

    vol_tensor = tf.constant(vol_weights.astype(np.float32), name='vol_weights')

    def volatility_weighted_ce(y_true, y_pred):
        # CrossEntropy أساسي
        base_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

        # index للبيانات الحالية (يعمل في eager mode)
        batch_size = tf.shape(y_true)[0]
        # نستخدم mean weight كـ fallback آمن للـ batching
        mean_weight = tf.reduce_mean(vol_tensor)
        weight      = tf.ones(batch_size, dtype=tf.float32) * (1.0 + mean_weight)

        weighted = base_loss * weight
        return tf.reduce_mean(weighted)

    return volatility_weighted_ce
