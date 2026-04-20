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

from sklearn.metrics import classification_report

BIAS_LABELS  = {0: 'LONG', 1: 'SHORT', 2: 'NEUTRAL'}
N_CLUSTERS   = 4
N_CB_PROBS   = 2   # P_LONG, P_SHORT
N_REGIME_SCORES = 3


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
class MetaLearnerLSTM:
    """
    LSTM يقرأ "القصة الكاملة" للسوق عبر الزمن ويصدر القرار النهائي.

    يستقبل في كل خطوة زمنية:
      - الفيتشرز الإحصائية (n_stat)
      - احتمالات CatBoost (2)
      - One-Hot Cluster (4)
      - Soft Regime Scores (3)
      - Visual Embeddings من CNN (8)
      المجموع: n_stat + 2 + 4 + 3 + 8 feature per timestep
    """

    def __init__(self,
                 seq_len:       int   = 50,
                 n_stat_feat:   int   = 25,    # statistical features for V19
                 n_visual_emb:  int   = 8,     # من DeepLOB CNN
                 brain_file:    str   = 'outputs/meta_learner.keras',
                 lstm_units_1:  int   = 128,
                 lstm_units_2:  int   = 64,
                 dropout:       float = 0.25,
                 confidence_threshold: float = 0.65):

        self.seq_len      = seq_len
        self.n_stat       = n_stat_feat
        self.n_visual     = n_visual_emb
        self.n_meta       = N_CB_PROBS + N_CLUSTERS + N_REGIME_SCORES
        self.n_total      = n_stat_feat + self.n_meta + n_visual_emb
        self.brain_file   = brain_file
        self.lstm1        = lstm_units_1
        self.lstm2        = lstm_units_2
        self.drop         = dropout
        self.conf_thresh  = confidence_threshold

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

        model = Model(inp,
                      {'bias_out': bias_out, 'conf_out': conf_out},
                      name='MetaLearner_LSTM')

        self.model = model
        self._recompile()
        model.summary(line_length=80)
        return model

    def _recompile(self):
        lr = WarmupCosineDecay(d_model=self.lstm2, warmup_steps=500)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss={
                'bias_out': 'sparse_categorical_crossentropy',
                'conf_out': 'binary_crossentropy',
            },
            loss_weights={'bias_out': 1.0, 'conf_out': 0.3},
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
                      class_weights: dict = None) -> 'History':
        if not TF_AVAILABLE or self.model is None:
            print("  ❌ TensorFlow غير متاح")
            return None

        os.makedirs(output_dir, exist_ok=True)
        brain_path = self.brain_file
        if not os.path.isabs(brain_path) and not os.path.dirname(brain_path):
            brain_path = os.path.join(output_dir, brain_path)

        sample_w = np.where(np.asarray(yc_tr, dtype=np.float32) > 0.5, 2.0, 1.0).astype(np.float32)
        conf_w = sample_w.copy()
        n_tr = len(X_tr)
        n_val = len(X_val)

        print(f"\n🧠 MetaLearner Training: {n_tr + n_val:,} sequences | split={n_tr:,}/{n_val:,}")
        print("   Quality Weights: STRONG=2.00 WEAK=1.00")

        cbs = [
            EarlyStopping(monitor='val_loss',
                          patience=15, mode='min',
                          restore_best_weights=True, verbose=1),
            ModelCheckpoint(brain_path,
                            monitor='val_loss',
                            save_best_only=True, mode='min', verbose=1),
        ]

        history = self.model.fit(
            X_tr,
            {'bias_out': yb_tr, 'conf_out': yc_tr},
            validation_data=(
                X_val,
                {'bias_out': yb_val, 'conf_out': yc_val}),
            epochs=epochs,
            batch_size=batch,
            sample_weight={
                'bias_out': sample_w,
                'conf_out': conf_w,
            },
            callbacks=cbs,
            verbose=1,
        )

        self._fitted = True
        self._report(X_val, yb_val, output_dir)
        return history

    def _report(self, X_val, yb_val, output_dir):
        """تقرير التحقق"""
        if not self._fitted:
            return
        preds = self.model.predict(X_val, verbose=0)
        bp    = np.argmax(preds['bias_out'], axis=1)
        rep   = classification_report(
            yb_val, bp,
            labels=[0, 1],
            target_names=['LONG', 'SHORT'],
            zero_division=0)
        print(f"\n📊 MetaLearner Validation:\n{rep}")
        path = os.path.join(output_dir, 'meta_learner_report.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(rep)

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

        # MC Dropout: n_mc forward passes
        X_tiled = np.tile(X_meta, (n_mc, 1, 1))
        out = self.model(X_tiled, training=True)  # Dropout active

        bp    = out['bias_out'].numpy()    # (n_mc, 2)
        conf  = out['conf_out'].numpy().flatten()   # (n_mc,)

        bias_mean = np.mean(bp.reshape(n_mc, -1, 2), axis=0)[0]  # (2,)
        conf_mean = float(np.mean(conf))
        conf_std  = float(np.std(conf))

        bias_idx = int(np.argmax(bias_mean))
        return {
            'bias':        BIAS_LABELS[bias_idx],
            'bias_idx':    bias_idx,
            'bias_probs':  bias_mean.tolist(),
            'confidence':  round(conf_mean, 4),
            'uncertainty': round(conf_std, 4),
            'tradeable':   (conf_mean >= self.conf_thresh),
        }

    def save(self, output_dir: str = 'outputs'):
        if self.model:
            path = os.path.join(output_dir, 'meta_learner.keras')
            self.model.save(path)
            print(f"  ✅ MetaLearner: {path}")
