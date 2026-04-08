"""
deeplob_cnn.py — DeepLOB Visual Engine (V19)
═══════════════════════════════════════════════════════════════════════
يبني "صورة رقمية" ثلاثية الأبعاد لعمق السوق:

  Tensor shape: (time_steps=50, price_levels=20, channels=3)
    Channel 0: MBP10 Depth Map   — حجم الأوامر المعلقة (bid+ask بـ 10 مستوى)
    Channel 1: Buy Footprint     — حجم الشراء الماركت (Buy Aggressors)
    Channel 2: Sell Footprint    — حجم البيع الماركت (Sell Aggressors)

  CNN Architecture (DeepLOB-style):
    Conv2D(32, 1×2, stride 1×2)  ← يدمج كل سعر مع حجمه فقط
    Conv2D(32, 4×1)              ← patterns زمنية قصيرة
    Conv2D(32, 4×1)              ← patterns زمنية متوسطة
    Inception Module ×3          ← patterns متعددة الأطوال
    Dense(8)                     ← الضغط لـ 8 Visual Embeddings

المخرج: visual_emb (8,) — بصمة بصرية تلخص حالة الـ book
═══════════════════════════════════════════════════════════════════════
"""

import numpy as np
import os
import pickle
from collections import deque

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("  ⚠️  TensorFlow غير مثبّت — DeepLOB CNN غير متاح")

# ── ثوابت ────────────────────────────────────────────────────────
N_TIME_STEPS   = 50    # طول النافذة الزمنية
N_PRICE_LEVELS = 20    # 10 bid + 10 ask → 20 مستوى سعري
N_CHANNELS     = 3     # Depth + Buy_FP + Sell_FP
VISUAL_EMB_DIM = 8     # بُعد الـ Visual Embeddings


# ══════════════════════════════════════════════════════════════════
# 1. TensorBuilder — يبني الـ 3D Tensor tick-by-tick
# ══════════════════════════════════════════════════════════════════
class LOBTensorBuilder:
    """
    يبني الـ 3D Tensor للـ DeepLOB من بيانات MBO + MBP10.

    كل snapshot يُضاف لنافذة زمنية منزلقة (rolling window).
    الـ Normalization: Rolling Z-Score على آخر z_window snapshot
    لتجنب look-ahead bias وتكيّف مع تقلب السوق.

    Usage:
        builder = LOBTensorBuilder()
        for tick in stream:
            builder.update_mbp(mbp_snapshot)
            builder.update_trade(price, size, is_buy)
            tensor = builder.get_tensor()  # (50, 20, 3) أو None
    """

    def __init__(self,
                 time_steps:   int = N_TIME_STEPS,
                 price_levels: int = N_PRICE_LEVELS,
                 z_window:     int = 200):
        self.T  = time_steps
        self.P  = price_levels   # 10 bid + 10 ask
        self.L  = N_PRICE_LEVELS // 2   # 10 levels per side
        self.z_window = z_window

        # Buffers للـ snapshots
        self._depth_buf  = deque(maxlen=time_steps)   # channel 0
        self._buy_buf    = deque(maxlen=time_steps)   # channel 1
        self._sell_buf   = deque(maxlen=time_steps)   # channel 2

        # Rolling Z-Score normalization windows
        self._norm_depth = deque(maxlen=z_window)
        self._norm_buy   = deque(maxlen=z_window)
        self._norm_sell  = deque(maxlen=z_window)

        # آخر snapshot MBP محفوظ
        self._last_mbp   = None
        # footprint للـ snapshot الحالية (تُصفَّر عند كل update_mbp)
        self._cur_buy_fp  = np.zeros(self.L, dtype=np.float32)
        self._cur_sell_fp = np.zeros(self.L, dtype=np.float32)
        self._cur_ref_price = 0.0

    def update_mbp(self, row: dict) -> None:
        """
        يستلم snapshot من MBP10 ويحسب depth channel.
        row: dict مثل {'bid_px_00': 1.2505, 'bid_sz_00': 10, ...}
        """
        self._last_mbp = row
        L = self.L

        # استخراج mid price للـ price grid
        bid0 = float(row.get('bid_px_00', 0) or 0)
        ask0 = float(row.get('ask_px_00', 0) or 0)
        if bid0 > 0 and ask0 > 0:
            self._cur_ref_price = (bid0 + ask0) / 2.0
        elif bid0 > 0:
            self._cur_ref_price = bid0
        elif ask0 > 0:
            self._cur_ref_price = ask0

        # Depth channel: [bid_sz_09..bid_sz_00, ask_sz_00..ask_sz_09]
        # المستوى الأقرب للمنتصف في المركز
        depth = np.zeros(self.P, dtype=np.float32)
        for i in range(L):
            # bid levels: index 0..9 (L-1-i للعكس: L0 في المنتصف)
            depth[L - 1 - i] = float(row.get(f'bid_sz_0{i}', 0) or 0)
            # ask levels: index 10..19 (i0 في المنتصف)
            depth[L + i]     = float(row.get(f'ask_sz_0{i}', 0) or 0)

        # تسجيل الـ snapshot وتصفير الـ footprint للـ bar الجديد
        self._depth_buf.append(depth)
        self._norm_depth.extend(depth.tolist())

        # احتساب الـ footprint الذي تراكم منذ آخر snapshot
        self._depth_buf[-1]  = depth
        buy_fp  = self._cur_buy_fp.copy()
        sell_fp = self._cur_sell_fp.copy()

        # Pad للـ full 20-level format (bid=0..9, ask=10..19)
        buy_fp_full  = np.zeros(self.P, dtype=np.float32)
        sell_fp_full = np.zeros(self.P, dtype=np.float32)
        buy_fp_full[L:]  = buy_fp   # buy aggressors → ask side
        sell_fp_full[:L] = sell_fp  # sell aggressors → bid side

        self._buy_buf.append(buy_fp_full)
        self._sell_buf.append(sell_fp_full)
        self._norm_buy.extend(buy_fp_full.tolist())
        self._norm_sell.extend(sell_fp_full.tolist())

        # تصفير الـ footprint للـ snapshot القادمة
        self._cur_buy_fp  = np.zeros(self.L, dtype=np.float32)
        self._cur_sell_fp = np.zeros(self.L, dtype=np.float32)

    def update_trade(self, price: float, size: int, is_buy: bool) -> None:
        """
        يستلم trade tick ويضيفه للـ footprint الحالي.
        يُحدد مستوى السعر بناءً على المسافة من الـ mid price.
        """
        if self._last_mbp is None or self._cur_ref_price <= 0:
            return

        L = self.L
        bid0 = float(self._last_mbp.get('bid_px_00', 0) or 0)
        ask0 = float(self._last_mbp.get('ask_px_00', 0) or 0)
        if bid0 <= 0 or ask0 <= 0:
            return

        tick = ask0 - bid0
        if tick <= 0:
            tick = abs(price * 0.0001) + 1e-8

        # map price → level index (0 = أقرب للـ mid)
        if is_buy:
            # Buy Aggressor: يضرب الـ ask → ask side (levels 0..L-1 من الـ ask)
            dist = (price - ask0) / tick
            idx  = int(round(abs(dist)))
            idx  = min(max(idx, 0), L - 1)
            self._cur_buy_fp[idx] += size
        else:
            # Sell Aggressor: يضرب الـ bid → bid side
            dist = (bid0 - price) / tick
            idx  = int(round(abs(dist)))
            idx  = min(max(idx, 0), L - 1)
            self._cur_sell_fp[idx] += size

    def get_tensor(self) -> np.ndarray:
        """
        يُعيد الـ 3D Tensor (T, P, 3) أو None لو البيانات غير كافية.
        يُطبَّق Rolling Z-Score normalization لكل channel.
        """
        if len(self._depth_buf) < self.T:
            return None

        # بناء المصفوفات
        depth_arr = np.array(list(self._depth_buf), dtype=np.float32)   # (T, P)
        buy_arr   = np.array(list(self._buy_buf),   dtype=np.float32)   # (T, P)
        sell_arr  = np.array(list(self._sell_buf),  dtype=np.float32)   # (T, P)

        # Rolling Z-Score normalization
        depth_arr = _zscore_normalize(depth_arr, self._norm_depth)
        buy_arr   = _zscore_normalize(buy_arr,   self._norm_buy)
        sell_arr  = _zscore_normalize(sell_arr,  self._norm_sell)

        # Stack → (T, P, 3)
        tensor = np.stack([depth_arr, buy_arr, sell_arr], axis=-1)
        return tensor.astype(np.float32)

    def get_state(self) -> dict:
        """تصدير الحالة الداخلية للـ multiprocessing"""
        return {
            'depth': list(self._depth_buf),
            'buy':   list(self._buy_buf),
            'sell':  list(self._sell_buf),
            'ref_price': self._cur_ref_price,
        }


def _zscore_normalize(arr: np.ndarray, history: deque) -> np.ndarray:
    """Rolling Z-Score: mean/std من الـ history window"""
    if len(history) < 20:
        # fallback: min-max على الـ arr نفسها
        mx = arr.max()
        if mx > 1e-8:
            return (arr / mx).clip(0, 1)
        return arr

    h = np.array(list(history), dtype=np.float32)
    h = h[h > 0]  # استبعاد الأصفار من الـ stats
    if len(h) < 10:
        return arr

    mu  = float(np.mean(h))
    std = float(np.std(h)) + 1e-8
    return np.clip((arr - mu) / std, -4.0, 4.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# 2. DeepLOBCNN — المعمارية البصرية
# ══════════════════════════════════════════════════════════════════
class DeepLOBCNN:
    """
    CNN يستوعب الـ 3D Tensor ويُخرج 8 Visual Embeddings.

    المعمارية:
      Input: (T=50, P=20, C=3)
      Conv2D(32, 1×2, stride 1×2)  → (50, 10, 32) — دمج كل level بحجمه
      Conv2D(32, 4×1, padding same) → (50, 10, 32) — patterns زمنية
      Conv2D(32, 4×1, padding same) → (50, 10, 32) — patterns عميقة
      Inception Block ×2            → (50, 10, 64)
      GlobalAvgPool2D               → (64,)
      Dense(32, relu)               → (32,)
      Dense(8)                      → (8,) — Visual Embeddings
    """

    def __init__(self,
                 time_steps:   int = N_TIME_STEPS,
                 price_levels: int = N_PRICE_LEVELS,
                 channels:     int = N_CHANNELS,
                 emb_dim:      int = VISUAL_EMB_DIM,
                 brain_file:   str = 'outputs/deeplob_cnn.keras'):
        self.T  = time_steps
        self.P  = price_levels
        self.C  = channels
        self.emb_dim   = emb_dim
        self.brain_file = brain_file
        self.model      = None
        self._fitted    = False

        if not TF_AVAILABLE:
            return

        if os.path.exists(brain_file):
            try:
                self.model   = tf.keras.models.load_model(brain_file, compile=False)
                self._fitted = True
                print(f"[DeepLOB] 👁️ تحميل CNN: {brain_file}")
            except Exception as e:
                print(f"[DeepLOB] ⚠️ مكسور ({e}) — بنبني جديد")
                self.model = self._build()
        else:
            print("[DeepLOB] 👁️ بناء DeepLOB CNN...")
            self.model = self._build()

    # ── Build ────────────────────────────────────────────────────
    def _build(self) -> 'tf.keras.Model':
        inp = layers.Input(shape=(self.T, self.P, self.C), name='lob_tensor')

        # Stage 1: Level-wise fusion (1×2, stride 1×2)
        # يدمج كل سعر (bid/ask) مع حجمه دون التداخل مع المستويات الأخرى
        x = layers.Conv2D(32, (1, 2), strides=(1, 2),
                          padding='valid', activation='relu',
                          name='conv_level_fusion')(inp)
        x = layers.BatchNormalization()(x)
        # shape: (T, P//2, 32) = (50, 10, 32)

        # Stage 2: Temporal patterns (4×1)
        x = layers.Conv2D(32, (4, 1), padding='same',
                          activation='relu', name='conv_temporal_1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (4, 1), padding='same',
                          activation='relu', name='conv_temporal_2')(x)
        x = layers.BatchNormalization()(x)
        # shape: (50, 10, 32)

        # Stage 3: Inception Module ×2
        x = self._inception_block(x, filters=32, name_prefix='inc1')
        x = self._inception_block(x, filters=32, name_prefix='inc2')
        # shape: (50, 10, 64)

        # Stage 4: Compress
        x = layers.GlobalAveragePooling2D(name='gap')(x)
        # shape: (64,)

        x = layers.Dense(32, activation='relu', name='dense_compress')(x)
        x = layers.Dropout(0.2)(x)

        # Output: 8 Visual Embeddings
        emb = layers.Dense(self.emb_dim, name='visual_embeddings')(x)

        model = Model(inp, emb, name='DeepLOB_CNN')

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='mse'  # تُدرَّب كـ auxiliary task مع الـ LSTM
        )
        model.summary(line_length=80)
        return model

    def _inception_block(self, x, filters=32, name_prefix='inc'):
        """
        Inception Module: 3 مسارات زمنية متوازية
          1×1 (point-wise)
          2×1 (short temporal)
          4×1 (medium temporal)
        يُدمج النتائج بـ Concatenate
        """
        f = filters // 2

        # 1×1
        x1 = layers.Conv2D(f, (1, 1), padding='same',
                           activation='relu', name=f'{name_prefix}_1x1')(x)
        # 2×1
        x2 = layers.Conv2D(f, (2, 1), padding='same',
                           activation='relu', name=f'{name_prefix}_2x1')(x)
        # 4×1
        x3 = layers.Conv2D(f, (4, 1), padding='same',
                           activation='relu', name=f'{name_prefix}_4x1')(x)

        out = layers.Concatenate(name=f'{name_prefix}_concat')([x1, x2, x3])
        out = layers.BatchNormalization(name=f'{name_prefix}_bn')(out)
        return out

    # ── Fit (Auxiliary Task) ─────────────────────────────────────
    def fit_auxiliary(self,
                      X_tensors: np.ndarray,
                      y_targets: np.ndarray,
                      epochs:    int = 30,
                      batch:     int = 256,
                      output_dir: str = 'outputs') -> None:
        """
        يُدرَّب الـ CNN كـ auxiliary task قبل دمجه مع الـ LSTM.
        y_targets: يمكن أن تكون OBI أو CVD أو bias labels (regression proxy).
        """
        if not TF_AVAILABLE or self.model is None:
            return

        os.makedirs(output_dir, exist_ok=True)

        # Train/Val split (80/20, time-ordered)
        n = len(X_tensors)
        split = int(n * 0.8)
        X_tr, X_val = X_tensors[:split], X_tensors[split:]
        y_tr, y_val = y_targets[:split],  y_targets[split:]

        cb = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5,
                restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                self.brain_file, save_best_only=True,
                monitor='val_loss', verbose=0),
        ]

        print(f"\n👁️ DeepLOB CNN Auxiliary Training: {n:,} samples...")
        self.model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch,
            callbacks=cb,
            verbose=1,
        )
        self._fitted = True
        print(f"  ✅ DeepLOB CNN محفوظ: {self.brain_file}")

    # ── Inference ────────────────────────────────────────────────
    def get_embeddings(self, tensor: np.ndarray) -> np.ndarray:
        """
        يُخرج الـ 8 Visual Embeddings من tensor واحد أو batch.
        tensor shape: (T, P, C) أو (N, T, P, C)
        """
        if not TF_AVAILABLE or self.model is None or not self._fitted:
            if tensor.ndim == 3:
                return np.zeros(self.emb_dim, dtype=np.float32)
            return np.zeros((tensor.shape[0], self.emb_dim), dtype=np.float32)

        single = (tensor.ndim == 3)
        if single:
            tensor = tensor[np.newaxis]  # (1, T, P, C)

        emb = self.model.predict(tensor, verbose=0)

        return emb[0] if single else emb

    def save(self, output_dir: str = 'outputs') -> None:
        if self.model is not None:
            path = os.path.join(output_dir, 'deeplob_cnn.keras')
            self.model.save(path)
            print(f"  ✅ DeepLOB CNN: {path}")

    def load(self, output_dir: str = 'outputs') -> bool:
        path = os.path.join(output_dir, 'deeplob_cnn.keras')
        if not os.path.exists(path):
            return False
        try:
            self.model   = tf.keras.models.load_model(path, compile=False)
            self._fitted = True
            return True
        except Exception as e:
            print(f"  ⚠️ DeepLOB load failed: {e}")
            return False


# ══════════════════════════════════════════════════════════════════
# 3. LOBSequenceDataset — يبني dataset كامل من DataFrame
# ══════════════════════════════════════════════════════════════════
def build_lob_tensor_dataset(
        df_mbo:    'pd.DataFrame',
        df_mbp:    'pd.DataFrame',
        time_steps: int = N_TIME_STEPS) -> 'np.ndarray':
    """
    يبني مصفوفة من الـ 3D Tensors لكامل الـ dataset.

    Returns:
        X_lob: (N, T, P, C) — N = عدد الـ snapshots الكاملة
        ts_idx: list of timestamps لكل tensor (للـ alignment مع الـ labels)
    """
    import pandas as pd

    builder = LOBTensorBuilder(time_steps=time_steps)
    tensors = []
    ts_list = []

    # دمج MBO trades مع MBP snapshots بترتيب زمني
    df_mbo = df_mbo.copy()
    df_mbp = df_mbp.copy()

    df_mbo['_src'] = 'mbo'
    df_mbp['_src'] = 'mbp'

    # خلط بترتيب زمني
    ts_col = 'ts_event'
    df_combined = pd.concat([df_mbo, df_mbp], ignore_index=True, sort=False)
    df_combined = df_combined.sort_values(ts_col).reset_index(drop=True)

    from modules.microstructure import TRADE_ACTIONS

    for row in df_combined.itertuples(index=False):
        src    = getattr(row, '_src', 'mbp')
        action = str(getattr(row, 'action', '')).upper()
        price  = float(getattr(row, 'price', 0) or 0)
        size   = int(getattr(row, 'size', 0) or 0)
        side   = str(getattr(row, 'side', '')).upper()
        ts     = getattr(row, ts_col, None)

        if src == 'mbp':
            row_d = {col: getattr(row, col, 0) for col in df_mbp.columns
                     if col not in ('_src',)}
            builder.update_mbp(row_d)

            tensor = builder.get_tensor()
            if tensor is not None:
                tensors.append(tensor)
                ts_list.append(ts)

        elif src == 'mbo' and action in TRADE_ACTIONS and price > 0:
            is_buy = side in ('B', 'BID')
            builder.update_trade(price, size, is_buy)

    if not tensors:
        return np.zeros((0, time_steps, N_PRICE_LEVELS, N_CHANNELS), dtype=np.float32), []

    return np.array(tensors, dtype=np.float32), ts_list
