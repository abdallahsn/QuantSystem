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
import math
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
        bid_sizes = [float(row.get(f'bid_sz_{i:02d}', 0) or 0) for i in range(self.L)]
        ask_sizes = [float(row.get(f'ask_sz_{i:02d}', 0) or 0) for i in range(self.L)]
        bid0 = float(row.get('bid_px_00', 0) or 0)
        ask0 = float(row.get('ask_px_00', 0) or 0)
        self.update_mbp_levels(bid0, ask0, bid_sizes, ask_sizes)

    def update_mbp_levels(self,
                          bid0: float,
                          ask0: float,
                          bid_sizes,
                          ask_sizes) -> None:
        """
        نسخة خفيفة من update_mbp تستقبل arrays/scalars مباشرة
        لتجنب بناء dict لكل snapshot في المسارات الكبيرة.
        """
        self._last_mbp = {'bid_px_00': bid0, 'ask_px_00': ask0}
        L = self.L

        # استخراج mid price للـ price grid
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
            depth[L - 1 - i] = float(bid_sizes[i] or 0)
            # ask levels: index 10..19 (i0 في المنتصف)
            depth[L + i]     = float(ask_sizes[i] or 0)

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


def estimate_lob_tensor_bytes(n_tensors: int,
                              time_steps: int = N_TIME_STEPS,
                              price_levels: int = N_PRICE_LEVELS,
                              channels: int = N_CHANNELS,
                              dtype=np.float32) -> int:
    return int(n_tensors) * int(time_steps) * int(price_levels) * int(channels) * np.dtype(dtype).itemsize


def _prepare_sorted_frame(df, ts_col: str = 'ts_event'):
    import pandas as pd

    if df is None or len(df) == 0:
        return pd.DataFrame()

    out = df.copy(deep=False)
    out[ts_col] = pd.to_datetime(out.get(ts_col), utc=True, errors='coerce').dt.tz_localize(None)
    out = out[out[ts_col].notna()]
    if len(out) == 0:
        return out.reset_index(drop=True)

    if not out[ts_col].is_monotonic_increasing:
        out = out.sort_values(ts_col)
    return out.reset_index(drop=True)


def _build_snapshot_sampling_plan(n_snapshots: int,
                                  time_steps: int,
                                  snapshot_stride: int | None = None,
                                  max_tensors: int | None = None) -> dict:
    if n_snapshots <= 0:
        return {'stride': 1, 'first_emit_pos': 0, 'planned_tensors': 0}

    stride = max(1, int(snapshot_stride or 1))
    if max_tensors and max_tensors > 0:
        stride = max(stride, int(math.ceil(n_snapshots / max_tensors)))

    min_ready_pos = max(time_steps - 1, 0)
    first_emit_pos = int(math.ceil(min_ready_pos / stride) * stride)
    if first_emit_pos >= n_snapshots:
        planned = 0
    else:
        planned = int(((n_snapshots - 1 - first_emit_pos) // stride) + 1)

    return {
        'stride': stride,
        'first_emit_pos': first_emit_pos,
        'planned_tensors': planned,
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
        time_steps: int = N_TIME_STEPS,
        output_path: str | None = None,
        timestamps_path: str | None = None,
        snapshot_stride: int | None = None,
        max_tensors: int | None = None,
        emit_positions=None) -> dict:
    """
    يبني مصفوفة من الـ 3D Tensors؛ يحمّل أعمدة MBP بالكامل في الذاكرة (مناسب لسيرفر RAM كبيرة).

    Returns:
        dict يحتوي metadata عن الملفات الناتجة والتخطيط المستخدم.
    """
    import pandas as pd
    from numpy.lib.format import open_memmap
    from modules.microstructure import TRADE_ACTIONS

    ts_col = 'ts_event'
    df_mbo = _prepare_sorted_frame(df_mbo, ts_col=ts_col)
    df_mbp = _prepare_sorted_frame(df_mbp, ts_col=ts_col)

    emit_positions_arr = None
    if emit_positions is not None:
        emit_positions_arr = np.asarray(emit_positions, dtype=np.int64).reshape(-1)
        if emit_positions_arr.size:
            min_ready_pos = max(int(time_steps) - 1, 0)
            emit_positions_arr = emit_positions_arr[
                (emit_positions_arr >= min_ready_pos) & (emit_positions_arr < len(df_mbp))
            ]
            emit_positions_arr = np.unique(emit_positions_arr)
        planned_tensors = int(len(emit_positions_arr))
        plan = {
            'stride': 1,
            'first_emit_pos': int(emit_positions_arr[0]) if planned_tensors else 0,
            'planned_tensors': planned_tensors,
        }
    else:
        plan = _build_snapshot_sampling_plan(
            len(df_mbp),
            time_steps=time_steps,
            snapshot_stride=snapshot_stride,
            max_tensors=max_tensors,
        )
        planned_tensors = int(plan['planned_tensors'])

    metadata = {
        'n_mbo_rows': int(len(df_mbo)),
        'n_mbp_rows': int(len(df_mbp)),
        'time_steps': int(time_steps),
        'snapshot_stride': int(plan['stride']),
        'planned_tensors': planned_tensors,
        'estimated_tensor_bytes': estimate_lob_tensor_bytes(planned_tensors, time_steps=time_steps),
        'output_path': output_path,
        'timestamps_path': timestamps_path,
        'emit_positions_requested': 0 if emit_positions_arr is None else int(len(emit_positions_arr)),
    }

    empty_tensors = np.zeros((0, time_steps, N_PRICE_LEVELS, N_CHANNELS), dtype=np.float32)
    empty_ts = np.array([], dtype=np.int64)
    if len(df_mbp) == 0 or planned_tensors <= 0:
        if output_path:
            np.save(output_path, empty_tensors)
        if timestamps_path:
            np.save(timestamps_path, empty_ts)
        metadata['built_tensors'] = 0
        return metadata

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tensor_store = open_memmap(
            output_path,
            mode='w+',
            dtype=np.float32,
            shape=(planned_tensors, time_steps, N_PRICE_LEVELS, N_CHANNELS),
        )
    else:
        tensor_store = np.empty((planned_tensors, time_steps, N_PRICE_LEVELS, N_CHANNELS), dtype=np.float32)
    ts_store = np.empty(planned_tensors, dtype=np.int64)

    builder = LOBTensorBuilder(time_steps=time_steps)

    mbo_ts = df_mbo[ts_col].to_numpy(dtype='datetime64[ns]', copy=False) if len(df_mbo) else np.empty(0, dtype='datetime64[ns]')
    mbo_action = df_mbo.get('action', pd.Series('', index=df_mbo.index)).astype(str).str.upper().to_numpy(copy=False) if len(df_mbo) else np.array([], dtype=object)
    mbo_price = pd.to_numeric(df_mbo.get('price', pd.Series(0, index=df_mbo.index)), errors='coerce').fillna(0.0).to_numpy(dtype=np.float32, copy=False) if len(df_mbo) else np.empty(0, dtype=np.float32)
    mbo_size = pd.to_numeric(df_mbo.get('size', pd.Series(0, index=df_mbo.index)), errors='coerce').fillna(0).to_numpy(dtype=np.int32, copy=False) if len(df_mbo) else np.empty(0, dtype=np.int32)
    mbo_side = df_mbo.get('side', pd.Series('', index=df_mbo.index)).astype(str).str.upper().to_numpy(copy=False) if len(df_mbo) else np.array([], dtype=object)

    mbp_ts = df_mbp[ts_col].to_numpy(dtype='datetime64[ns]', copy=False)
    bid_px = [pd.to_numeric(df_mbp.get(f'bid_px_{i:02d}', pd.Series(0, index=df_mbp.index)), errors='coerce').fillna(0.0).to_numpy(dtype=np.float32, copy=False) for i in range(N_PRICE_LEVELS // 2)]
    ask_px = [pd.to_numeric(df_mbp.get(f'ask_px_{i:02d}', pd.Series(0, index=df_mbp.index)), errors='coerce').fillna(0.0).to_numpy(dtype=np.float32, copy=False) for i in range(N_PRICE_LEVELS // 2)]
    bid_sz = [pd.to_numeric(df_mbp.get(f'bid_sz_{i:02d}', pd.Series(0, index=df_mbp.index)), errors='coerce').fillna(0.0).to_numpy(dtype=np.float32, copy=False) for i in range(N_PRICE_LEVELS // 2)]
    ask_sz = [pd.to_numeric(df_mbp.get(f'ask_sz_{i:02d}', pd.Series(0, index=df_mbp.index)), errors='coerce').fillna(0.0).to_numpy(dtype=np.float32, copy=False) for i in range(N_PRICE_LEVELS // 2)]

    i_mbo = 0
    i_mbp = 0
    mbp_pos = -1
    next_emit_pos = int(plan['first_emit_pos'])
    write_pos = 0
    emit_lookup = set(int(pos) for pos in emit_positions_arr.tolist()) if emit_positions_arr is not None else None

    while i_mbo < len(df_mbo) or i_mbp < len(df_mbp):
        use_mbo = (
            i_mbo < len(df_mbo) and (
                i_mbp >= len(df_mbp) or mbo_ts[i_mbo] <= mbp_ts[i_mbp]
            )
        )

        if use_mbo:
            action = mbo_action[i_mbo]
            price = float(mbo_price[i_mbo])
            if action in TRADE_ACTIONS and price > 0:
                builder.update_trade(
                    price,
                    int(mbo_size[i_mbo]),
                    mbo_side[i_mbo] in ('B', 'BID'),
                )
            i_mbo += 1
            continue

        bid0 = float(bid_px[0][i_mbp]) if bid_px else 0.0
        ask0 = float(ask_px[0][i_mbp]) if ask_px else 0.0
        bid_sizes = [arr[i_mbp] for arr in bid_sz]
        ask_sizes = [arr[i_mbp] for arr in ask_sz]
        builder.update_mbp_levels(bid0, ask0, bid_sizes, ask_sizes)

        mbp_pos += 1
        should_emit = False
        if emit_lookup is not None:
            should_emit = mbp_pos in emit_lookup
        elif mbp_pos == next_emit_pos:
            should_emit = True

        if should_emit:
            tensor = builder.get_tensor()
            if tensor is not None and write_pos < planned_tensors:
                tensor_store[write_pos] = tensor
                ts_store[write_pos] = mbp_ts[i_mbp].astype('datetime64[ns]').astype(np.int64)
                write_pos += 1
            if emit_lookup is None:
                next_emit_pos += int(plan['stride'])

        i_mbp += 1

    if output_path and hasattr(tensor_store, 'flush'):
        tensor_store.flush()
    if timestamps_path:
        np.save(timestamps_path, ts_store[:write_pos])

    metadata['built_tensors'] = int(write_pos)
    metadata['tensor_timestamps_saved'] = int(write_pos)
    metadata['estimated_tensor_bytes'] = estimate_lob_tensor_bytes(write_pos, time_steps=time_steps)
    return metadata
