"""
autoencoder_extractor.py — مستخرج الميزات بالـ Autoencoder
"""
import numpy as np
import os
import pickle

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class AutoencoderExtractor:
    def __init__(self, input_dim: int = 54, bottleneck: int = 8): # زيادة البوتل نيك لـ 8 لاستيعاب السياق
        self.input_dim   = input_dim
        self.bottleneck  = bottleneck
        self.model       = None
        self.encoder     = None # إضافة نموذج الـ Encoder لاستخراج البصمة
        self._fitted     = False
        self._mean       = None
        self._std        = None

    def _build(self) -> None:
        if not TF_AVAILABLE: return

        # بناء المعمارية
        inp = layers.Input(shape=(self.input_dim,), name='ae_input')
        
        # الضاغط (Encoder)
        x = layers.Dense(64, activation='relu')(inp)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        encoded = layers.Dense(self.bottleneck, activation='relu', name='bottleneck')(x)

        # المُفكك (Decoder)
        x = layers.Dense(32, activation='relu')(encoded)
        x = layers.Dense(64, activation='relu')(x)
        out = layers.Dense(self.input_dim, activation='linear', name='ae_output')(x)

        # الموديل الكامل (للتدريب)
        self.model = Model(inp, out, name='AutoencoderFull')
        self.model.compile(optimizer='adam', loss='mse')

        # موديل الضاغط فقط (لاستخراج الفيتشرز العميقة)
        self.encoder = Model(inp, encoded, name='AutoencoderEncoder')

    def fit(self, X_roll: np.ndarray, epochs: int = 30, batch_size: int = 512, output_dir: str = 'outputs') -> 'AutoencoderExtractor':
        if not TF_AVAILABLE: 
            print("⚠️ TensorFlow غير متاح، لن يتم تدريب Autoencoder.")
            return self

        print(f"\n🤖 Autoencoder Training (input={X_roll.shape[1]}, bottleneck={self.bottleneck})...")

        # تحويل لـ Numpy Array لتجنب مشاكل Pandas Broadcast
        X_arr = np.asarray(X_roll, dtype=np.float32)

        self._mean = np.mean(X_arr, axis=0)
        self._std  = np.std(X_arr, axis=0) + 1e-8
        X_norm     = (X_arr - self._mean) / self._std

        if self.model is None:
            self._build()

        self.model.fit(
            X_norm, X_norm, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1, # Verbose 1 لرؤية التقدم
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
        )

        self._fitted = True
        self._save(output_dir)
        print("✅ تم تدريب الـ Autoencoder بنجاح.")
        return self

    def score(self, X_roll: np.ndarray) -> np.ndarray:
        """يعيد نسبة الخطأ في إعادة البناء (Reconstruction Error) كـ Anomaly Score"""
        if not self._fitted or self.model is None:
            return np.zeros(len(X_roll))

        X_arr = np.asarray(X_roll, dtype=np.float32)
        X_norm = (X_arr - self._mean) / self._std
        X_pred = self.model.predict(X_norm, verbose=0)
        errors = np.mean((X_norm - X_pred) ** 2, axis=1)

        e_min, e_max = errors.min(), errors.max()
        if e_max > e_min: return (errors - e_min) / (e_max - e_min)
        return errors

    def get_embeddings(self, X_roll: np.ndarray) -> np.ndarray:
        """يستخرج الأرقام السحرية (Latent Features) من طبقة البوتل نيك"""
        if not self._fitted or self.encoder is None:
            # لو الموديل مش جاهز، رجع أصفار عشان السيستم ما يقعش
            return np.zeros((len(X_roll), self.bottleneck), dtype=np.float32)
            
        X_arr = np.asarray(X_roll, dtype=np.float32)
        X_norm = (X_arr - self._mean) / self._std
        embeddings = self.encoder.predict(X_norm, verbose=0)
        return embeddings

    def _save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'autoencoder_extractor.pkl'), 'wb') as f:
            pickle.dump({'mean': self._mean, 'std': self._std, 'fitted': self._fitted, 'input_dim': self.input_dim, 'bottleneck': self.bottleneck}, f)
        
        if self.model: 
            self.model.save(os.path.join(output_dir, 'autoencoder_model.keras'))
        if self.encoder:
            self.encoder.save(os.path.join(output_dir, 'autoencoder_encoder.keras'))

    def load(self, output_dir: str) -> bool:
        path_pkl = os.path.join(output_dir, 'autoencoder_extractor.pkl')
        if not os.path.exists(path_pkl): return False
        
        with open(path_pkl, 'rb') as f:
            d = pickle.load(f)
            
        self._mean = d['mean']
        self._std = d['std']
        self._fitted = d['fitted']
        self.input_dim = d['input_dim']
        self.bottleneck = d.get('bottleneck', 4) # التوافق مع النسخ القديمة
        
        path_mdl = os.path.join(output_dir, 'autoencoder_model.keras')
        path_enc = os.path.join(output_dir, 'autoencoder_encoder.keras')
        
        if TF_AVAILABLE:
            if os.path.exists(path_mdl): 
                self.model = tf.keras.models.load_model(path_mdl)
            if os.path.exists(path_enc):
                self.encoder = tf.keras.models.load_model(path_enc)
            elif self.model:
                 # في حالة عدم وجود الانكودر محفوظ، نعيد بناءه من الموديل الكامل
                 inp = self.model.input
                 encoded = self.model.get_layer('bottleneck').output
                 self.encoder = Model(inp, encoded, name='AutoencoderEncoder')
                 
        return True