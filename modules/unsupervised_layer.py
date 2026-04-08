"""
unsupervised_layer.py — طبقة التعلم غير الخاضع للإشراف (Deep Autoencoder Bridge)
"""
import numpy as np
import os
import pickle
# ③ FIX: import path مقاوم لكل بيئات التشغيل
try:
    from modules.autoencoder_extractor import AutoencoderExtractor
except ImportError:
    try:
        from autoencoder_extractor import AutoencoderExtractor
    except ImportError:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from autoencoder_extractor import AutoencoderExtractor

# احتفظنا بقاموس الميزات للتقارير لو احتجناه
SIGNATURE_FEATURES = {
    'cvd':               'ضغط الشراء/البيع',
    'obi':               'ميزان الـ book',
    'absorption_intensity': 'الامتصاص',
    'cancel_ratio':      'الجس',
    'spoofing_ratio':    'الـ Spoofing',
    'volume_burst':      'انفجار الحجم',
    'cvd_momentum':      'زخم CVD',
    'liquidity_sweep':   'سحب السيولة',
    'kyle_lambda':       'سيولة هشة',
    'hawkes_intensity':  'تجمع أوامر',
    'vnet':              'Smart Money',
}

def _aggregate_sequence(sequence: np.ndarray, window: int = 20) -> np.ndarray:
    """يجمع المتتالية الزمنية لتكون صالحة للإدخال في الـ Autoencoder"""
    if sequence.ndim == 3:
        seq = sequence[0]   
    else:
        seq = sequence      

    window = min(window, len(seq))
    if window == 0:
        return np.zeros(seq.shape[-1] * 4, dtype=np.float32)
        
    recent = seq[-window:]  

    mean_v = recent.mean(axis=0)
    std_v  = recent.std(axis=0) + 1e-8
    min_v  = recent.min(axis=0)
    max_v  = recent.max(axis=0)

    return np.concatenate([mean_v, std_v, min_v, max_v])

class UnsupervisedLayer:
    """
    هذه الطبقة الآن تستخدم Deep Autoencoder لضغط الفيتشرز واستخراج البصمة العميقة (Embeddings)
    بدلاً من الـ K-Means الكلاسيكي.
    """
    def __init__(self, feature_cols: list = None, window_size: int = 20, bottleneck_size: int = 8):
        self.feature_cols = feature_cols or []
        self.window_size  = window_size
        self.bottleneck_size = bottleneck_size
        self.autoencoder = None
        self._fitted      = False

    def _prepare_input(self, X: np.ndarray) -> np.ndarray:
        results = [_aggregate_sequence(x, window=self.window_size) for x in X]
        return np.array(results, dtype=np.float32)

    def fit(self, X: np.ndarray, y_bias: np.ndarray = None, output_dir: str = 'outputs') -> 'UnsupervisedLayer':
        """
        يقوم بتدريب الـ Autoencoder على البيانات الخام. 
        y_bias موجود للتوافق مع الـ API القديم ولكنه لا يستخدم في التعلم العميق غير الموجه.
        """
        print(f"\n🔍 Deep Unsupervised Layer (window={self.window_size} ticks)...")
        X_agg = self._prepare_input(X)
        
        input_dim = X_agg.shape[1]
        self.autoencoder = AutoencoderExtractor(input_dim=input_dim, bottleneck=self.bottleneck_size)
        
        # تدريب الـ Autoencoder
        self.autoencoder.fit(X_agg, epochs=30, batch_size=256, output_dir=output_dir)
        
        self._fitted = True
        self._save(output_dir)
        return self

    def predict(self, sequence: np.ndarray, bias_label: int = None) -> dict:
        """
        يستخرج الـ Embeddings للمتتالية الحالية.
        """
        if not self._fitted or self.autoencoder is None:
            # إعادة أصفار إذا لم يكن الموديل جاهزاً
            return {
                'anomaly_score': 0.0,
                'embeddings': np.zeros(self.bottleneck_size, dtype=np.float32)
            }

        agg = _aggregate_sequence(sequence, window=self.window_size)
        x_input = agg.reshape(1, -1)
        
        # استخراج درجة الشذوذ (Anomaly Score)
        score = float(self.autoencoder.score(x_input)[0])
        
        # استخراج الأرقام السحرية (Embeddings)
        embeddings = self.autoencoder.get_embeddings(x_input)[0]

        return {
            'anomaly_score': round(score, 4),
            'embeddings': embeddings
        }

    def fit_from_rolling(self, X_roll: np.ndarray, y_bias: np.ndarray = None, output_dir: str = 'outputs') -> 'UnsupervisedLayer':
        """تدريب الموديل لو الداتا مجهزة كـ Rolling Window (مسطحة)"""
        print(f"\n🔍 Deep Unsupervised Layer (from rolling data)...")
        input_dim = X_roll.shape[1]
        self.autoencoder = AutoencoderExtractor(input_dim=input_dim, bottleneck=self.bottleneck_size)
        
        self.autoencoder.fit(X_roll, epochs=30, batch_size=256, output_dir=output_dir)
        
        self._fitted = True
        self._save(output_dir)
        return self

    def predict_from_rolling(self, x_roll: np.ndarray, bias_label: int = None) -> dict:
        """استخراج البصمة من صف واحد مجهز"""
        if not self._fitted or self.autoencoder is None:
            return {
                'anomaly_score': 0.0,
                'embeddings': np.zeros(self.bottleneck_size, dtype=np.float32)
            }

        x_input = x_roll.reshape(1, -1)
        score = float(self.autoencoder.score(x_input)[0])
        embeddings = self.autoencoder.get_embeddings(x_input)[0]

        return {
            'anomaly_score': round(score, 4),
            'embeddings': embeddings
        }

    def _save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'unsupervised_layer.pkl')
        with open(path, 'wb') as f:
            pickle.dump({
                'fitted':       self._fitted,
                'window_size':  self.window_size,
                'feature_cols': self.feature_cols,
                'bottleneck_size': self.bottleneck_size
            }, f)

    def load(self, output_dir: str) -> bool:
        path = os.path.join(output_dir, 'unsupervised_layer.pkl')
        if not os.path.exists(path): return False
        
        with open(path, 'rb') as f: 
            d = pickle.load(f)
            
        self._fitted      = d.get('fitted', False)
        self.window_size  = d.get('window_size', 20)
        self.feature_cols = d.get('feature_cols', [])
        self.bottleneck_size = d.get('bottleneck_size', 8)
        
        # تحميل موديل الـ Autoencoder
        self.autoencoder = AutoencoderExtractor()
        ae_loaded = self.autoencoder.load(output_dir)
        
        if not ae_loaded:
            self._fitted = False
            return False
            
        return True

    def report(self) -> str:
        if not self._fitted: return "Deep Unsupervised Layer: غير مدرّب"
        return f"\n🔍 Deep Unsupervised Layer Report:\n  ✅ Autoencoder is Active.\n  ✅ Bottleneck Size (Embeddings): {self.bottleneck_size}"