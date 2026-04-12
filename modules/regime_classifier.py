import numpy as np
import pandas as pd
import os
import pickle
from collections import Counter

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    GMM_AVAILABLE = True
except ImportError:
    GMM_AVAILABLE = False


REGIME_NAMES = {
    0: 'Trending',
    1: 'Ranging',
    2: 'Volatile',
    3: 'Low_Liquidity',
}

REGIME_TRADEABLE = {
    0: True,   
    1: False,  
    2: True,   
    3: False,  
}

def _build_regime_features(df: pd.DataFrame) -> np.ndarray:
    features = pd.DataFrame()

    if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
        features['volatility'] = (df['high'] - df['low']) / df['close'].clip(lower=1e-8)
    elif 'micro_atr' in df.columns:
        features['volatility'] = df['micro_atr'].abs()
    else:
        features['volatility'] = df.get('volume_burst', pd.Series(np.zeros(len(df))))

    vol = df.get('volume', df.get('volume_burst', pd.Series(np.ones(len(df)))))
    # Past-only normalization baseline; avoid backfilling from future rows.
    roll_mean = vol.shift(1).rolling(20, min_periods=1).mean().fillna(1.0)
    features['volume_ratio'] = (vol / roll_mean.clip(lower=1e-8)).clip(0, 5)

    cvd = df.get('cvd_delta', df.get('cvd', pd.Series(np.zeros(len(df)))))
    features['cvd_strength'] = cvd.shift(1).rolling(10, min_periods=1).mean().fillna(0.0)

    if 'imbalance' in df.columns:
        features['imbalance'] = df['imbalance'].abs()
    elif 'obi' in df.columns:
        features['imbalance'] = df['obi'].abs()
    else:
        features['imbalance'] = np.zeros(len(df))

    if 'bar_duration_s' in df.columns:
        dur = df['bar_duration_s']
        features['activity'] = 1.0 / dur.clip(lower=1).values
    else:
        features['activity'] = features['volume_ratio']

    return features.fillna(0).values.astype(np.float64)


class RegimeClassifier:
    def __init__(self, n_regimes: int = 4, model_type: str = 'auto'):
        self.n_regimes  = n_regimes
        self.model_type = model_type
        self.model      = None
        self.scaler     = None
        self._fitted        = False
        self._cluster_names = []   # ✅ FIX
        self._labels        = None
        self._regime_map = {}   

    def fit(self, df: pd.DataFrame, output_dir: str = 'outputs') -> 'RegimeClassifier':
        X = _build_regime_features(df)

        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled    = self.scaler.fit_transform(X)

        use_hmm = (HMM_AVAILABLE and self.model_type in ('hmm', 'auto') and len(X) >= self.n_regimes * 10)

        if use_hmm:
            self._fit_hmm(X_scaled) # تمرير الـ X_scaled فقط لتجنب خطأ الـ Signature
            print(f"  ✅ Regime HMM fitted ({self.n_regimes} regimes)")
        elif GMM_AVAILABLE:
            self._fit_gmm(X_scaled)
            print(f"  ✅ Regime GMM fitted ({self.n_regimes} regimes)")
        else:
            print("  ⚠️ مفيش HMM أو GMM — regime=Trending دايماً")
            self._fitted = True
            return self

        # تمرير الـ labels مباشرة لتجنب تكرار الـ Predict وتوحيد المنطق
        labels = self.model.predict(X_scaled)
        self._labels = labels  # ✅ FIX: حفظ labels للـ report
        self._cluster_names = [f'Cluster_{i}' for i in range(self.n_regimes)]
        self._map_clusters(labels, self.model.means_)

        self._fitted = True
        self._save(output_dir)
        return self

    def _fit_hmm(self, X_scaled: np.ndarray):
        model = hmm.GaussianHMM(
            n_components=self.n_regimes, covariance_type='diag',
            n_iter=200, tol=1e-4, random_state=42
        )
        model.fit(X_scaled)
        self.model = model

    def _fit_gmm(self, X_scaled: np.ndarray):
        model = GaussianMixture(
            n_components=self.n_regimes, covariance_type='diag',
            max_iter=300, n_init=5, random_state=42
        )
        model.fit(X_scaled)
        self.model = model

    def _map_clusters(self, labels: np.ndarray, means: np.ndarray):
        """
        يربط كل Cluster بحالة سوق (Regime) بشكل سليم ومنطقي
        """
        if self.model is None: return

        # إنشاء قائمة مؤقتة لترتيب الكلاسترات حسب خصائصها
        # features: [0:volatility, 1:volume_ratio, 2:cvd_strength, 3:imbalance, 4:activity]
        cluster_profiles = []
        for c in range(self.n_regimes):
            m = means[c]
            vol_score = m[0]
            vol_ratio = m[1]
            cvd_str   = abs(m[2])
            activity  = m[4]
            
            # حساب "Score" لكل حالة عشان التعيين يكون دقيق ومفيش حالة تاخد مكان التانية
            scores = {
                2: vol_score + vol_ratio,        # Volatile (high vol & volume)
                0: cvd_str + activity,           # Trending (high direction & speed)
                3: -activity - vol_ratio,        # Low Liquidity (low speed & low volume)
                1: 0                             # Ranging (Default/Fallback)
            }
            cluster_profiles.append({'cluster': c, 'scores': scores})

        self._regime_map = {}
        unassigned_regimes = {0, 1, 2, 3}
        
        # تعيين الحالات بدءاً من الأكثر تطرفاً (Volatile -> Low Liq -> Trending -> Ranging)
        for target_regime in [2, 3, 0, 1]:
            if not unassigned_regimes: break
            
            # العثور على أفضل Cluster للـ Regime ده
            best_cluster = None
            best_score = -float('inf')
            
            for profile in cluster_profiles:
                if profile['cluster'] not in self._regime_map:
                    if profile['scores'][target_regime] > best_score:
                        best_score = profile['scores'][target_regime]
                        best_cluster = profile['cluster']
            
            if best_cluster is not None:
                self._regime_map[best_cluster] = target_regime
                unassigned_regimes.remove(target_regime)

        # Print Distribution
        for c in range(self.n_regimes):
            regime_id = self._regime_map.get(c, 1)
            count     = int(np.sum(labels == c))
            pct       = count / max(len(labels), 1)
            print(f"     Cluster {c} → {REGIME_NAMES[regime_id]}: {count} ({pct:.0%})")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self.model is None:
            return np.zeros(len(df), dtype=np.int8)

        X = _build_regime_features(df)
        X_scaled = self.scaler.transform(X)
        clusters = self.model.predict(X_scaled)
        
        return np.array([self._regime_map.get(int(c), 1) for c in clusters], dtype=np.int8)

    def predict_current(self, recent_bars: pd.DataFrame) -> dict:
        if not self._fitted or self.model is None or len(recent_bars) == 0:
            return {'regime_id': 0, 'regime_name': 'Trending', 'tradeable': True, 'confidence': 0.5}

        X = _build_regime_features(recent_bars)
        X_scaled = self.scaler.transform(X)
        clusters = self.model.predict(X_scaled)

        if isinstance(self.model, GaussianMixture):
            probs = self.model.predict_proba(X_scaled)
            conf = float(probs[-1].max())
        else:
            # HMM Confidence Approximation
            try:
                # حساب الـ posterior probabilities للحالة الأخيرة فقط (بدل الـ Log Likelihood الكلي)
                probs = self.model.predict_proba(X_scaled)
                conf = float(probs[-1].max())
            except Exception:
                conf = 0.6

        recent_regimes = [self._regime_map.get(int(c), 1) for c in clusters[-min(10, len(clusters)):]]
        regime_id = Counter(recent_regimes).most_common(1)[0][0]

        return {
            'regime_id':   int(regime_id),
            'regime_name': REGIME_NAMES[regime_id],
            'tradeable':   REGIME_TRADEABLE[regime_id],
            'confidence':  round(conf, 4),
        }

    def _save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'regime_classifier.pkl'), 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler, 'n_regimes': self.n_regimes,
                         'regime_map': self._regime_map, 'fitted': self._fitted, 'model_type': self.model_type}, f)

    def load(self, output_dir: str) -> bool:
        path = os.path.join(output_dir, 'regime_classifier.pkl')
        if not os.path.exists(path): return False
        with open(path, 'rb') as f: d = pickle.load(f)
        self.model, self.scaler, self.n_regimes = d['model'], d['scaler'], d['n_regimes']
        self._regime_map, self._fitted, self.model_type = d['regime_map'], d['fitted'], d.get('model_type', 'auto')
        return True
    def predict_from_features(self, X: np.ndarray,
                               feature_names: list = None) -> np.ndarray:
        """
        V19: يُتنبَّأ بالـ Regime مباشرة من مصفوفة features.
        X: (N, n_features)
        """
        if not self._fitted or self.model is None:
            return np.zeros(len(X), dtype=np.int8)
        try:
            X_s = self.scaler.transform(X)
            raw = self.model.predict(X_s)
            labels = np.array([self._regime_map.get(int(c), 0) for c in raw], dtype=np.int8)
            return labels
        except Exception as e:
            return np.zeros(len(X), dtype=np.int8)


    def report(self) -> str:
        """تقرير مختصر عن حالة الـ Regime Classifier"""
        if not self._fitted:
            return "RegimeClassifier: غير مدرَّب"
        lines = ["\n🎯 Regime Classifier Report:"]
        regime_map = getattr(self, '_regime_map', {})
        labels = getattr(self, '_labels', None)
        for cluster_id, regime_name in regime_map.items():
            count = int((labels == cluster_id).sum()) if labels is not None else 0
            lines.append(f"   Cluster {cluster_id} → {regime_name}: {count} samples")
        return "\n".join(lines)
