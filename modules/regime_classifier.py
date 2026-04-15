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

def _build_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    price = pd.to_numeric(
        df.get('price', pd.Series(np.zeros(len(df)), index=df.index)),
        errors='coerce',
    ).ffill().bfill().fillna(0.0)

    if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
        features['volatility'] = (df['high'] - df['low']) / df['close'].clip(lower=1e-8)
    elif 'micro_atr' in df.columns:
        features['volatility'] = df['micro_atr'].abs()
    else:
        features['volatility'] = df.get('volume_burst', pd.Series(np.zeros(len(df))))

    vol = pd.to_numeric(
        df.get('size', df.get('volume', df.get('volume_burst', pd.Series(np.ones(len(df)), index=df.index)))),
        errors='coerce',
    ).fillna(0.0).clip(lower=0.0)
    # Past-only normalization baseline; avoid backfilling from future rows.
    roll_mean = vol.shift(1).rolling(20, min_periods=1).mean().fillna(1.0)
    features['volume_ratio'] = (vol / roll_mean.clip(lower=1e-8)).clip(0, 5)

    cvd = df.get('cvd_delta', df.get('cvd', pd.Series(np.zeros(len(df)))))
    cvd_delta = pd.to_numeric(cvd, errors='coerce').diff().abs().fillna(0.0)
    features['cvd_strength'] = cvd_delta.shift(1).rolling(10, min_periods=1).mean().fillna(0.0)

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
        iet = pd.to_numeric(df.get('inter_event_time', pd.Series(np.nan, index=df.index)), errors='coerce')
        if iet.notna().any():
            baseline = iet.shift(1).rolling(20, min_periods=1).median().fillna(iet.median())
            features['activity'] = (baseline / iet.clip(lower=1e-6)).clip(0, 5).fillna(features['volume_ratio'])
        else:
            features['activity'] = features['volume_ratio']

    gross_move = price.diff().abs().rolling(12, min_periods=2).sum()
    net_move = price.diff(12).abs()
    features['trend_efficiency'] = (net_move / gross_move.clip(lower=1e-8)).fillna(0.0).clip(0.0, 1.0)

    return features.fillna(0.0).astype(np.float64)


def _pct_rank(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors='coerce').fillna(0.0)
    if len(series) == 0:
        return pd.Series(dtype=np.float64)
    return series.rank(method='average', pct=True).astype(np.float64)


def _build_rule_scores(X: pd.DataFrame) -> pd.DataFrame:
    vol_rank = _pct_rank(X['volatility'])
    activity_rank = _pct_rank(X['activity'])
    volume_rank = _pct_rank(X['volume_ratio'])
    cvd_rank = _pct_rank(X['cvd_strength'])
    trend_rank = _pct_rank(X['trend_efficiency'])
    imbalance_rank = _pct_rank(X['imbalance'])

    scores = pd.DataFrame(index=X.index)
    scores['volatile_score'] = (
        0.55 * vol_rank
        + 0.20 * activity_rank
        + 0.15 * volume_rank
        + 0.10 * imbalance_rank
    )
    scores['trend_score'] = (
        0.55 * trend_rank
        + 0.25 * cvd_rank
        + 0.10 * activity_rank
        + 0.10 * imbalance_rank
    )
    scores['low_liq_score'] = (
        0.60 * (1.0 - activity_rank)
        + 0.30 * (1.0 - volume_rank)
        + 0.10 * (1.0 - cvd_rank)
    )
    return scores.fillna(0.0).astype(np.float64)


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
        self.rule_stats = {}

    def fit(self, df: pd.DataFrame, output_dir: str = 'outputs') -> 'RegimeClassifier':
        X = _build_regime_features(df)
        requested_model = (self.model_type or 'auto').strip().lower()
        use_rules = requested_model in ('auto', 'rules')

        if use_rules:
            self.model = None
            self.scaler = None
            self._fit_rules(X)
            labels = self._predict_rules(X)
            self._labels = labels
            self._cluster_names = [REGIME_NAMES[i] for i in range(self.n_regimes)]
            self._regime_map = {i: i for i in range(self.n_regimes)}
            self._fitted = True
            print("  ✅ Regime rules fitted (semantic regime labels)")
            self._print_distribution(labels)
            self._save(output_dir)
            return self

        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        if requested_model == 'hmm':
            if not HMM_AVAILABLE:
                print("  ⚠️ HMM غير متاح — fallback إلى rules")
                self.model_type = 'rules'
                return self.fit(df, output_dir=output_dir)
            self._fit_hmm(X_scaled)
            print(f"  ✅ Regime HMM fitted ({self.n_regimes} regimes)")
        elif requested_model == 'gmm':
            if not GMM_AVAILABLE:
                print("  ⚠️ GMM غير متاح — fallback إلى rules")
                self.model_type = 'rules'
                return self.fit(df, output_dir=output_dir)
            self._fit_gmm(X_scaled)
            print(f"  ✅ Regime GMM fitted ({self.n_regimes} regimes)")
        else:
            self.model_type = 'rules'
            return self.fit(df, output_dir=output_dir)

        labels = self.model.predict(X_scaled)
        self._labels = labels
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

    def _fit_rules(self, X: pd.DataFrame):
        scores = _build_rule_scores(X)
        self.rule_stats = {
            'low_activity_q': float(X['activity'].quantile(0.30)),
            'low_volume_q': float(X['volume_ratio'].quantile(0.30)),
            'high_vol_q': float(X['volatility'].quantile(0.75)),
            'extreme_vol_q': float(X['volatility'].quantile(0.90)),
            'trend_eff_q': float(X['trend_efficiency'].quantile(0.45)),
            'cvd_q': float(X['cvd_strength'].quantile(0.45)),
            'activity_med': float(X['activity'].median()),
            'volume_med': float(X['volume_ratio'].median()),
            'volatile_score_q': float(scores['volatile_score'].quantile(0.82)),
            'trend_score_q': float(scores['trend_score'].quantile(0.58)),
            'low_liq_score_q': float(scores['low_liq_score'].quantile(0.80)),
        }

    def _predict_rules(self, X: pd.DataFrame) -> np.ndarray:
        if X.empty:
            return np.zeros(0, dtype=np.int8)

        stats = self.rule_stats or {}
        scores = _build_rule_scores(X)

        low_activity_q = float(stats.get('low_activity_q', X['activity'].quantile(0.30)))
        low_volume_q = float(stats.get('low_volume_q', X['volume_ratio'].quantile(0.30)))
        high_vol_q = float(stats.get('high_vol_q', X['volatility'].quantile(0.75)))
        extreme_vol_q = float(stats.get('extreme_vol_q', X['volatility'].quantile(0.90)))
        trend_eff_q = float(stats.get('trend_eff_q', X['trend_efficiency'].quantile(0.45)))
        cvd_q = float(stats.get('cvd_q', X['cvd_strength'].quantile(0.45)))
        activity_med = float(stats.get('activity_med', X['activity'].median()))
        volume_med = float(stats.get('volume_med', X['volume_ratio'].median()))
        volatile_score_q = float(stats.get('volatile_score_q', scores['volatile_score'].quantile(0.82)))
        trend_score_q = float(stats.get('trend_score_q', scores['trend_score'].quantile(0.58)))
        low_liq_score_q = float(stats.get('low_liq_score_q', scores['low_liq_score'].quantile(0.80)))

        volatile = (
            (scores['volatile_score'] >= volatile_score_q) &
            (
                (X['activity'] >= max(activity_med, 1e-6)) |
                (X['volume_ratio'] >= max(volume_med, 1e-6))
            )
        ) | (X['volatility'] >= max(extreme_vol_q, 1e-6))
        trending = (
            (scores['trend_score'] >= trend_score_q) &
            (X['trend_efficiency'] >= max(trend_eff_q, 0.25)) &
            (X['cvd_strength'] >= max(cvd_q, 1e-6) * 0.95) &
            (X['activity'] > max(low_activity_q, 1e-6)) &
            (X['volume_ratio'] > max(low_volume_q, 1e-6))
        )
        low_liq = (
            (scores['low_liq_score'] >= low_liq_score_q) &
            (X['activity'] <= max(activity_med, 1e-6)) &
            (X['volume_ratio'] <= max(volume_med, 1e-6)) &
            (X['volatility'] <= max(high_vol_q, 1e-6))
        )

        labels = np.full(len(X), 1, dtype=np.int8)
        labels[volatile.values] = 2
        labels[(low_liq & ~volatile).values] = 3
        labels[(trending & ~volatile & ~low_liq).values] = 0
        return labels

    def _print_distribution(self, labels: np.ndarray):
        for regime_id in range(self.n_regimes):
            count = int(np.sum(labels == regime_id))
            pct = count / max(len(labels), 1)
            print(f"     {REGIME_NAMES.get(regime_id, regime_id)}: {count} ({pct:.0%})")

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
            trend_eff = abs(m[5]) if len(m) > 5 else 0.0
            
            # حساب "Score" لكل حالة عشان التعيين يكون دقيق ومفيش حالة تاخد مكان التانية
            scores = {
                2: vol_score + vol_ratio + 0.25 * activity,
                0: trend_eff + cvd_str + 0.20 * activity,
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
        if not self._fitted:
            return np.zeros(len(df), dtype=np.int8)

        X = _build_regime_features(df)
        if self.model is None or (self.model_type or '').strip().lower() in ('auto', 'rules'):
            return self._predict_rules(X)
        X_scaled = self.scaler.transform(X)
        clusters = self.model.predict(X_scaled)
        
        return np.array([self._regime_map.get(int(c), 1) for c in clusters], dtype=np.int8)

    def predict_current(self, recent_bars: pd.DataFrame) -> dict:
        if not self._fitted or len(recent_bars) == 0:
            return {'regime_id': 0, 'regime_name': 'Trending', 'tradeable': True, 'confidence': 0.5}

        if self.model is None or (self.model_type or '').strip().lower() in ('auto', 'rules'):
            recent_regimes = self.predict(recent_bars)
            regime_id = Counter(recent_regimes[-min(10, len(recent_regimes)):]).most_common(1)[0][0]
            conf = 0.7 if recent_regimes[-1] == regime_id else 0.55
        else:
            X = _build_regime_features(recent_bars)
            X_scaled = self.scaler.transform(X)
            clusters = self.model.predict(X_scaled)

            if isinstance(self.model, GaussianMixture):
                probs = self.model.predict_proba(X_scaled)
                conf = float(probs[-1].max())
            else:
                try:
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
                         'regime_map': self._regime_map, 'fitted': self._fitted,
                         'model_type': self.model_type, 'rule_stats': self.rule_stats}, f)

    def load(self, output_dir: str) -> bool:
        path = os.path.join(output_dir, 'regime_classifier.pkl')
        if not os.path.exists(path): return False
        with open(path, 'rb') as f: d = pickle.load(f)
        self.model, self.scaler, self.n_regimes = d['model'], d['scaler'], d['n_regimes']
        self._regime_map, self._fitted, self.model_type = d['regime_map'], d['fitted'], d.get('model_type', 'auto')
        self.rule_stats = d.get('rule_stats', {})
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
