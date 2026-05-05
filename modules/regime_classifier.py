import numpy as np
import pandas as pd
import os
import pickle
from collections import Counter

try:
    from scipy.stats import wasserstein_distance as SCIPY_WASSERSTEIN_DISTANCE
except ImportError:
    SCIPY_WASSERSTEIN_DISTANCE = None

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

REGIME_ONE_HOT_COLS = tuple(f'cluster_{i}' for i in range(4))
REGIME_META_SCORE_COLS = (
    'regime_trend_score',
    'regime_volatile_score',
    'regime_low_liq_score',
)
N_REGIME_SCORE_COLS = len(REGIME_META_SCORE_COLS)
TREND_EFFICIENCY_WINDOW = 50
CVD_PERSISTENCE_WINDOW = 50

REGIME_TRADEABLE = {
    0: True,   
    1: False,  
    2: False,  
    3: False,  
}


def _past_only_numeric_series(values, *, index, context: str, fill_value: float = 0.0) -> pd.Series:
    series = pd.to_numeric(values, errors='coerce')
    if not isinstance(series, pd.Series):
        series = pd.Series(series, index=index)
    if len(series) == 0:
        return pd.Series(dtype=np.float64, index=index)
    if series.notna().sum() == 0:
        raise ValueError(f'❌ {context} has no valid numeric observations')
    return series.ffill().fillna(float(fill_value)).astype(np.float64)


def _resolve_cvd_level(df: pd.DataFrame) -> pd.Series:
    if 'cvd' in df.columns:
        base = pd.to_numeric(df.get('cvd'), errors='coerce')
    elif 'cvd_delta' in df.columns:
        base = pd.to_numeric(df.get('cvd_delta'), errors='coerce').fillna(0.0).cumsum()
    else:
        base = pd.Series(np.zeros(len(df)), index=df.index, dtype=np.float64)
    return _past_only_numeric_series(
        base,
        index=df.index,
        context='regime_classifier.cvd',
        fill_value=0.0,
    )

def _build_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    price = _past_only_numeric_series(
        df.get('price', pd.Series(np.zeros(len(df)), index=df.index)),
        index=df.index,
        context='regime_classifier.price',
        fill_value=0.0,
    )

    if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
        features['volatility'] = (df['high'] - df['low']) / df['close'].clip(lower=1e-8)
    elif 'raw__micro_atr' in df.columns:
        features['volatility'] = pd.to_numeric(df['raw__micro_atr'], errors='coerce').abs()
    elif 'micro_atr' in df.columns:
        features['volatility'] = pd.to_numeric(df['micro_atr'], errors='coerce').abs()
    else:
        features['volatility'] = df.get('volume_burst', pd.Series(np.zeros(len(df))))

    vol = pd.to_numeric(
        df.get('size', df.get('volume', df.get('volume_burst', pd.Series(np.ones(len(df)), index=df.index)))),
        errors='coerce',
    ).fillna(0.0).clip(lower=0.0)
    # Past-only normalization baseline; avoid backfilling from future rows.
    roll_mean = vol.shift(1).rolling(20, min_periods=1).mean().fillna(1.0)
    features['volume_ratio'] = (vol / roll_mean.clip(lower=1e-8)).clip(0, 5)

    cvd_level = _resolve_cvd_level(df)
    cvd_delta = cvd_level.diff().abs().fillna(0.0)
    features['cvd_impulse'] = cvd_delta.shift(1).rolling(10, min_periods=1).mean().fillna(0.0)
    features['cvd_persistence'] = (
        cvd_level.shift(1).diff(CVD_PERSISTENCE_WINDOW).abs().rolling(5, min_periods=1).mean().fillna(0.0)
    )
    features['cvd_strength'] = features['cvd_impulse']

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

    gross_move = price.diff().abs().rolling(TREND_EFFICIENCY_WINDOW, min_periods=2).sum()
    net_move = price.diff(TREND_EFFICIENCY_WINDOW).abs()
    features['trend_efficiency'] = (net_move / gross_move.clip(lower=1e-8)).fillna(0.0).clip(0.0, 1.0)

    return features.fillna(0.0).astype(np.float64)


def _pct_rank(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors='coerce').fillna(0.0)
    if len(series) == 0:
        return pd.Series(dtype=np.float64)
    return series.rank(method='average', pct=True).astype(np.float64)


def _clip01(values) -> pd.Series:
    return pd.Series(values).fillna(0.0).clip(0.0, 1.0).astype(np.float64)


def _bounded_scale(series: pd.Series, low: float, high: float) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce').fillna(0.0).astype(np.float64)
    lo = float(low)
    hi = float(high)
    if not np.isfinite(lo):
        lo = float(s.min()) if len(s) else 0.0
    if not np.isfinite(hi):
        hi = float(s.max()) if len(s) else lo + 1.0
    if hi <= lo:
        hi = lo + 1e-6
    return ((s - lo) / max(hi - lo, 1e-6)).clip(0.0, 1.0)


def _fit_rule_feature_stats(X: pd.DataFrame) -> dict:
    stats: dict[str, dict[str, float]] = {}
    for col in ('volatility', 'activity', 'volume_ratio', 'cvd_impulse', 'cvd_persistence', 'trend_efficiency', 'imbalance'):
        s = pd.to_numeric(X.get(col, pd.Series(dtype=np.float64)), errors='coerce').fillna(0.0).astype(np.float64)
        stats[col] = {
            'q10': float(s.quantile(0.10)) if len(s) else 0.0,
            'q25': float(s.quantile(0.25)) if len(s) else 0.0,
            'q50': float(s.quantile(0.50)) if len(s) else 0.0,
            'q75': float(s.quantile(0.75)) if len(s) else 0.0,
            'q90': float(s.quantile(0.90)) if len(s) else 0.0,
        }
    return stats


def _score_feature_from_stats(
    X: pd.DataFrame,
    feature_stats: dict,
    col: str,
    low_key: str = 'q25',
    high_key: str = 'q90',
) -> pd.Series:
    stats = feature_stats.get(col, {})
    return _bounded_scale(
        pd.to_numeric(X.get(col, pd.Series(np.zeros(len(X)), index=X.index)), errors='coerce').fillna(0.0),
        float(stats.get(low_key, 0.0)),
        float(stats.get(high_key, 1.0)),
    ).set_axis(X.index)


def _build_rule_scores(X: pd.DataFrame, feature_stats: dict | None = None) -> pd.DataFrame:
    if feature_stats:
        vol_rank = _score_feature_from_stats(X, feature_stats, 'volatility')
        activity_rank = _score_feature_from_stats(X, feature_stats, 'activity')
        volume_rank = _score_feature_from_stats(X, feature_stats, 'volume_ratio')
        cvd_impulse_rank = _score_feature_from_stats(X, feature_stats, 'cvd_impulse')
        cvd_persistence_rank = _score_feature_from_stats(X, feature_stats, 'cvd_persistence')
        trend_rank = _score_feature_from_stats(X, feature_stats, 'trend_efficiency', low_key='q10', high_key='q75')
        imbalance_rank = _score_feature_from_stats(X, feature_stats, 'imbalance')
    else:
        vol_rank = _pct_rank(X['volatility'])
        activity_rank = _pct_rank(X['activity'])
        volume_rank = _pct_rank(X['volume_ratio'])
        cvd_impulse_rank = _pct_rank(X['cvd_impulse'])
        cvd_persistence_rank = _pct_rank(X['cvd_persistence'])
        trend_rank = _pct_rank(X['trend_efficiency'])
        imbalance_rank = _pct_rank(X['imbalance'])

    scores = pd.DataFrame(index=X.index)
    scores[REGIME_META_SCORE_COLS[1]] = (
        0.50 * vol_rank
        + 0.20 * activity_rank
        + 0.15 * volume_rank
        + 0.10 * cvd_impulse_rank
        + 0.05 * imbalance_rank
    )
    scores[REGIME_META_SCORE_COLS[0]] = (
        0.55 * trend_rank
        + 0.25 * cvd_persistence_rank
        + 0.10 * activity_rank
        + 0.10 * imbalance_rank
    )
    scores[REGIME_META_SCORE_COLS[2]] = (
        0.55 * (1.0 - activity_rank)
        + 0.25 * (1.0 - volume_rank)
        + 0.20 * (1.0 - cvd_impulse_rank)
    )
    return scores.fillna(0.0).clip(0.0, 1.0).astype(np.float64)


def _normalize_prob_rows(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError(f"Expected 2D regime probabilities, got {out.shape}")
    out = np.clip(out, 0.0, None)
    row_sums = out.sum(axis=1, keepdims=True)
    invalid = row_sums[:, 0] <= 0
    if np.any(invalid):
        out[invalid] = 1.0
        row_sums = out.sum(axis=1, keepdims=True)
    return (out / np.clip(row_sums, 1e-12, None)).astype(np.float64, copy=False)


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
        self.last_rule_diagnostics = {}

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
            self._print_rule_diagnostics(len(labels))
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
        feature_stats = _fit_rule_feature_stats(X)
        scores = _build_rule_scores(X, feature_stats=feature_stats)
        self.rule_stats = {
            'feature_stats': feature_stats,
            'low_activity_q': float(X['activity'].quantile(0.30)),
            'low_volume_q': float(X['volume_ratio'].quantile(0.30)),
            'high_vol_q': float(X['volatility'].quantile(0.75)),
            'extreme_vol_q': float(X['volatility'].quantile(0.90)),
            'trend_eff_q': float(X['trend_efficiency'].quantile(0.45)),
            'cvd_impulse_q': float(X['cvd_impulse'].quantile(0.45)),
            'cvd_persistence_q': float(X['cvd_persistence'].quantile(0.45)),
            'activity_med': float(X['activity'].median()),
            'volume_med': float(X['volume_ratio'].median()),
            'volatile_score_q': float(scores[REGIME_META_SCORE_COLS[1]].quantile(0.82)),
            'trend_score_q': float(scores[REGIME_META_SCORE_COLS[0]].quantile(0.55)),
            'low_liq_score_q': float(scores[REGIME_META_SCORE_COLS[2]].quantile(0.80)),
        }

    def predict_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            return pd.DataFrame(columns=list(REGIME_META_SCORE_COLS))

        X = _build_regime_features(df)
        if self.model is None or (self.model_type or '').strip().lower() in ('auto', 'rules'):
            feature_stats = (self.rule_stats or {}).get('feature_stats', {})
            return _build_rule_scores(X, feature_stats=feature_stats).reindex(
                columns=list(REGIME_META_SCORE_COLS),
                fill_value=0.0,
            )

        X_scaled = self.scaler.transform(X)
        probs = None
        if hasattr(self.model, 'predict_proba'):
            try:
                probs = np.asarray(self.model.predict_proba(X_scaled), dtype=np.float64)
            except Exception:
                probs = None
        if probs is None:
            raw = np.asarray(self.model.predict(X_scaled), dtype=np.int32)
            probs = np.zeros((len(raw), self.n_regimes), dtype=np.float64)
            probs[np.arange(len(raw)), np.clip(raw, 0, self.n_regimes - 1)] = 1.0

        regime_probs = np.zeros((len(probs), self.n_regimes), dtype=np.float64)
        for cluster_idx in range(probs.shape[1]):
            regime_idx = int(self._regime_map.get(int(cluster_idx), 1))
            regime_probs[:, regime_idx] += probs[:, cluster_idx]

        return pd.DataFrame(
            {
                REGIME_META_SCORE_COLS[0]: _clip01(regime_probs[:, 0]).values,
                REGIME_META_SCORE_COLS[1]: _clip01(regime_probs[:, 2]).values,
                REGIME_META_SCORE_COLS[2]: _clip01(regime_probs[:, 3]).values,
            },
            index=X.index,
        )

    def _predict_rules(self, X: pd.DataFrame) -> np.ndarray:
        if X.empty:
            return np.zeros(0, dtype=np.int8)

        stats = self.rule_stats or {}
        feature_stats = stats.get('feature_stats', {})
        scores = _build_rule_scores(X, feature_stats=feature_stats)

        low_activity_q = float(stats.get('low_activity_q', X['activity'].quantile(0.30)))
        low_volume_q = float(stats.get('low_volume_q', X['volume_ratio'].quantile(0.30)))
        high_vol_q = float(stats.get('high_vol_q', X['volatility'].quantile(0.75)))
        extreme_vol_q = float(stats.get('extreme_vol_q', X['volatility'].quantile(0.90)))
        trend_eff_q = float(stats.get('trend_eff_q', X['trend_efficiency'].quantile(0.45)))
        cvd_impulse_q = float(stats.get('cvd_impulse_q', X['cvd_impulse'].quantile(0.45)))
        cvd_persistence_q = float(stats.get('cvd_persistence_q', X['cvd_persistence'].quantile(0.45)))
        activity_med = float(stats.get('activity_med', X['activity'].median()))
        volume_med = float(stats.get('volume_med', X['volume_ratio'].median()))
        volatile_score_q = float(stats.get('volatile_score_q', scores[REGIME_META_SCORE_COLS[1]].quantile(0.82)))
        trend_score_q = float(stats.get('trend_score_q', scores[REGIME_META_SCORE_COLS[0]].quantile(0.55)))
        low_liq_score_q = float(stats.get('low_liq_score_q', scores[REGIME_META_SCORE_COLS[2]].quantile(0.80)))

        volatile = (
            (scores[REGIME_META_SCORE_COLS[1]] >= volatile_score_q) &
            (
                (X['activity'] >= max(activity_med, 1e-6)) |
                (X['volume_ratio'] >= max(volume_med, 1e-6)) |
                (X['cvd_impulse'] >= max(cvd_impulse_q, 1e-6))
            )
        ) | (X['volatility'] >= max(extreme_vol_q, 1e-6))
        trend_guard_1 = X['trend_efficiency'] >= max(trend_eff_q, 0.25)
        trend_guard_2 = X['cvd_persistence'] >= max(cvd_persistence_q, 1e-6) * 0.95
        trend_guard_3 = X['activity'] > max(low_activity_q, 1e-6)
        trend_guard_4 = X['volume_ratio'] > max(low_volume_q, 1e-6)
        trend_guard_count = (
            trend_guard_1.astype(np.int8)
            + trend_guard_2.astype(np.int8)
            + trend_guard_3.astype(np.int8)
            + trend_guard_4.astype(np.int8)
        )
        trending = (
            (scores[REGIME_META_SCORE_COLS[0]] >= trend_score_q) &
            (trend_guard_count >= 2)
        )
        low_liq = (
            (scores[REGIME_META_SCORE_COLS[2]] >= low_liq_score_q) &
            (X['activity'] <= max(activity_med, 1e-6)) &
            (X['volume_ratio'] <= max(volume_med, 1e-6)) &
            (X['volatility'] <= max(high_vol_q, 1e-6)) &
            (X['cvd_impulse'] <= max(cvd_impulse_q, 1e-6))
        )

        labels = np.full(len(X), 1, dtype=np.int8)
        labels[low_liq.values] = 3
        labels[(trending & ~low_liq).values] = 0
        labels[(volatile & ~low_liq & ~trending).values] = 2
        self.last_rule_diagnostics = {
            'trend_volatile_overlap_count': int((trending & volatile).sum()),
            'trending_masked_by_priority_count': int((trending & low_liq).sum()),
            'trending_share': float(np.mean(labels == 0)) if len(labels) else 0.0,
            'volatile_share': float(np.mean(labels == 2)) if len(labels) else 0.0,
            'low_liq_share': float(np.mean(labels == 3)) if len(labels) else 0.0,
        }
        return labels

    def _print_distribution(self, labels: np.ndarray):
        for regime_id in range(self.n_regimes):
            count = int(np.sum(labels == regime_id))
            pct = count / max(len(labels), 1)
            print(f"     {REGIME_NAMES.get(regime_id, regime_id)}: {count} ({pct:.0%})")

    def _print_rule_diagnostics(self, n_rows: int):
        if not self.last_rule_diagnostics:
            return
        print(
            "     Diagnostics: "
            f"trend∩volatile={int(self.last_rule_diagnostics.get('trend_volatile_overlap_count', 0)):,} | "
            f"trending_share={float(self.last_rule_diagnostics.get('trending_share', 0.0)):.1%} | "
            f"volatile_share={float(self.last_rule_diagnostics.get('volatile_share', 0.0)):.1%} | "
            f"trend_masked={int(self.last_rule_diagnostics.get('trending_masked_by_priority_count', 0)):,}/{max(int(n_rows), 1):,}"
        )

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

        probs = self.predict_regime_posteriors(df)
        if probs.empty:
            return np.zeros(len(df), dtype=np.int8)
        return np.argmax(probs.values, axis=1).astype(np.int8)

    def predict_regime_posteriors(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted or len(df) == 0:
            return pd.DataFrame(
                np.zeros((len(df), len(REGIME_ONE_HOT_COLS)), dtype=np.float32),
                index=df.index,
                columns=list(REGIME_ONE_HOT_COLS),
            )

        X = _build_regime_features(df)
        if self.model is None or (self.model_type or '').strip().lower() in ('auto', 'rules'):
            labels = self._predict_rules(X)
            scores = _build_rule_scores(X, feature_stats=(self.rule_stats or {}).get('feature_stats', {}))
            ranging_score = (
                1.0
                - np.maximum.reduce(
                    [
                        scores[REGIME_META_SCORE_COLS[0]].to_numpy(dtype=np.float64),
                        scores[REGIME_META_SCORE_COLS[1]].to_numpy(dtype=np.float64),
                        scores[REGIME_META_SCORE_COLS[2]].to_numpy(dtype=np.float64),
                    ]
                )
            )
            base = np.full((len(X), len(REGIME_ONE_HOT_COLS)), 1e-3, dtype=np.float64)
            base[:, 0] += scores[REGIME_META_SCORE_COLS[0]].to_numpy(dtype=np.float64) * 0.55
            base[:, 2] += scores[REGIME_META_SCORE_COLS[1]].to_numpy(dtype=np.float64) * 0.55
            base[:, 3] += scores[REGIME_META_SCORE_COLS[2]].to_numpy(dtype=np.float64) * 0.55
            base[:, 1] += np.clip(ranging_score, 0.0, 1.0) * 0.55
            base[np.arange(len(labels), dtype=np.int32), np.clip(labels, 0, len(REGIME_ONE_HOT_COLS) - 1)] += 0.45
            probs = _normalize_prob_rows(base)
        else:
            X_scaled = self.scaler.transform(X)
            raw_probs = None
            if hasattr(self.model, 'predict_proba'):
                try:
                    raw_probs = np.asarray(self.model.predict_proba(X_scaled), dtype=np.float64)
                except Exception:
                    raw_probs = None
            if raw_probs is None:
                raw_labels = np.asarray(self.model.predict(X_scaled), dtype=np.int32)
                raw_probs = np.zeros((len(raw_labels), self.n_regimes), dtype=np.float64)
                raw_probs[np.arange(len(raw_labels), dtype=np.int32), np.clip(raw_labels, 0, self.n_regimes - 1)] = 1.0

            probs = np.zeros((len(raw_probs), len(REGIME_ONE_HOT_COLS)), dtype=np.float64)
            for cluster_idx in range(raw_probs.shape[1]):
                regime_idx = int(self._regime_map.get(int(cluster_idx), 1))
                probs[:, regime_idx] += raw_probs[:, cluster_idx]
            probs = _normalize_prob_rows(probs)

        probs = _normalize_prob_rows(probs)
        probs32 = probs.astype(np.float32)
        if probs32.shape[1] > 1:
            probs32[:, -1] = np.float32(1.0) - np.sum(probs32[:, :-1], axis=1, dtype=np.float32)
        return pd.DataFrame(
            probs32,
            index=df.index,
            columns=list(REGIME_ONE_HOT_COLS),
        )

    def predict_regime_meta(self, df: pd.DataFrame) -> pd.DataFrame:
        posteriors = self.predict_regime_posteriors(df)
        scores = self.predict_scores(df)
        out = pd.DataFrame(
            np.concatenate(
                [
                    posteriors.reindex(columns=list(REGIME_ONE_HOT_COLS), fill_value=0.0).values.astype(np.float32),
                    np.zeros((len(df), len(REGIME_META_SCORE_COLS)), dtype=np.float32),
                ],
                axis=1,
            ),
            index=df.index,
            columns=[*REGIME_ONE_HOT_COLS, *REGIME_META_SCORE_COLS],
        )
        for col in REGIME_META_SCORE_COLS:
            out[col] = scores.reindex(index=df.index, columns=[col], fill_value=0.0)[col].astype(np.float32)
        return out

    def predict_current(self, recent_bars: pd.DataFrame) -> dict:
        if not self._fitted or len(recent_bars) == 0:
            return {'regime_id': 0, 'regime_name': 'Trending', 'tradeable': True, 'confidence': 0.5}
        recent_posteriors = self.predict_regime_posteriors(recent_bars)
        if recent_posteriors.empty:
            return {'regime_id': 0, 'regime_name': 'Trending', 'tradeable': True, 'confidence': 0.5}
        smoothed = recent_posteriors.tail(min(10, len(recent_posteriors))).mean(axis=0)
        regime_id = int(np.argmax(smoothed.values))
        conf = float(smoothed.iloc[regime_id]) if len(smoothed) else 0.5

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


# ═══════════════════════════════════════════════════════════════════
# التعديل 4 — Wasserstein Regime Classifier
# ═══════════════════════════════════════════════════════════════════

class WassersteinRegimeClassifier:
    """
    التعديل 4: تصنيف الـ Regime باستخدام Wasserstein Distance.

    لماذا Wasserstein بدلاً من GMM؟
      - GMM يعتمد على Euclidean distance → يميل للـ Ranging (59%) لأن
        معظم النقاط قريبة من المتوسط
      - Wasserstein يقيس المسافة بين توزيعات العوائد كاملة
      - يميّز Trending الخفي (moves منتظمة) من Ranging (moves عشوائية)
        حتى لو المتوسطات متقاربة

    الخوارزمية:
      1. قسّم البيانات لـ windows متداخلة (rolling)
      2. لكل window احسب distribution of returns
      3. صنّف بناءً على مسافة Wasserstein من prototype distributions
    """

    REGIME_NAMES = {0: 'Trending', 1: 'Ranging', 2: 'Volatile', 3: 'Low_Liquidity'}

    def __init__(
        self,
        window: int = 50,
        n_regimes: int = 4,
        percentile_trending: float = 30.0,
        percentile_volatile: float = 70.0,
        low_liq_threshold: float = 0.1,
        progress_every: int = 0,
    ):
        self.window              = max(int(window), 10)
        self.n_regimes           = n_regimes
        self.pct_trending        = percentile_trending
        self.pct_volatile        = percentile_volatile
        self.low_liq_thr         = low_liq_threshold
        self.progress_every      = max(int(progress_every), 0)
        self._vol_low_thr        = None
        self._vol_high_thr       = None
        self._trend_thr          = None
        self._liq_thr            = None
        self._fitted             = False
        self._feature_cache_key  = None
        self._feature_cache      = None

    @staticmethod
    def _frame_cache_key(prices: np.ndarray, volumes: np.ndarray) -> tuple:
        if len(prices) == 0:
            return (0, 0.0, 0.0, 0.0, 0.0)
        return (
            int(len(prices)),
            float(prices[0]),
            float(prices[-1]),
            float(volumes[0]) if len(volumes) else 0.0,
            float(volumes[-1]) if len(volumes) else 0.0,
        )

    def _rolling_features(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        *,
        progress_label: str = '',
    ) -> np.ndarray:
        """
        يحسب لكل نافذة:
          [vol_std, trend_score, liq_score, skewness]
        """
        cache_key = self._frame_cache_key(prices, volumes)
        if self._feature_cache_key == cache_key and self._feature_cache is not None:
            return self._feature_cache.copy()

        price_s = pd.Series(prices, dtype=np.float64)
        vol_s = pd.Series(volumes, dtype=np.float64)
        ret_s = price_s.diff().fillna(0.0)

        # Wasserstein to a zero reference in 1D collapses here to a rolling L1 move proxy.
        vol_std = ret_s.rolling(self.window, min_periods=3).std().fillna(0.0)
        l1_move = ret_s.abs().rolling(self.window, min_periods=2).mean().fillna(0.0)
        trend_score = (l1_move / vol_std.clip(lower=1e-10)).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        all_vol_med = float(np.median(volumes)) if len(volumes) > 0 else 1.0
        liq_score = (
            vol_s.rolling(self.window, min_periods=1).mean().fillna(0.0)
            / max(all_vol_med, 1e-10)
        )
        skew = ret_s.rolling(self.window, min_periods=3).skew().fillna(0.0)

        feats = np.column_stack(
            [
                vol_std.to_numpy(dtype=np.float32),
                trend_score.to_numpy(dtype=np.float32),
                liq_score.to_numpy(dtype=np.float32),
                skew.to_numpy(dtype=np.float32),
            ]
        )
        self._feature_cache_key = cache_key
        self._feature_cache = feats.copy()
        return feats

    def fit(self, df: pd.DataFrame) -> 'WassersteinRegimeClassifier':
        price_series = (
            df['price']
            if 'price' in df.columns
            else df['close']
        )
        prices = _past_only_numeric_series(
            price_series,
            index=df.index,
            context='wasserstein_regime.fit.price',
            fill_value=0.0,
        ).to_numpy(dtype=np.float64)
        volumes = df['volume'].fillna(0).values.astype(np.float64) \
                  if 'volume' in df.columns else \
                  pd.to_numeric(df.get('size', pd.Series(np.ones(len(df)), index=df.index)), errors='coerce').fillna(0).values.astype(np.float64)

        feats = self._rolling_features(prices, volumes, progress_label='fit')
        vol_arr   = feats[:, 0]
        trend_arr = feats[:, 1]
        liq_arr   = feats[:, 2]

        # حساب thresholds على Train data فقط
        self._vol_low_thr  = float(np.percentile(vol_arr[vol_arr > 0],   self.pct_trending))
        self._vol_high_thr = float(np.percentile(vol_arr[vol_arr > 0],   self.pct_volatile))
        self._trend_thr    = float(np.percentile(trend_arr[trend_arr > 0], 50.0))
        self._liq_thr      = float(np.percentile(liq_arr, self.low_liq_thr * 100))
        self._fitted       = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("WassersteinRegimeClassifier: call fit() first")

        price_series = (
            df['price']
            if 'price' in df.columns
            else df['close']
        )
        prices = _past_only_numeric_series(
            price_series,
            index=df.index,
            context='wasserstein_regime.predict.price',
            fill_value=0.0,
        ).to_numpy(dtype=np.float64)
        volumes = df['volume'].fillna(0).values.astype(np.float64) \
                  if 'volume' in df.columns else \
                  pd.to_numeric(df.get('size', pd.Series(np.ones(len(df)), index=df.index)), errors='coerce').fillna(0).values.astype(np.float64)

        feats     = self._rolling_features(prices, volumes, progress_label='predict')
        vol_arr   = feats[:, 0]
        trend_arr = feats[:, 1]
        liq_arr   = feats[:, 2]

        labels = np.full(len(df), 1, dtype=np.int8)  # default = Ranging

        # Low Liquidity
        labels[liq_arr < self._liq_thr] = 3

        # Volatile
        labels[vol_arr > self._vol_high_thr] = 2

        # Trending (يتجاوز كل ما سبق)
        trending_mask = (trend_arr > self._trend_thr) & (vol_arr <= self._vol_high_thr)
        labels[trending_mask] = 0

        return labels

    def fit_predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).predict(df)
