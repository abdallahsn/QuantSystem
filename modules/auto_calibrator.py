"""
auto_calibrator.py — معايرة تلقائية من الداتا
"""
import numpy as np
import pandas as pd
from collections import Counter

class AutoCalibrator:
    def __init__(self, n_ticks: int = 2000):
        self.n_ticks = n_ticks
        self.tick_size      = 0.25   
        self.min_price_move = 0.25
        self.typical_size   = 1.0
        self.price_scale    = 1.0
        self.volatility     = 1.0
        self.mean_price     = 0.0
        self.symbol         = 'UNKNOWN'
        self._fitted        = False

    def fit(self, df: pd.DataFrame, price_col: str = 'price', size_col: str = 'size',
            action_col: str = 'action', symbol_col: str = 'symbol') -> 'AutoCalibrator':
        
        sample = df.head(self.n_ticks).copy()
        if symbol_col in sample.columns: self.symbol = str(sample[symbol_col].iloc[0]).strip()

        prices = pd.to_numeric(sample[price_col], errors='coerce').dropna()
        prices = prices[prices > 0]

        if len(prices) < 10: return self

        q1, q3 = prices.quantile(0.25), prices.quantile(0.75)
        iqr     = q3 - q1
        if iqr > 0:
            prices = prices[(prices >= q1 - 3 * iqr) & (prices <= q3 + 3 * iqr)]

        if len(prices) < 10: return self

        self.mean_price  = float(prices.mean())
        self.price_scale = float(prices.std()) if prices.std() > 0 else 1.0

        diffs = prices.diff().abs().dropna()
        diffs = diffs[diffs > 0]

        if len(diffs) > 10:
            counts  = Counter(diffs.round(4).tolist())
            candidates = [v for v, c in sorted(counts.items(), key=lambda x: -x[1]) if c >= 3][:10]
            self.tick_size = float(min(candidates)) if candidates else float(diffs.quantile(0.05))
            self.min_price_move = max(self.tick_size, float(diffs.quantile(0.25)))

        if size_col in sample.columns:
            sizes = pd.to_numeric(sample[size_col], errors='coerce').dropna()
            if len(sizes[sizes > 0]) > 10: self.typical_size = float(sizes[sizes > 0].median())

        if action_col in sample.columns:
            trade_mask = sample[action_col].astype(str).str.upper().isin({'T', 'F', 'TRADE', 'EXECUTE', 'E'})
            trade_prices = pd.to_numeric(sample.loc[trade_mask, price_col], errors='coerce').dropna()
        else:
            trade_prices = prices

        if len(trade_prices) > 10:
            self.volatility = max(float(trade_prices.diff().abs().dropna().mean()), self.tick_size)

        self._fitted = True
        return self

    def summary(self) -> str:
        return f"\n  🔧 Auto-Calibrator [{self.symbol}] | Tick: {self.tick_size:.4f} | Vol: {self.volatility:.4f}"

    def build_engines(self, include_extended: bool = False) -> dict:
        from modules.microstructure import AbsorptionIntensityEngine, CancelRatioEngine
        from modules.context_features import MomentumContextEngine, LiquiditySweepDetector

        absorb = AbsorptionIntensityEngine(min_price_move=self.min_price_move)
        cancel = CancelRatioEngine(large_mult=2.0)
        momentum = MomentumContextEngine(momentum_window=100, swing_window=200)
        sweep = LiquiditySweepDetector(sweep_threshold=max(0.1, min(self.volatility * 2, 0.3)))
        engines = {
            'absorb': absorb,
            'cancel': cancel,
            'momentum': momentum,
            'sweep': sweep,
        }

        if not include_extended:
            return engines

        from modules.fisher_alpha import FastFisherAlpha
        from modules.fim_anomaly import FastFIMDetector
        from modules.orderbook import SpoofingDetector
        from modules.market_research_features import (
            KylesLambdaEngine,
            HawkesIntensityEngine,
            LiquidityGapsEngine,
            VNETEngine,
        )

        mean_price = max(self.mean_price, self.tick_size, 1e-8)
        engines.update({
            'fisher': FastFisherAlpha(
                threshold=max(0.15, min(self.volatility / mean_price * 50, 0.5))
            ),
            'fim': FastFIMDetector(threshold_multiplier=2.0),
            'spoofing': SpoofingDetector(large_mult=1.5),
            'kyle': KylesLambdaEngine(window=50),
            'hawkes': HawkesIntensityEngine(alpha=0.7, beta=0.5),
            'gaps': LiquidityGapsEngine(levels=10, gap_threshold=2.0),
            'vnet': VNETEngine(window=100, large_mult=2.0),
        })
        return engines
