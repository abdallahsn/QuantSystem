import numpy as np
from collections import deque


# ══════════════════════════════════════════════════════════════════
# 1. Kyle's Lambda
# ══════════════════════════════════════════════════════════════════
class KylesLambdaEngine:
    """
    Kyle's Lambda = تكلفة تحريك السعر بوحدة حجم واحدة
    """

    def __init__(self, window: int = 50):
        self.window     = window
        self._prices    = deque(maxlen=window + 1)
        self._volumes   = deque(maxlen=window)
        self._lambdas   = deque(maxlen=window)

    def update(self, price: float, volume: float) -> float:
        self._prices.append(price)
        self._volumes.append(max(volume, 1e-8))

        if len(self._prices) < 2:
            return 0.0

        dp = abs(float(self._prices[-1]) - float(self._prices[-2]))
        dv = float(self._volumes[-1])
        
        # λ = ΔP / V 
        lam = dp / dv if dv > 0 else 0.0
        self._lambdas.append(lam)

        # تحسين: حساب الـ Z-Score لـ Kyle's Lambda ليكون معبر أكثر للموديل (Stationary)
        if len(self._lambdas) >= 10:
            arr = np.array(self._lambdas)
            mean_lam = float(np.mean(arr))
            std_lam = float(np.std(arr)) + 1e-8
            z_score = (lam - mean_lam) / std_lam
            return round(float(np.clip(z_score, -4.0, 4.0)), 4)
            
        return 0.0


# ══════════════════════════════════════════════════════════════════
# 2. Hawkes Process — Event Clustering
# ══════════════════════════════════════════════════════════════════
class HawkesIntensityEngine:
    """
    محسّن رياضياً ليعمل بسرعة O(1) بدلاً من O(N) لكل حدث.
    """

    def __init__(self, baseline: float = 0.1, alpha: float = 0.5, beta: float = 1.0):
        self.baseline   = baseline
        self.alpha      = alpha
        self.beta       = beta
        
        self._last_ts   = None
        self._intensity = 0.0  # Recursive computation
        self._history   = deque(maxlen=100)

    def update(self, ts_ns: int, action: str) -> float:
        # نقبل Add و Cancel فقط كمؤشرات تلاعب (Spoofing)
        is_event = str(action).strip().upper() in ('A', 'ADD', 'C', 'CANCEL')
        
        if self._last_ts is None:
            self._last_ts = ts_ns
            self._intensity = 0.0
            return self.baseline

        # تحويل الزمن لثواني
        dt = (ts_ns - self._last_ts) / 1e9
        
        # حماية من التواريخ الفاسدة
        if dt < 0: return self.baseline
            
        # Recursive Hawkes Update (السر هنا لـ O(1) performance)
        # λ(t) = λ_0 + (λ(t-1) - λ_0) * exp(-β * Δt) + α * I(event)
        decay = np.exp(-self.beta * dt)
        self._intensity = (self._intensity) * decay
        
        if is_event:
            self._intensity += self.alpha

        self._last_ts = ts_ns
        
        current_val = self.baseline + self._intensity
        self._history.append(current_val)

        # تطبيع النتيجة (Normalization)
        if len(self._history) >= 10:
            mean_val = float(np.mean(self._history))
            if mean_val > 0:
                return round(float(current_val / mean_val), 4)

        return round(float(current_val), 4)


# ══════════════════════════════════════════════════════════════════
# 3. Liquidity Gaps
# ══════════════════════════════════════════════════════════════════
class LiquidityGapsEngine:
    """
    يكشف الفراغات والكثافة في الـ Order Book
    """

    def __init__(self, levels: int = 10, gap_threshold: float = 1.5):
        self.levels        = levels
        self.gap_threshold = gap_threshold
        self._gaps         = deque(maxlen=50)

    def update(self, row: dict, tick_size: float = 0.0001) -> float:
        tick = max(tick_size, 1e-8)

        # القراءة السريعة والمضمونة بدون Exceptions
        bid_px = [float(row.get(f'bid_px_{i:02d}', 0) or 0) for i in range(self.levels)]
        ask_px = [float(row.get(f'ask_px_{i:02d}', 0) or 0) for i in range(self.levels)]
        bid_sz = [int(row.get(f'bid_sz_{i:02d}', 0) or 0) for i in range(self.levels)]
        ask_sz = [int(row.get(f'ask_sz_{i:02d}', 0) or 0) for i in range(self.levels)]

        valid_bids = [(p, s) for p, s in zip(bid_px, bid_sz) if p > 0]
        valid_asks = [(p, s) for p, s in zip(ask_px, ask_sz) if p > 0]

        if len(valid_bids) < 2 or len(valid_asks) < 2:
            return 0.0

        gaps = 0.0
        total_checks = 0

        # فحص الفجوات السعرية (المسافة بين المستويات)
        for i in range(len(valid_bids) - 1):
            total_checks += 1
            diff = abs(valid_bids[i][0] - valid_bids[i+1][0])
            if diff > tick * self.gap_threshold:
                gaps += 1.0
            elif valid_bids[i][1] == 0:
                gaps += 0.5 # مستوى وهمي/مخترق

        for i in range(len(valid_asks) - 1):
            total_checks += 1
            diff = abs(valid_asks[i][0] - valid_asks[i+1][0])
            if diff > tick * self.gap_threshold:
                gaps += 1.0
            elif valid_asks[i][1] == 0:
                gaps += 0.5

        # Spread Penalty
        spread = abs(valid_asks[0][0] - valid_bids[0][0])
        if spread > tick * 2:
            gaps += 1.0
            total_checks += 1

        gap_score = gaps / max(total_checks, 1)
        self._gaps.append(gap_score)

        return round(float(np.mean(self._gaps)) if len(self._gaps) >= 3 else gap_score, 4)


# ══════════════════════════════════════════════════════════════════
# 4. VNET — Volume Net
# ══════════════════════════════════════════════════════════════════
class VNETEngine:
    """
    VNET = حجم صافي الاتجاه
    تم تصحيح اتجاه الإشارة ليعكس اتجاه الـ Smart Money بدقة.
    """

    def __init__(self, window: int = 100, large_mult: float = 2.0):
        self.window      = window
        self.large_mult  = large_mult
        self._trades     = deque(maxlen=window)
        self._size_hist  = deque(maxlen=200)

    def update(self, price: float, size: float, side: str) -> float:
        size  = float(size)
        side  = str(side).upper()

        if size <= 0: return 0.0

        self._size_hist.append(size)
        mean_size = float(np.mean(self._size_hist)) if len(self._size_hist) >= 10 else 1.0

        is_large = size >= (mean_size * self.large_mult)
        
        # نعطي وزن مضاعف للأوامر الكبيرة (Smart Money)
        weight = 2.0 if is_large else 1.0

        # الاتجاه يتم تحديده من الـ aggressor side (Buy=الشراء من العرض، Sell=البيع للطلب)
        if side in ('B', 'BID'):
            direction = 1.0
        elif side in ('A', 'S', 'ASK', 'SELL'):
            direction = -1.0
        else:
            direction = 0.0

        # حجم ذو إشارة (Signed Volume) 
        signed_vol = size * weight * direction
        actual_vol = size * weight

        self._trades.append((signed_vol, actual_vol))

        if len(self._trades) < 5: return 0.0

        total_signed = sum(t[0] for t in self._trades)
        total_volume = sum(t[1] for t in self._trades)

        if total_volume == 0: return 0.0

        return round(float(np.clip(total_signed / total_volume, -1.0, 1.0)), 4)