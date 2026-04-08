import numpy as np
from collections import deque

TRADE_ACTIONS = {'T', 'F', 'TRADE', 'EXECUTE', 'E', '0'}

class MicroVolatilityEngine:
    """
    يحسب الثلاث features من تدفق MBO أو MBP مباشرة.
    محسّن رياضياً لمنع تسرب البيانات وضبط دقة الحسابات.
    """

    def __init__(self,
                 window_atr: int = 100,    
                 window_vbi: int = 500,    
                 window_iet: int = 50):    

        self.window_atr = window_atr
        self.window_vbi = window_vbi
        self.window_iet = window_iet

        # MicroATR
        self._last_price = None
        self._tick_moves = deque(maxlen=window_atr)

        # Volume Burst
        self._sizes      = deque(maxlen=window_vbi)
        self._last_volume_burst = 1.0 # للاحتفاظ بآخر قيمة بدل إرجاع 1.0 دايماً

        # Inter-Event Time
        self._last_ts    = None
        self._intervals  = deque(maxlen=window_iet)
        self._last_iet_ms = 0.0

    def process_tick(self, action: str, price: float, size: float, ts_ns: int) -> tuple:
        action = str(action).strip().upper()
        is_trade = action in TRADE_ACTIONS

        # ── 1. MicroATR ───────────────────────────────────────────
        if price > 0:
            if self._last_price is not None:
                # نحسب الفرق، حتى لو كان صفر (عشان لو السعر ثبت فترة، الـ ATR يقل)
                move = abs(price - self._last_price)
                self._tick_moves.append(move)
            self._last_price = price

        micro_atr = float(np.mean(self._tick_moves)) if len(self._tick_moves) >= 5 else 0.0

        # ── 2. Volume Burst & IET ─────────────────────────────────
        if is_trade and size > 0:
            # Volume Burst: الحساب السليم للمتوسط المتحرك
            if len(self._sizes) >= 10:
                # نحسب المتوسط *قبل* إضافة الحجم الحالي عشان نتجنب Data Leakage (الحجم الحالي ميكبرش المتوسط اللي بيتقسم عليه)
                mean_sz = float(np.mean(self._sizes))
                if mean_sz > 0:
                    self._last_volume_burst = size / mean_sz
            else:
                self._last_volume_burst = 1.0
                
            self._sizes.append(size)

            # Inter-Event Time
            if self._last_ts is not None:
                interval_ns = ts_ns - self._last_ts
                # حماية من الطوابع الزمنية المتطابقة (نفس الملي ثانية) أو الفاسدة
                if interval_ns >= 0: 
                    self._intervals.append(interval_ns / 1_000_000.0) # → ms
            
            self._last_ts = ts_ns

            if len(self._intervals) >= 5:
                self._last_iet_ms = float(np.mean(self._intervals))

        return (
            round(micro_atr, 6),
            round(self._last_volume_burst, 6),
            round(self._last_iet_ms, 4)
        )