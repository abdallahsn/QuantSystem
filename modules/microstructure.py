import numpy as np
import pandas as pd
from collections import deque
import time

TRADE_ACTIONS = {'TRADE', 'T', 'E', 'EXECUTE', 'F', '0'}

class FastMicrostructureEngine:
    """
    Hidden Volume Ratio (HVR) — NQ Iceberg Detection
    تم تصحيح منطق اكتشاف الأوامر المخفية وإدارة الذاكرة.
    """

    def __init__(self, window: int = 200, time_limit_ms: int = 500):
        self.slice_count  = {}   
        self.slice_hist   = deque(maxlen=window)   
        self.pending_fill = {}   
        # لحماية الذاكرة من الأوامر المعلقة للأبد
        self.time_limit_ns = time_limit_ms * 1_000_000 
        self.last_cleanup = time.time()

    def process_mbo_tick(self, action: str, order_id, side: str, size: float, price: float, ts_event: int) -> float:
        action = str(action).strip().upper()
        side   = str(side).strip().upper()
        hvr    = 0.0

        if action in ('A', 'ADD'):
            # Iceberg يُعرف بتكرار نفس السعر والاتجاه (وليس بالضرورة نفس الحجم بالضبط كل مرة)
            # مفتاح التجميع هو (Price, Side) وليس (Size, Side)
            key = (price, side)
            
            self.pending_fill[order_id] = {'key': key, 'ts': ts_event}

            # زيادة العداد لهذا السعر والاتجاه
            self.slice_count[key] = self.slice_count.get(key, 0) + 1
            count = self.slice_count[key]
            
            self.slice_hist.append(count)

            if len(self.slice_hist) >= 10:
                max_slices = max(self.slice_hist)
                if max_slices > 2:
                    hvr = min(count / max_slices, 1.0)

        elif action in TRADE_ACTIONS:
            # لو اتنفذ، بنخليه يكمل عد عادي لأن الـ Iceberg بيجدد الكمية
            if order_id in self.pending_fill:
                order_info = self.pending_fill[order_id]
                key = order_info['key']
                
                if key in self.slice_count:
                    count = self.slice_count[key]
                    self.slice_hist.append(count)
                    if len(self.slice_hist) >= 10:
                        max_slices = max(self.slice_hist)
                        if max_slices > 2:
                            hvr = min(count / max_slices, 1.0)

        elif action in ('C', 'CANCEL'):
            if order_id in self.pending_fill:
                order_info = self.pending_fill.pop(order_id)
                key = order_info['key']
                
                # تقليل العداد لأن الأمر اتلغى ولم ينفذ
                if key in self.slice_count:
                    self.slice_count[key] -= 1
                    if self.slice_count[key] <= 0:
                        del self.slice_count[key]

        # تنظيف دوري ذكي (مبني على الزمن وليس العدد فقط) لمنع الـ Memory Leak
        current_time = time.time()
        if current_time - self.last_cleanup > 5.0:  # كل 5 ثواني
            self._cleanup_stale_orders(ts_event)
            self.last_cleanup = current_time

        return round(hvr, 6)

    def _cleanup_stale_orders(self, current_ts: int):
        # حذف الأوامر التي مر عليها وقت طويل دون تنفيذ أو إلغاء
        # current_ts و info['ts'] كلاهما int (nanoseconds) ← نطرح int من int مباشرة
        stale_threshold = current_ts - self.time_limit_ns
        keys_to_delete = [order_id for order_id, info in self.pending_fill.items() if info['ts'] < stale_threshold]
        
        for k in keys_to_delete:
            del self.pending_fill[k]
            
        # تنظيف الـ slice_count من الأسعار القديمة
        if len(self.slice_count) > 1000:
            self.slice_count.clear()


class CancelRatioEngine:
    """
    Cancel Ratio — نسبة الأوامر الكبيرة التي تلغى للمناورة.
    تم تصحيح خلل تسجيل الأحداث وحساب المتوسط.
    """

    def __init__(self, window: int = 500, large_mult: float = 3.0):
        self.window      = window
        self.large_mult  = large_mult
        self._active     = {}          
        self._size_hist  = deque(maxlen=200)
        self._mean_size  = 1.0
        # نافذة لحساب نسبة (الإلغاء الكبير / إجمالي الإضافات)
        self._events     = deque(maxlen=window)  

    def process_tick(self, action: str, order_id, size: float) -> float:
        action = str(action).strip().upper()
        size   = float(size) if size else 0.0

        if action in ('A', 'ADD'):
            self._active[order_id] = size
            self._size_hist.append(size)
            if len(self._size_hist) >= 10:
                self._mean_size = float(np.mean(self._size_hist))
            
            # تسجيل حدث إضافة (المقام سيزيد)
            self._events.append(0)

        elif action in ('C', 'CANCEL'):
            # ✅ FIX: سجّل الإلغاء بغضّ النظر عن تطابق order_id
            # (في بعض feeds، order_id يتغير أو يُرسل بشكل منفصل)
            if order_id in self._active:
                orig_size = self._active.pop(order_id)
                is_large  = orig_size >= (self._mean_size * self.large_mult)
                weight = 1.0 if is_large else 0.3
            else:
                # cancel بدون add مطابق → نستخدم الـ mean_size للتقدير
                is_large = False
                weight = 0.3
            self._events.append(weight)

        elif action in TRADE_ACTIONS:
            self._active.pop(order_id, None)

        if len(self._active) > 50_000:
            # تنظيف الذاكرة عشوائياً/بسرعة
            keys_to_remove = list(self._active.keys())[:10_000]
            for k in keys_to_remove:
                del self._active[k]

        # ✅ FIX: خفّض minimum من 10 إلى 3 لاكتشاف cancel patterns مبكراً
        if len(self._events) < 3:
            return 0.0

        # النسبة هي مجموع الإلغاءات الكبيرة / إجمالي الأحداث في النافذة
        return round(float(np.mean(self._events)), 6)


class AbsorptionIntensityEngine:
    """
    Absorption Intensity Index (AII)
    """

    def __init__(self, window: int = 50, min_price_move: float = 0.25):
        self.window         = window
        self.min_price_move = min_price_move
        self._prices  = deque(maxlen=window + 1)
        self._cvds    = deque(maxlen=window + 1)
        self._aii_buf = deque(maxlen=window * 4)

    def update(self, price: float, cvd: float) -> float:
        self._prices.append(price)
        self._cvds.append(cvd)
        
        if len(self._prices) < self.window:
            return 0.0
            
        d_cvd   = abs(float(self._cvds[-1])   - float(self._cvds[0]))
        d_price = max(abs(float(self._prices[-1]) - float(self._prices[0])),
                      self.min_price_move)
                      
        raw = d_cvd / d_price
        self._aii_buf.append(raw)
        
        if len(self._aii_buf) >= 20:
            # Use robust center scaling to preserve spike structure instead of
            # over-compressing around p95.
            med = float(np.median(self._aii_buf))
            scale = max(med, 1e-6)
            return round(min(raw / scale, 10.0), 6)

        return round(min(raw / 500.0, 10.0), 6)


class FastTapeSpeedTracker:
    def __init__(self, window_seconds: float = 1.0):
        self.trade_times = deque()
        self.window      = pd.Timedelta(seconds=window_seconds)

    def update_and_get_speed(self, timestamp, action) -> int:
        if str(action).strip().upper() in TRADE_ACTIONS:
            self.trade_times.append(timestamp)
        cutoff = timestamp - self.window
        while self.trade_times and self.trade_times[0] < cutoff:
            self.trade_times.popleft()
        return len(self.trade_times)