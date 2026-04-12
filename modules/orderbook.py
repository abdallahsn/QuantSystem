import numpy as np
from collections import deque

# أعمدة MBP10 الثابتة
BID_SZ = [f'bid_sz_0{i}' for i in range(10)]
ASK_SZ = [f'ask_sz_0{i}' for i in range(10)]
BID_CT = [f'bid_ct_0{i}' for i in range(10)]
ASK_CT = [f'ask_ct_0{i}' for i in range(10)]
BID_PX = [f'bid_px_0{i}' for i in range(10)]
ASK_PX = [f'ask_px_0{i}' for i in range(10)]

# أوزان الـ levels: L0 أهم من L1 أهم من L2...
LEVEL_WEIGHTS = np.array([1.0 / (i + 1) for i in range(10)], dtype=np.float64)

TRADE_ACTIONS = {'T', 'F', 'TRADE', 'EXECUTE', 'E', '0'} # تم إضافة 0 لدعم Databento


def _safe_float(val, default=0.0):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


class OrderBookSnapshotEngine:
    """
    يحسب OBI من كل snapshot MBP10.
    V16Pro: أضاف OBI Z-Score Dynamic
    """

    def __init__(self, zscore_window: int = 200):
        self._obi_history = deque(maxlen=zscore_window)

    def compute_obi(self, row) -> float:
        bid_vols = np.array([_safe_float(row.get(c, 0)) for c in BID_SZ], dtype=np.float64)
        ask_vols = np.array([_safe_float(row.get(c, 0)) for c in ASK_SZ], dtype=np.float64)

        w_bid = float(np.dot(bid_vols, LEVEL_WEIGHTS))
        w_ask = float(np.dot(ask_vols, LEVEL_WEIGHTS))
        total = w_bid + w_ask

        if total == 0:
            return 0.0

        raw_obi = float(np.clip((w_bid - w_ask) / total, -1.0, 1.0))
        self._obi_history.append(raw_obi)

        return raw_obi

    @staticmethod
    def compute_spread(row) -> float:
        bid = _safe_float(row.get('bid_px_00', 0))
        ask = _safe_float(row.get('ask_px_00', 0))
        if bid > 0 and ask > 0 and ask > bid:
            return ask - bid
        return 0.0

    @staticmethod
    def total_book_depth(row) -> float:
        bid_total = sum(_safe_float(row.get(c, 0)) for c in BID_SZ)
        ask_total = sum(_safe_float(row.get(c, 0)) for c in ASK_SZ)
        return bid_total + ask_total


class SpoofingDetector:
    """
    Spoofing Detection لبيئة الـ MBP10 (Snapshot/Bar level)
    تمت إعادة هندسته بالكامل لكشف الانسحابات الوهمية للسيولة
    """

    def __init__(self, large_mult: float = 1.5, window: int = 50):
        self.large_mult   = large_mult
        self.window       = window
        self._prev        = {}
        
        # تتبع الانسحابات (Spoofs)
        self._spoof_count = deque(maxlen=window)
        self._trade_count = deque(maxlen=window)
        
        self._size_hist   = deque(maxlen=200)
        self._mean_size   = 1.0

    def process_snapshot(self, row: dict, action: str) -> tuple:
        """
        يكتشف الـ Spoofing من خلال مقارنة الـ Snapshot الحالي بالسابق.
        Spoofing = اختفاء مفاجئ لكمية كبيرة من الـ Book بدون حدوث Trade.
        """
        action = str(action).strip().upper()
        
        # استخراج السيولة في أول 3 مستويات (الأكثر عرضة للـ Spoofing)
        current_bid_vol = sum(_safe_float(row.get(f'bid_sz_0{i}', 0)) for i in range(3))
        current_ask_vol = sum(_safe_float(row.get(f'ask_sz_0{i}', 0)) for i in range(3))
        
        curr_max = max(current_bid_vol, current_ask_vol)
        if curr_max > 0:
            self._size_hist.append(curr_max)
            
        if len(self._size_hist) >= 10:
            self._mean_size = float(np.mean(self._size_hist))

        # جلب البيانات السابقة
        prev_bid_vol = self._prev.get('bid_vol', current_bid_vol)
        prev_ask_vol = self._prev.get('ask_vol', current_ask_vol)

        spoof_ratio = 0.0
        spoof_duration = 0.0 # غير دقيق حسابه في بيئة الـ MBP، سنتركه كإشارة قوة السحب

        # إذا كانت الحركة الحالية Trade، نسجلها لدعم المقام (Denominator) في نسبة الـ Spoofing
        if action in TRADE_ACTIONS:
            self._trade_count.append(1)
            self._spoof_count.append(0)
        else:
            self._trade_count.append(0)
            
            # حساب الانسحاب (Drop) في السيولة
            bid_drop = prev_bid_vol - current_bid_vol
            ask_drop = prev_ask_vol - current_ask_vol
            
            size_threshold = self._mean_size * self.large_mult

            # هل حدث انسحاب وهمي كبير؟ (بدون Trade)
            is_spoof = 0
            max_drop = 0.0
            
            if bid_drop > size_threshold:
                is_spoof = 1
                max_drop = bid_drop
            elif ask_drop > size_threshold:
                is_spoof = 1
                max_drop = ask_drop
                
            self._spoof_count.append(is_spoof)
            if is_spoof:
                # نستخدم الـ Duration هنا للتعبير عن "حجم/قوة" الانسحاب كبديل زمني
                spoof_duration = max_drop / max(self._mean_size, 1.0) 

        # تحديث الذاكرة
        self._prev = {'bid_vol': current_bid_vol, 'ask_vol': current_ask_vol}

        # حساب الـ Ratio (إجمالي الانسحابات الوهمية / إجمالي الصفقات المنفذة) في النافذة
        total_spoofs = sum(self._spoof_count)
        total_trades = max(sum(self._trade_count), 1) # حماية من القسمة على صفر
        
        spoof_ratio = min(total_spoofs / total_trades, 1.0) # تقييد بـ 1.0 كحد أقصى

        return round(spoof_ratio, 6), round(spoof_duration, 4)


class LiquidityTrapDetector:
    """
    Liquidity Trap Score:
    يكشف فخاخ السيولة بناءً على علاقة الـ OBI بحركة السعر.
    """

    def __init__(self, price_move_threshold: float = 0.0002):
        self.threshold  = price_move_threshold
        self._last_obi  = 0.0
        self._last_price= None
        self._lts_buffer= deque(maxlen=100)

    def update(self, price: float, obi: float) -> float:
        if self._last_price is None:
            self._last_price = price
            self._last_obi   = obi
            return 0.0

        delta = price - self._last_price

        # إذا تجاوز السعر العتبة المحددة (Tick Size أو Pips)
        if abs(delta) >= self.threshold:
            direction = 1.0 if delta > 0 else -1.0
            
            # LTS المعادلة: OBI السابق * اتجاه السعر * -1
            # إذا كان OBI شرائي (+) والسعر هبط (-) -> النتيجة الموجبة = فخ شراء
            lts = self._last_obi * direction * -1.0
            self._lts_buffer.append(lts)
            self._last_price = price

        self._last_obi = obi
        return float(np.mean(self._lts_buffer)) if self._lts_buffer else 0.0
