"""
context_features.py — السياق الكامل للسوق
"""
import numpy as np
import pandas as pd
from collections import deque

def compute_daily_weekly_levels(df_all: pd.DataFrame, price_col: str = 'price', ts_col: str = 'ts_event') -> pd.DataFrame:
    df = df_all.copy()
    ts_series = pd.to_datetime(df[ts_col], utc=True, errors='coerce')
    ts_series = ts_series.dt.tz_localize(None)
    dates = ts_series.dt.date
    weeks = ts_series.dt.to_period('W')

    daily_hl = df.groupby(dates)[price_col].agg(['max', 'min'])
    weekly_hl = df.groupby(weeks)[price_col].agg(['max', 'min'])

    daily_hl.index = daily_hl.index + pd.Timedelta(days=1)
    weekly_hl.index = weekly_hl.index + 1

    df['pdh'] = dates.map(daily_hl['max']).ffill().fillna(df[price_col])
    df['pdl'] = dates.map(daily_hl['min']).ffill().fillna(df[price_col])
    df['pwh'] = weeks.map(weekly_hl['max']).ffill().fillna(df[price_col])
    df['pwl'] = weeks.map(weekly_hl['min']).ffill().fillna(df[price_col])

    df['dist_to_pdh'] = (df['pdh'] - df[price_col]).round(4)
    df['dist_to_pdl'] = (df[price_col] - df['pdl']).round(4)

    rng = (df['pdh'] - df['pdl']).replace(0, np.nan)
    df['price_position'] = ((df[price_col] - df['pdl']) / rng).clip(0, 1).fillna(0.5)

    return df

class MomentumContextEngine:
    def __init__(self, momentum_window: int = 100, swing_window: int = 200):
        self.mom_win   = momentum_window
        self.swing_win = swing_window
        self._cvd_hist   = deque(maxlen=swing_window)
        self._price_hist = deque(maxlen=swing_window)
        self._trend_prices = deque(maxlen=momentum_window)

    def update(self, price: float, cvd: float) -> tuple:
        self._cvd_hist.append(cvd)
        self._price_hist.append(price)
        self._trend_prices.append(price)

        if len(self._cvd_hist) >= self.mom_win:
            cvd_now  = float(self._cvd_hist[-1])
            cvd_past = float(self._cvd_hist[-self.mom_win])
            cvd_range = max(abs(cvd_now), abs(cvd_past), 1)
            cvd_momentum = round((cvd_now - cvd_past) / cvd_range, 4)
        else:
            cvd_momentum = 0.0

        divergence = 0.0
        if len(self._cvd_hist) >= self.swing_win:
            n = self.swing_win
            prices = list(self._price_hist)[-n:]
            cvds   = list(self._cvd_hist)[-n:]
            half = n // 2
            price_dir = np.sign(np.mean(prices[half:]) - np.mean(prices[:half]))
            cvd_dir   = np.sign(np.mean(cvds[half:])   - np.mean(cvds[:half]))

            if price_dir == cvd_dir and price_dir != 0: divergence = 1.0
            elif price_dir != cvd_dir and price_dir != 0: divergence = -1.0

        trend_strength = 0.0
        if len(self._trend_prices) >= 20:
            prices_arr = np.array(list(self._trend_prices))
            price_range = prices_arr.max() - prices_arr.min()
            total_move  = abs(prices_arr[-1] - prices_arr[0])
            if price_range > 0:
                trend_strength = round(min(total_move / price_range, 1.0), 4)

        correction_depth = 0.0
        if len(self._price_hist) >= 20:
            prices_arr = np.array(list(self._price_hist))
            recent_max = prices_arr.max(); recent_min = prices_arr.min()
            current    = prices_arr[-1]
            rng        = recent_max - recent_min

            if rng > 0:
                if prices_arr[-1] > prices_arr[0]: correction_depth = round((recent_max - current) / rng, 4)
                else: correction_depth = round((current - recent_min) / rng, 4)

        return (cvd_momentum, divergence, trend_strength, correction_depth)

class LiquiditySweepDetector:
    def __init__(self, lookback: int = 300, sweep_threshold: float = 0.05):
        self.lookback   = lookback
        self.threshold  = sweep_threshold
        self._prices    = deque(maxlen=lookback)

    def update(self, price: float) -> float:
        self._prices.append(price)
        if len(self._prices) < 50: return 0.0

        prices_arr = np.array(list(self._prices))
        ref_window = prices_arr[:int(len(prices_arr) * 0.8)]
        recent     = prices_arr[int(len(prices_arr) * 0.8):]

        ref_high = ref_window.max(); ref_low  = ref_window.min()
        ref_rng  = max(ref_high - ref_low, np.std(prices_arr) * 2, 1e-8)

        current = prices_arr[-1]
        rec_max = recent.max(); rec_min = recent.min()

        if rec_max > ref_high and current < ref_high:
            sweep_size = (rec_max - ref_high) / ref_rng
            if sweep_size > self.threshold: return round(min(sweep_size * 5, 1.0), 4)

        if rec_min < ref_low and current > ref_low:
            sweep_size = (ref_low - rec_min) / ref_rng
            if sweep_size > self.threshold: return round(-min(sweep_size * 5, 1.0), 4)

        return 0.0

# --- الإضافات الجديدة للمرحلة الأولى ---

class LiquidityWallsEngine:
    """
    محرك حيطان السيولة: يبحث في دفتر الأوامر عن أقوى مستويات التمركز
    لصناع السوق لتحديد الوقف الديناميكي والأهداف.
    """
    def __init__(self, depth_levels: int = 10, wall_threshold_multiplier: float = 3.0):
        self.depth_levels = depth_levels
        self.multiplier = wall_threshold_multiplier

    def update(self, current_price: float, bids: list, asks: list) -> tuple:
        """
        bids / asks: list of [price, size]
        """
        if not bids or not asks:
            return 0.0, 0.0

        # تحويل لـ Numpy لسرعة الحساب
        bids_arr = np.array(bids[:self.depth_levels])
        asks_arr = np.array(asks[:self.depth_levels])

        if len(bids_arr) == 0 or len(asks_arr) == 0:
            return 0.0, 0.0

        # حساب متوسط حجم الأوامر الطبيعي في الدفتر
        avg_bid_size = np.mean(bids_arr[:, 1])
        avg_ask_size = np.mean(asks_arr[:, 1])

        # البحث عن الحيطة (مستوى فيه فوليوم أعلى من المتوسط بـ X مرات)
        bid_wall_idx = np.argmax(bids_arr[:, 1])
        ask_wall_idx = np.argmax(asks_arr[:, 1])

        best_bid_wall_price = bids_arr[bid_wall_idx, 0]
        best_ask_wall_price = asks_arr[ask_wall_idx, 0]

        # هل هي حيطة حقيقية أم مجرد سيولة عادية؟
        is_bid_wall = bids_arr[bid_wall_idx, 1] > (avg_bid_size * self.multiplier)
        is_ask_wall = asks_arr[ask_wall_idx, 1] > (avg_ask_size * self.multiplier)

        # حساب المسافة من السعر الحالي للحيطة (بالتيك أو النقطة)
        dist_to_bid_wall = round(current_price - best_bid_wall_price, 4) if is_bid_wall else 0.0
        dist_to_ask_wall = round(best_ask_wall_price - current_price, 4) if is_ask_wall else 0.0

        return dist_to_bid_wall, dist_to_ask_wall


class DailyContextEngine:
    """
    محرك سياق اليوم: يراقب اختراق أول ساعة (IB) والوقود اليومي المتبقي.
    """
    def __init__(
        self,
        default_adr: float = 80.0,
        adr_lookback_days: int = 20,
        min_adr_samples: int = 3,
    ):
        self.default_adr = float(default_adr)
        self.adr_lookback_days = max(int(adr_lookback_days), 1)
        self.min_adr_samples = max(int(min_adr_samples), 1)
        self._daily_ranges = deque(maxlen=self.adr_lookback_days)
        self._prev_price = None
        self._pip_size = None

        # التوافق مع المسارات القديمة: training يمرر ADR بوحدة السعر،
        # بينما predict_v19 يمرر قيمة بالـ pips.
        self.adr = float(default_adr)
        self.adr_pips = float(default_adr) if float(default_adr) > 1.0 else 0.0
        self.current_day = None
        self.ib_high = -np.inf
        self.ib_low = np.inf
        self.day_high = -np.inf
        self.day_low = np.inf

    @staticmethod
    def _guess_pip_size(price: float) -> float:
        txt = f"{float(price):.10f}".rstrip('0').rstrip('.')
        decimals = len(txt.split('.')[1]) if '.' in txt else 0
        if decimals >= 4:
            return 0.0001
        if decimals == 3:
            return 0.001
        if decimals == 2:
            return 0.01
        if decimals == 1:
            return 0.1
        return 1.0

    def _update_pip_size(self, price: float) -> float:
        price = float(price)
        if self._prev_price is not None:
            delta = abs(price - self._prev_price)
            if delta > 0:
                self._pip_size = delta if self._pip_size is None else min(self._pip_size, delta)
        self._prev_price = price
        if self._pip_size is None or self._pip_size <= 0:
            self._pip_size = self._guess_pip_size(price)
        return max(float(self._pip_size), 1e-9)

    def _resolve_adr(self, pip_size: float) -> float:
        if len(self._daily_ranges) >= self.min_adr_samples:
            adr_value = float(np.median(self._daily_ranges))
        elif self.default_adr > 1.0:
            adr_value = float(self.default_adr) * pip_size
        else:
            adr_value = float(self.default_adr)

        adr_value = max(float(adr_value), pip_size)
        self.adr = adr_value
        self.adr_pips = round(float(adr_value / max(pip_size, 1e-9)), 1)
        return adr_value

    def update(self, ts: pd.Timestamp, price: float) -> tuple:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is not None:
            ts = ts.tz_convert('UTC').tz_localize(None)

        price = float(price)
        pip_size = self._update_pip_size(price)
        day = ts.date()
        hour = ts.hour

        # تصفير العدادات مع بداية يوم جديد (منتصف الليل)
        if self.current_day != day:
            if self.current_day is not None and np.isfinite(self.day_high) and np.isfinite(self.day_low):
                completed_range = max(float(self.day_high - self.day_low), pip_size)
                self._daily_ranges.append(completed_range)
            self.current_day = day
            self.ib_high = -np.inf
            self.ib_low = np.inf
            self.day_high = price
            self.day_low = price

        # تحديث قمة وقاع اليوم
        if price > self.day_high: self.day_high = price
        if price < self.day_low: self.day_low = price

        # أول ساعة في الجلسة (Initial Balance) - بافتراض الجلسة تبدأ 8 صباحاً مثلاً
        if hour == 8: 
            if price > self.ib_high: self.ib_high = price
            if price < self.ib_low: self.ib_low = price
            ib_status = 0.0 # لسه جوه الرينج
        else:
            # تقييم الكسر
            if price > self.ib_high and self.ib_high != -np.inf:
                ib_status = 1.0 # كسر شرائي
            elif price < self.ib_low and self.ib_low != np.inf:
                ib_status = -1.0 # كسر بيعي
            else:
                ib_status = 0.0 # تذبذب

        # حساب الوقود المتبقي (Remaining Fuel)
        adr_value = self._resolve_adr(pip_size)
        current_range = self.day_high - self.day_low
        remaining_fuel = round(max(0.0, adr_value - current_range), 6)
        fuel_exhausted = 1.0 if remaining_fuel <= (adr_value * 0.1) else 0.0 # لو فاضل أقل من 10% يبقى البنزين خلص

        return ib_status, remaining_fuel, fuel_exhausted, self.adr_pips
