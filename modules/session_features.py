import pandas as pd
import numpy as np
from collections import deque

# ── Session windows (UTC) ─────────────────────────────────────────
SESSION_WINDOWS = {
    'asia':    (0,   8),   # 00:00 → 08:00 UTC
    'london':  (7,  16),   # 07:00 → 16:00 UTC
    'ny':      (13, 22),   # 13:00 → 22:00 UTC
    'overlap': (13, 16),   # 13:00 → 16:00 UTC (peak)
    'off':     (22, 24),   # 22:00 → 00:00 UTC
}

SESSION_LABEL = {
    'overlap': 3,
    'ny':      2,
    'london':  1,
    'asia':    0,
    'off':     4,
}


def _coerce_session_timestamp_series(values, *, context: str) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    ts = pd.to_datetime(series, utc=True, errors='coerce')
    if not isinstance(ts, pd.Series):
        ts = pd.Series(ts, index=series.index)
    ts = ts.dt.tz_localize(None)
    invalid_mask = ts.isna()
    if bool(invalid_mask.any()):
        sample = invalid_mask[invalid_mask].index[:5].tolist()
        raise ValueError(
            f"❌ Invalid session timestamps in {context}: "
            f"rows={int(invalid_mask.sum())} sample_indices={sample}"
        )
    return ts


def _coerce_session_timestamp_scalar(value, *, context: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors='coerce')
    if pd.isna(ts):
        raise ValueError(f"❌ Invalid session timestamp in {context}: value={value!r}")
    return ts.tz_convert(None) if getattr(ts, 'tzinfo', None) is not None else pd.Timestamp(ts)


def add_session_features(df: pd.DataFrame, ts_col: str = None) -> pd.DataFrame:
    """
    يضيف session zone features للـ DataFrame بأمان
    تم تصحيح مشكلة اختفاء عمود الوقت والـ TZ stripping
    """
    df = df.copy()

    # 1. البحث الذكي عن عمود الوقت (للتوافق مع MBO أو Volume Bars)
    if ts_col is None or ts_col not in df.columns:
        possible_ts_cols = ['ts_event', 'ts_open', 'ts_close', 'timestamp', 'date', 'time']
        found_col = next((c for c in possible_ts_cols if c in df.columns), None)
        
        if found_col is None:
            # لو فشلنا تماماً، نستخدم الـ Index لو كان DatetimeIndex
            if isinstance(df.index, pd.DatetimeIndex):
                ts_series = df.index.to_series()
                print("  ⚠️ Session: استخدام الـ Index كعمود وقت.")
            else:
                raise ValueError("❌ Session features require a valid timestamp column or DatetimeIndex")
        else:
            ts_series = df[found_col]
            print(f"  🕒 Session: استخدام عمود '{found_col}' لحساب الجلسات.")
    else:
        ts_series = df[ts_col]

    # 2. معالجة الـ Timezone بأمان تام
    ts = _coerce_session_timestamp_series(ts_series, context=f'session_features.{ts_col or "auto"}')
    hour = ts.dt.hour.fillna(-1).astype(np.int8)

    # 3. حساب الفيتشرز
    df['session_hour']    = hour
    df['session_asia']    = ((hour >= 0)  & (hour < 8)).astype(np.int8)
    df['session_london']  = ((hour >= 7)  & (hour < 16)).astype(np.int8)
    df['session_ny']      = ((hour >= 13) & (hour < 22)).astype(np.int8)
    df['session_overlap'] = ((hour >= 13) & (hour < 16)).astype(np.int8)
    
    # الجلسة المغلقة (Off) هي أي وقت لا يقع في الـ 3 جلسات الرئيسية (وأن لا يكون الوقت فاسد/NaT)
    df['session_off']     = ((hour >= 22) | ((hour >= 0) & (df['session_asia'] == 0) & (df['session_london'] == 0) & (df['session_ny'] == 0))).astype(np.int8)
    
    # حماية من القيم الفاسدة (NaT)
    invalid_mask = (hour == -1)
    for col in ['session_asia', 'session_london', 'session_ny', 'session_overlap', 'session_off']:
        df.loc[invalid_mask, col] = 0

    # 4. الـ Labels بناءً على الأولوية
    label = np.full(len(df), SESSION_LABEL['off'], dtype=np.int8)
    
    # الأولوية الأدنى للأعلى (يتم التغطية عليها)
    label[df['session_asia']    == 1] = SESSION_LABEL['asia']
    label[df['session_london']  == 1] = SESSION_LABEL['london']
    label[df['session_ny']      == 1] = SESSION_LABEL['ny']
    label[df['session_overlap'] == 1] = SESSION_LABEL['overlap']
    
    # إعادة تعيين الـ Labels للبيانات الفاسدة لتكون Off
    label[invalid_mask] = SESSION_LABEL['off']
    
    df['session_label'] = label

    # 5. طباعة الإحصائيات للتأكد من نجاح العملية
    counts = {
        'Asia':    int(df['session_asia'].sum()),
        'London':  int(df['session_london'].sum()),
        'NY':      int(df['session_ny'].sum()),
        'Overlap': int(df['session_overlap'].sum()),
        'Off':     int(df['session_off'].sum()),
    }
    
    total = len(df)
    parts = [f"{s}={n:,}({n/total:.0%})" for s,n in counts.items() if n > 0]
    
    if sum(counts.values()) == 0:
        print("  ❌ Sessions: فشل في استخراج الجلسات، كل القيم صفر!")
    else:
        print(f"  ✅ Sessions: {' | '.join(parts)}")

    return df

SESSION_FEATURE_COLS = [
    'session_asia',
    'session_london',
    'session_ny',
    'session_overlap',
    'session_off',
    'session_hour',
    'session_label',
]

# --- الإضافات الجديدة للمرحلة الأولى ---

class SessionVWAPEngine:
    """
    محرك الجلسة: يحسب الـ VWAP التراكمي، الـ Z-Score (انحراف السعر)، 
    والـ Session CVD (بوصلة السيولة) من بداية اليوم.
    """
    def __init__(self, z_score_window: int = 500):
        self.current_day = None
        self.cum_volume = 0.0
        self.cum_pv = 0.0
        self.session_cvd = 0.0
        
        self.z_score_window = z_score_window
        self._price_history = deque(maxlen=z_score_window)
        self._vwap_history = deque(maxlen=z_score_window)

    def update(self, ts, price: float, volume: float, is_buy_aggressor: bool = True) -> tuple:
        # تحويل ts لـ Timestamp بأمان
        # ② FIX: معالجة شاملة لكل أنواع ts بما فيها numpy.datetime64
        if isinstance(ts, (int, float, np.integer)):
            ts = _coerce_session_timestamp_scalar(int(ts), context='SessionVWAPEngine.update')
        elif isinstance(ts, np.datetime64):
            ts = _coerce_session_timestamp_scalar(ts, context='SessionVWAPEngine.update')
        elif isinstance(ts, str):
            ts = _coerce_session_timestamp_scalar(ts, context='SessionVWAPEngine.update')
        elif not isinstance(ts, pd.Timestamp):
            ts = _coerce_session_timestamp_scalar(ts, context='SessionVWAPEngine.update')
        elif ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        day = ts.date()

        # تصفير مع بداية يوم تداول جديد
        if self.current_day != day:
            self.current_day = day
            self.cum_volume = 0.0
            self.cum_pv = 0.0
            self.session_cvd = 0.0
            self._price_history.clear()
            self._vwap_history.clear()

        # 1. تحديث الـ Session CVD
        if is_buy_aggressor:
            self.session_cvd += volume
        else:
            self.session_cvd -= volume

        # 2. تحديث الـ VWAP
        self.cum_volume += volume
        self.cum_pv += (price * volume)
        
        current_vwap = price # افتراضي لو مفيش فوليوم
        if self.cum_volume > 0:
            current_vwap = self.cum_pv / self.cum_volume

        # 3. حساب نظرية "الأستك المشدود" (VWAP Z-Score)
        self._price_history.append(price)
        self._vwap_history.append(current_vwap)
        
        vwap_z_score = 0.0
        if len(self._price_history) > 50:
            prices_arr = np.array(self._price_history)
            vwaps_arr = np.array(self._vwap_history)
            
            # حساب الانحراف المعياري للسعر حول الـ VWAP
            deviations = prices_arr - vwaps_arr
            std_dev = np.std(deviations) + 1e-8 # حماية من القسمة على صفر
            
            # كم انحراف معياري يبعد السعر الآن عن الـ VWAP؟
            current_deviation = price - current_vwap
            vwap_z_score = round(current_deviation / std_dev, 4)

        # حساب ميل خط الـ VWAP (Slope) لمعرفة الترند
        vwap_slope = 0.0
        if len(self._vwap_history) >= 20:
            vwap_slope = round(self._vwap_history[-1] - self._vwap_history[-20], 6)

        return current_vwap, vwap_z_score, vwap_slope, self.session_cvd

# ═══════════════════════════════════════════════════════════════════
# التعديل 1 — Cyclical Session Encoding (لا leakage)
# ═══════════════════════════════════════════════════════════════════
CYCLICAL_SESSION_COLS = [
    'hour_sin', 'hour_cos',
    'minute_sin', 'minute_cos',
    'london_active', 'ny_active', 'overlap_active',
]

def add_cyclical_session_features(df: pd.DataFrame, ts_col: str = 'ts_event') -> pd.DataFrame:
    """
    يحوّل الوقت لتمثيل دائري (Cyclical Encoding) بدلاً من session flags.

    لماذا لا يسبب leakage؟
      - sin/cos للساعة تعتمد فقط على الوقت الحالي (timestamp)
      - لا تعكس أي معلومة مستقبلية
      - london_active/ny_active تعتمد فقط على الساعة الحالية

    لماذا أفضل من session flags؟
      - 08:59 و 09:01 متشابهان في الحساب (continuity)
      - النموذج يتعلم العلاقة بين الوقت والسلوك بشكل تدريجي
    """
    df = df.copy()

    if ts_col not in df.columns:
        raise ValueError(f"❌ Cyclical session features require '{ts_col}' in the dataframe")
    ts = _coerce_session_timestamp_series(
        df[ts_col],
        context=f'cyclical_session_features.{ts_col}',
    )

    hour   = ts.dt.hour.fillna(0).astype(float)
    minute = ts.dt.minute.fillna(0).astype(float)

    # Cyclical: ساعة الكرة الأرضية
    df['hour_sin']   = np.sin(2 * np.pi * hour   / 24.0).astype(np.float32)
    df['hour_cos']   = np.cos(2 * np.pi * hour   / 24.0).astype(np.float32)
    df['minute_sin'] = np.sin(2 * np.pi * minute / 60.0).astype(np.float32)
    df['minute_cos'] = np.cos(2 * np.pi * minute / 60.0).astype(np.float32)

    # Binary session flags (causal — بناءً على الوقت الحالي فقط)
    df['london_active']  = ((hour >= 7)  & (hour < 16)).astype(np.float32)
    df['ny_active']      = ((hour >= 13) & (hour < 22)).astype(np.float32)
    df['overlap_active'] = ((hour >= 13) & (hour < 16)).astype(np.float32)

    return df
