"""
range_state_machine.py — آلة حالة ذكية بدون قواعد ثابتة
══════════════════════════════════════════════════════════════════════════
المشكلة مع القواعد الثابتة (50%، 3 pip، إلخ):
  - رينج 10 pip → 50% = 5 pip  → حساس جداً
  - رينج 50 pip → 50% = 25 pip → بطيء جداً
  نفس القاعدة، سلوك مختلف في كل سوق.

الحل — المايكروستراكتشر يقرر، مش نسبة سعر:
  الانعكاس الحقيقي له بصمة واضحة:
    CVD ينعكس    → الضغط تغيّر
    OBI ينعكس    → الكتاب تغيّر
    Hawkes يرتفع → الزخم الجديد بدأ
    الثلاثة مع بعض = انعكاس حقيقي مش ضوضاء

  الحدود بـ ATR مش نسبة ثابتة:
    الحافة = السعر ± 1.5 × ATR (يتكيف مع التذبذب)
══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from collections import deque
from enum import Enum
from typing import Optional
import numpy as np


class MarketState(Enum):
    RANGING   = "RANGING"
    TRENDING  = "TRENDING"
    UNCERTAIN = "UNCERTAIN"


class SignalState(Enum):
    IDLE         = "IDLE"
    ACCUMULATING = "ACCUMULATING"
    CONFIRMED    = "CONFIRMED"
    LOCKED       = "LOCKED"


# ══════════════════════════════════════════════════════════════════════════
# محرك المايكروستراكتشر
# ══════════════════════════════════════════════════════════════════════════

class MicrostructureReversalEngine:
    """
    يكشف الانعكاس من CVD + OBI + Hawkes.
    درجة الانعكاس = عدد الإشارات المؤكدة (0 - 3).
    """

    def __init__(self, cvd_window=8, obi_window=5, hawkes_window=10,
                 hawkes_mult=1.4, cvd_flip_thr=0.3, obi_flip_thr=0.15,
                 min_signals=2):
        self.cvd_window   = cvd_window
        self.obi_window   = obi_window
        self.hawkes_window= hawkes_window
        self.hawkes_mult  = hawkes_mult
        self.cvd_flip_thr = cvd_flip_thr
        self.obi_flip_thr = obi_flip_thr
        self.min_signals  = min_signals

        self._cvd_hist    = deque(maxlen=cvd_window * 2)
        self._obi_hist    = deque(maxlen=obi_window * 2)
        self._hawkes_hist = deque(maxlen=hawkes_window * 2)
        self._cvd_dir = 0
        self._obi_dir = 0

    def update(self, cvd, obi, hawkes, absorption=0.0):
        self._cvd_hist.append(cvd)
        self._obi_hist.append(obi)
        self._hawkes_hist.append(hawkes)

        result = {'reversal_score': 0.0, 'cvd_flipped': False,
                  'obi_flipped': False, 'hawkes_spike': False, 'direction': 0}

        if len(self._cvd_hist) < self.cvd_window:
            return result

        # CVD
        cvd_arr  = np.array(self._cvd_hist)
        half     = self.cvd_window // 2
        cvd_old  = float(cvd_arr[:half].mean())
        cvd_new  = float(cvd_arr[-half:].mean())
        cvd_rng  = max(abs(cvd_arr).max(), 1e-10)
        if abs((cvd_new - cvd_old) / cvd_rng) > self.cvd_flip_thr and \
           np.sign(cvd_new) != np.sign(cvd_old):
            result['cvd_flipped']    = True
            result['reversal_score'] += 1.0
        self._cvd_dir = int(np.sign(cvd_new))

        # OBI
        obi_arr = np.array(self._obi_hist)
        if len(obi_arr) >= self.obi_window:
            obi_old = float(obi_arr[:self.obi_window//2].mean())
            obi_new = float(obi_arr[-3:].mean())
            if abs(obi_new) > self.obi_flip_thr and \
               np.sign(obi_new) != np.sign(obi_old) and \
               abs(obi_new - obi_old) > self.obi_flip_thr:
                result['obi_flipped']    = True
                result['reversal_score'] += 1.0
            self._obi_dir = int(np.sign(obi_new))

        # Hawkes
        hk = np.array(self._hawkes_hist)
        if len(hk) >= self.hawkes_window:
            hk_mean = float(hk[:-3].mean())
            hk_curr = float(hk[-3:].mean())
            if hk_curr > hk_mean * self.hawkes_mult and hk_curr > hk_mean + 0.05:
                result['hawkes_spike']   = True
                result['reversal_score'] += 1.0

        vote = self._cvd_dir + self._obi_dir
        result['direction'] = int(np.sign(vote)) if vote != 0 else 0
        return result

    def is_reversal(self, result):
        return result['reversal_score'] >= self.min_signals

    def reset(self):
        self._cvd_hist.clear()
        self._obi_hist.clear()
        self._hawkes_hist.clear()
        self._cvd_dir = 0
        self._obi_dir = 0


# ══════════════════════════════════════════════════════════════════════════
# محرك الحدود الديناميكية (ATR)
# ══════════════════════════════════════════════════════════════════════════

class DynamicBoundaryEngine:
    """
    حدود الرينج بـ ATR بدل نسبة ثابتة.
      near_bottom = price <= range_low  + 1.5 × ATR
      near_top    = price >= range_high - 1.5 × ATR
      broke_out   = price > range_high + 2.5 × ATR  (أو العكس)
    """

    def __init__(self, boundary_mult=1.5, breakout_mult=2.5,
                 midzone_mult=0.5, tick_size=0.0001, atr_window=14):
        self.boundary_mult = boundary_mult
        self.breakout_mult = breakout_mult
        self.midzone_mult  = midzone_mult
        self.tick_size     = tick_size
        self._atr_window   = atr_window
        self._price_hist   = deque(maxlen=atr_window * 3)
        self._atr_hist     = deque(maxlen=atr_window)
        self._range_high   = 0.0
        self._range_low    = 0.0

    def update(self, price, micro_atr=0.0):
        self._price_hist.append(price)

        if micro_atr > self.tick_size:
            atr = micro_atr
        elif len(self._price_hist) >= 3:
            prices  = np.array(self._price_hist)
            returns = np.abs(np.diff(prices))
            atr = max(float(np.percentile(returns[-self._atr_window:], 80)),
                      self.tick_size * 2)
        else:
            atr = self.tick_size * 5

        self._atr_hist.append(atr)
        smooth_atr = float(np.median(self._atr_hist)) if self._atr_hist else atr

        if len(self._price_hist) >= 5:
            prices = np.array(self._price_hist)
            self._range_high = float(prices.max())
            self._range_low  = float(prices.min())

        rng_size = max(self._range_high - self._range_low, smooth_atr * 2)
        rng_mid  = (self._range_high + self._range_low) / 2.0
        boundary = smooth_atr * self.boundary_mult
        midzone  = smooth_atr * self.midzone_mult

        range_pos = float(np.clip(
            (price - self._range_low) / max(rng_size, 1e-10), 0.0, 1.0
        ))

        return {
            'atr':         round(smooth_atr, 6),
            'near_bottom': price <= self._range_low  + boundary,
            'near_top':    price >= self._range_high - boundary,
            'in_midzone':  abs(price - rng_mid) <= midzone,
            'broke_out':   (price > self._range_high + smooth_atr * self.breakout_mult or
                            price < self._range_low  - smooth_atr * self.breakout_mult),
            'range_pos':   round(range_pos, 3),
            'range_high':  round(self._range_high, 5),
            'range_low':   round(self._range_low,  5),
            'boundary':    round(boundary, 6),
        }

    def reset(self):
        self._price_hist.clear()
        self._atr_hist.clear()
        self._range_high = 0.0
        self._range_low  = 0.0


# ══════════════════════════════════════════════════════════════════════════
# الآلة الرئيسية
# ══════════════════════════════════════════════════════════════════════════

class RangeStateMachine:
    """
    آلة حالة ذكية — الانعكاس من المايكروستراكتشر والحدود من ATR.
    """

    def __init__(self, min_confirmations=3, confirmation_window=8,
                 min_confidence=0.55, min_candles_between=4,
                 cvd_window=8, obi_window=5, hawkes_window=10,
                 hawkes_mult=1.4, reversal_min_signals=2,
                 boundary_mult=1.5, breakout_mult=2.5,
                 tick_size=0.0001, range_window=20,
                 max_lock_bars=8,
                 trend_early_entry_confidence=0.78,
                 trend_early_entry_range_pos=0.85,
                 trend_early_entry_min_reversal=1.0):

        self.min_confirmations   = min_confirmations
        self.confirmation_window = confirmation_window
        self.min_confidence      = min_confidence
        self.min_candles_between = min_candles_between
        self.max_lock_bars       = max(int(max_lock_bars), 1)
        self.tick_size           = tick_size
        self.trend_early_entry_confidence = trend_early_entry_confidence
        self.trend_early_entry_range_pos  = trend_early_entry_range_pos
        self.trend_early_entry_min_reversal = trend_early_entry_min_reversal

        self.reversal_engine = MicrostructureReversalEngine(
            cvd_window=cvd_window, obi_window=obi_window,
            hawkes_window=hawkes_window, hawkes_mult=hawkes_mult,
            min_signals=reversal_min_signals)

        self.boundary_engine = DynamicBoundaryEngine(
            boundary_mult=boundary_mult, breakout_mult=breakout_mult,
            tick_size=tick_size)

        self.market_state            = MarketState.UNCERTAIN
        self.signal_state            = SignalState.IDLE
        self.candle_idx              = 0
        self._signal_buffer          = deque(maxlen=confirmation_window)
        self._accumulating_direction = None
        self._last_trade_direction   = None
        self._last_trade_candle      = -999
        self._last_trade_price       = 0.0

    def process(self, price, raw_signal, confidence,
                cvd=0.0, obi=0.0, hawkes=0.0, absorption=0.0,
                micro_atr=0.0, regime='Ranging'):
        self.candle_idx += 1

        reversal = self.reversal_engine.update(cvd, obi, hawkes, absorption)
        boundary = self.boundary_engine.update(price, micro_atr)

        self._detect_market_state(boundary, regime)
        decision = self._filter_signal(price, raw_signal, confidence, reversal, boundary)

        decision.update({
            'market_state':   self.market_state.value,
            'signal_state':   self.signal_state.value,
            'confirmations':  self._count_confirmations(),
            'range_pos':      boundary['range_pos'],
            'atr':            boundary['atr'],
            'reversal_score': reversal['reversal_score'],
            'cvd_flipped':    reversal['cvd_flipped'],
            'obi_flipped':    reversal['obi_flipped'],
            'hawkes_spike':   reversal['hawkes_spike'],
        })
        return decision

    def _detect_market_state(self, boundary, regime):
        if regime in ('Trending', 'Volatile'):
            self.market_state = MarketState.TRENDING; return
        if regime in ('Ranging', 'Low_Liquidity'):
            self.market_state = MarketState.RANGING;  return
        if boundary['broke_out'] and self.market_state == MarketState.RANGING:
            self.market_state = MarketState.TRENDING
            self._signal_buffer.clear(); self._accumulating_direction = None; return
        if self.market_state == MarketState.UNCERTAIN:
            self.market_state = MarketState.RANGING

    def _filter_signal(self, price, raw_signal, confidence, reversal, boundary):
        if confidence < self.min_confidence and raw_signal != 'NEUTRAL':
            self._signal_buffer.append('NEUTRAL')
            return self._hold('low_confidence')

        if self.signal_state == SignalState.LOCKED:
            if self.reversal_engine.is_reversal(reversal):
                return {'action': 'CLOSE', 'direction': None,
                        'reason': f'microstructure_reversal_{reversal["reversal_score"]:.1f}'}
            held_bars = self.candle_idx - self._last_trade_candle
            if held_bars >= self.max_lock_bars:
                # Don't keep the state machine frozen for hours; release and
                # let the current bar compete for a fresh entry immediately.
                self.unlock()
            else:
                return self._hold('position_locked')

        if self.candle_idx - self._last_trade_candle < self.min_candles_between:
            self._signal_buffer.append('NEUTRAL')
            return self._hold('cooldown')

        if self.market_state == MarketState.RANGING:
            return self._process_ranging(price, raw_signal, confidence, reversal, boundary)
        return self._process_trending(price, raw_signal, confidence, reversal, boundary)

    def _process_ranging(self, price, raw_signal, confidence, reversal, boundary):
        near_bot = boundary['near_bottom']
        near_top = boundary['near_top']
        midzone  = boundary['in_midzone']

        if midzone and not near_bot and not near_top:
            self._signal_buffer.append('NEUTRAL')
            self._accumulating_direction = None
            return self._hold('range_midzone')

        allowed = 'LONG' if near_bot else ('SHORT' if near_top else None)

        if allowed and raw_signal not in ('NEUTRAL', allowed):
            self._signal_buffer.append('NEUTRAL')
            return self._hold(f'conflicts_pos_{boundary["range_pos"]:.2f}')

        if raw_signal in ('LONG', 'SHORT'):
            eff = raw_signal if (allowed is None or raw_signal == allowed) else 'NEUTRAL'
            self._signal_buffer.append(eff)
            if self._accumulating_direction != eff and eff != 'NEUTRAL':
                self._accumulating_direction = eff
                self.signal_state = SignalState.ACCUMULATING
        else:
            self._signal_buffer.append('NEUTRAL')

        confs = self._count_confirmations()
        if confs >= self.min_confirmations and self._accumulating_direction:
            return self._confirm_entry(
                self._accumulating_direction, price,
                f'range_confirmed_{confs}confs_rev{reversal["reversal_score"]:.1f}')

        self.signal_state = SignalState.ACCUMULATING
        return self._hold(f'accumulating_{confs}/{self.min_confirmations}')

    def _process_trending(self, price, raw_signal, confidence, reversal, boundary):
        if self.reversal_engine.is_reversal(reversal):
            if (self._accumulating_direction == 'LONG' and reversal['direction'] == -1) or \
               (self._accumulating_direction == 'SHORT' and reversal['direction'] == 1):
                self._signal_buffer.clear()
                self._accumulating_direction = None
                self.signal_state = SignalState.IDLE
                return self._hold(f'trend_reversal_{reversal["reversal_score"]:.1f}')

        if raw_signal == 'NEUTRAL':
            self._signal_buffer.append('NEUTRAL')
            return self._hold('trend_no_signal')

        if (raw_signal == self._last_trade_direction and
                abs(price - self._last_trade_price) < boundary['atr'] * 2):
            return self._hold('trend_same_zone')

        self._signal_buffer.append(raw_signal)
        if self._accumulating_direction != raw_signal:
            self._accumulating_direction = raw_signal
            self.signal_state = SignalState.ACCUMULATING

        needed = max(2, self.min_confirmations - 1)
        if self._is_strong_trend_edge_signal(raw_signal, confidence, reversal, boundary):
            needed = 1
        confs  = self._count_confirmations()
        if confs >= needed:
            suffix = '_early' if needed == 1 else ''
            return self._confirm_entry(raw_signal, price, f'trend_confirmed_{confs}confs{suffix}')
        return self._hold(f'trend_accumulating_{confs}/{needed}')

    def _is_strong_trend_edge_signal(self, raw_signal, confidence, reversal, boundary):
        if raw_signal not in ('LONG', 'SHORT'):
            return False
        if confidence < self.trend_early_entry_confidence:
            return False
        if reversal.get('reversal_score', 0.0) < self.trend_early_entry_min_reversal:
            return False
        range_pos = float(boundary.get('range_pos', 0.5))
        edge_pos = float(self.trend_early_entry_range_pos)
        if raw_signal == 'SHORT':
            return range_pos >= edge_pos
        return range_pos <= (1.0 - edge_pos)

    def _count_confirmations(self):
        if not self._signal_buffer or self._accumulating_direction is None:
            return 0
        return list(self._signal_buffer).count(self._accumulating_direction)

    def _confirm_entry(self, direction, price, reason):
        self._last_trade_direction   = direction
        self._last_trade_candle      = self.candle_idx
        self._last_trade_price       = price
        self.signal_state            = SignalState.CONFIRMED
        self._signal_buffer.clear()
        self._accumulating_direction = None
        return {'action': 'ENTER', 'direction': direction, 'reason': reason}

    def _hold(self, reason):
        if self.signal_state == SignalState.CONFIRMED:
            self.signal_state = SignalState.IDLE
        return {'action': 'HOLD', 'direction': None, 'reason': reason}

    def unlock(self):
        self.signal_state = SignalState.IDLE
        self._signal_buffer.clear()
        self._accumulating_direction = None

    def reset(self):
        self.market_state = MarketState.UNCERTAIN
        self.signal_state = SignalState.IDLE
        self.candle_idx   = 0
        self._signal_buffer.clear()
        self._accumulating_direction = None
        self._last_trade_direction   = None
        self._last_trade_candle      = -999
        self._last_trade_price       = 0.0
        self.reversal_engine.reset()
        self.boundary_engine.reset()

    def summary(self):
        return {
            'market_state':      self.market_state.value,
            'signal_state':      self.signal_state.value,
            'candle_idx':        self.candle_idx,
            'accumulating_dir':  self._accumulating_direction,
            'confirmations':     self._count_confirmations(),
            'last_trade':        self._last_trade_direction,
        }


# ══════════════════════════════════════════════════════════════════════════
# تطبيق على DataFrame
# ══════════════════════════════════════════════════════════════════════════

def apply_range_filter_to_dataframe(
    df, signal_col='cb_direction', conf_col='cb_confidence',
    price_col='close', regime_col='regime_name',
    cvd_col='cvd_delta', obi_col='obi',
    hawkes_col='hawkes_intensity', absorb_col='absorption_intensity',
    atr_col='micro_atr', tick_size=0.0001, **rsm_kwargs
):
    rsm = RangeStateMachine(tick_size=tick_size, **rsm_kwargs)
    actions, directions, reasons, states, confs = [], [], [], [], []

    def _g(row, col, d=0.0):
        try: return float(row.get(col, d) or d)
        except: return d

    for _, row in df.iterrows():
        regime = str(
            row.get(
                regime_col,
                row.get("regime_label", row.get("regime", "Ranging")),
            )
        )
        if regime.lstrip('-').isdigit():
            regime = {'0':'Trending','1':'Ranging','2':'Volatile','3':'Low_Liquidity'}.get(regime,'Ranging')

        decision = rsm.process(
            price      = _g(row, price_col),
            raw_signal = str(row.get(signal_col, 'NEUTRAL')),
            confidence = _g(row, conf_col, 0.5),
            cvd        = _g(row, cvd_col),
            obi        = _g(row, obi_col),
            hawkes     = _g(row, hawkes_col),
            absorption = _g(row, absorb_col),
            micro_atr  = _g(row, atr_col),
            regime     = regime,
        )

        actions.append(decision['action'])
        directions.append(decision.get('direction') or 'NEUTRAL')
        reasons.append(decision['reason'])
        states.append(decision['market_state'])
        confs.append(decision['confirmations'])

        if decision['action'] == 'ENTER':
            rsm.signal_state = SignalState.LOCKED
        elif decision['action'] == 'CLOSE':
            rsm.unlock()

    out = df.copy()
    out['rsm_action']    = actions
    out['rsm_direction'] = directions
    out['rsm_reason']    = reasons
    out['rsm_state']     = states
    out['rsm_confs']     = confs
    return out
