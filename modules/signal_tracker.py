"""
signal_tracker.py — V3: Unified Pipeline Tracker
═══════════════════════════════════════════════════════════════════════════
التحديثات (V3):
  - يستخدم الآن OrderWallScanner مباشرة لكل tick
  - يمرر scan_result الكامل لـ DynamicTargetManager
  - يحفظ context_features مع كل صفقة للتحليل اللاحق
  - يدعم regime_tradeable من خارج النظام
═══════════════════════════════════════════════════════════════════════════
"""

import logging
import datetime
import os
from typing import Optional, List, Dict, Any

import numpy as np

try:
    from modules.dynamic_target import DynamicTargetManager, TradeState
    from modules.dynamic_labels  import OrderWallScanner, compute_dynamic_levels
except ImportError:
    from dynamic_target import DynamicTargetManager, TradeState
    from dynamic_labels  import OrderWallScanner, compute_dynamic_levels


class QuantSignalTracker:
    """
    يتتبع إشارات التداول ويحاكي التنفيذ الحي.

    V3 — Pipeline موحد:
      - يستخدم OrderWallScanner لكل tick → scan_result كامل
      - يمرر scan_result لـ DynamicTargetManager (بدل أرقام مجردة)
      - يحفظ context_features مع كل صفقة للتحليل اللاحق
    """

    def __init__(self,
                 log_filename: str   = 'outputs/live_trades.log',
                 tick_size:    float = 0.0001,
                 wall_mult:    float = 3.0,
                 gap_mult:     float = 2.0):

        self.tick_size      = tick_size
        self.total_closed   = 0
        self.successful     = 0
        self.active_signals: Dict[int, dict] = {}
        self.signal_counter = 0

        # OrderWallScanner مشترك بين كل الصفقات
        self.scanner = OrderWallScanner(wall_mult=wall_mult, gap_mult=gap_mult)

        # ── Logging ──────────────────────────────────────────────────────
        os.makedirs(os.path.dirname(log_filename) or '.', exist_ok=True)
        self.logger = logging.getLogger('QuantTracker')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.handlers:
            fh = logging.FileHandler(log_filename)
            fh.setFormatter(logging.Formatter(
                '[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter(
                '[%(asctime)s] %(message)s', '%H:%M:%S'))
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    # ══════════════════════════════════════════════════════════════════════
    # فتح صفقة جديدة
    # ══════════════════════════════════════════════════════════════════════

    def evaluate_and_open_virtual_trade(self,
                                         signal_data:      dict,
                                         order_book_row:   dict,
                                         context:          dict,
                                         timestamp=None) -> bool:
        """
        يفتح صفقة جديدة.

        Parameters
        ----------
        signal_data     : {'bias': 'LONG'/'SHORT', 'price': float,
                           'cvd_delta': float, 'confidence': float,
                           'tradeable': bool}
        order_book_row  : صف MBP10 الحالي (bid_px_00 ... ask_sz_09)
        context         : {'remaining_fuel': float, 'tick_size': float, ...}
        """
        bias = signal_data.get('bias', 'NEUTRAL')
        prob = signal_data.get('confidence', 0.0)

        if not (bias in ('LONG', 'SHORT') and signal_data.get('tradeable', False)):
            return False

        self.signal_counter += 1
        price = signal_data.get('price', 0.0)

        # ── مسح دفتر الأوامر ─────────────────────────────────────────────
        scan_result = self.scanner.scan(order_book_row, self.tick_size)
        levels      = compute_dynamic_levels(scan_result, price, self.tick_size)

        # ── FIX: سحب remaining_fuel و adr_pips الحقيقيين من signal_data ──
        # predict_step يحسبهم من DailyContextEngine ويحطهم في النتيجة
        # لو مش موجودين → نستخدم قيم من context أو defaults معقولة
        real_fuel = (
            signal_data.get('remaining_fuel') or
            context.get('remaining_fuel') or
            0.0060   # 60pip fallback
        )
        real_adr  = (
            signal_data.get('adr_pips') or
            context.get('adr_pips') or
            80.0
        )

        # ── فتح الصفقة عبر DynamicTargetManager ─────────────────────────
        mgr = DynamicTargetManager()
        ctx = {
            **context,
            'tick_size':      self.tick_size,
            'remaining_fuel': float(real_fuel),
            'adr_pips':       float(real_adr),
        }
        trade = mgr.open_trade(signal_data, levels, scan_result, ctx, 4)

        trade_record = {
            'trade_id':        self.signal_counter,
            'entry_price':     price,
            'direction':       bias,
            'entry_time':      str(timestamp or datetime.datetime.now()),
            'exit_time':       None,
            'pnl_pips':        0.0,
            'max_favorable':   0.0,
            'max_adverse':     0.0,
            'highest':         price,
            'lowest':          price,
            'status':          'OPEN',
            'probability':     round(float(prob), 4),
            'manager':         mgr,
            'trade_obj':       trade,
            # حفظ scan snapshot عند الدخول
            'entry_scan':      {k: v for k, v in scan_result.items()
                                if not isinstance(v, type(None))},
            'entry_levels':    levels,
            'context_features_history': [],
        }
        self.active_signals[self.signal_counter] = trade_record

        self.logger.info(
            f"{'='*60}\n"
            f"🟢 [ENTRY #{self.signal_counter}] {bias} @ {price:.5f} "
            f"| SL:{trade.sl:.5f} TP1:{trade.tp1:.5f} "
            f"| RR≈{levels['long_rr'] if bias=='LONG' else levels['short_rr']:.2f} "
            f"| WallStr={'%.2f'%scan_result.get('bid_wall_strength' if bias=='LONG' else 'ask_wall_strength', 0)} "
            f"| prob={prob:.1%}\n"
            f"{'='*60}"
        )
        return True

    # ══════════════════════════════════════════════════════════════════════
    # تحديث الصفقات المفتوحة
    # ══════════════════════════════════════════════════════════════════════

    def update_and_get_closed_trades(self,
                                      current_price:    float,
                                      current_cvd:      float,
                                      order_book_row:   dict,
                                      vwap_zscore:      float,
                                      regime_tradeable: bool = True,
                                      timestamp=None) -> List[dict]:
        """
        يُستدعى كل tick لتحديث الصفقات المفتوحة.

        Parameters
        ----------
        current_price   : آخر سعر
        current_cvd     : CVD الحالي
        order_book_row  : صف MBP10 الحالي (لحساب حجم الحائط الحالي)
        vwap_zscore     : قراءة رادار الانعكاس
        regime_tradeable: هل السوق قابل للتداول؟
        """
        closed = []
        now    = str(timestamp or datetime.datetime.now())

        # مسح دفتر الأوامر مرة واحدة لكل الصفقات
        current_scan = self.scanner.scan(order_book_row, self.tick_size)

        # حجم الحائط الحالي (يختلف حسب اتجاه الصفقة)
        bid_wall_now = current_scan.get('bid_wall_size_raw', 0.0)
        ask_wall_now = current_scan.get('ask_wall_size_raw', 0.0)

        for sig_id in list(self.active_signals.keys()):
            rec   = self.active_signals[sig_id]
            entry = rec['entry_price']
            mgr   = rec['manager']
            dirn  = rec['direction']

            # MFE / MAE
            rec['highest'] = max(rec['highest'], current_price)
            rec['lowest']  = min(rec['lowest'],  current_price)
            if dirn == 'LONG':
                rec['max_favorable'] = (rec['highest'] - entry) / self.tick_size
                rec['max_adverse']   = (entry - rec['lowest'])  / self.tick_size
            else:
                rec['max_favorable'] = (entry - rec['lowest'])  / self.tick_size
                rec['max_adverse']   = (rec['highest'] - entry) / self.tick_size

            # حجم الحائط المناسب للاتجاه
            wall_now = bid_wall_now if dirn == 'LONG' else ask_wall_now

            # تحديث المحرك
            result = mgr.update(
                current_price     = current_price,
                current_cvd_delta = current_cvd,
                current_wall_size = wall_now,
                vwap_zscore       = vwap_zscore,
                regime_tradeable  = regime_tradeable,
                tick_size         = self.tick_size,
            )

            action = result.get('action', 'none')
            ctx_f  = result.get('context_features', {})

            # حفظ context_features للتحليل
            if ctx_f:
                rec['context_features_history'].append(ctx_f)

            # ── إغلاق جزئي ───────────────────────────────────────────────
            if 'partial' in action:
                self.logger.info(
                    f"⚡ [PARTIAL #{sig_id}] {dirn} | {action} "
                    f"| Pips Locked: +{result.get('pips_locked', 0):.1f} "
                    f"| Next: {result.get('next_target', 'N/A')} "
                    f"| WallRate: {ctx_f.get('wall_consumption_rate', 0):.1%}"
                )

            # ── إغلاق كلي ─────────────────────────────────────────────────
            elif 'close_all' in action:
                t_obj = result.get('trade')
                rec['status']    = 'CLOSED'
                rec['exit_time'] = now
                rec['pnl_pips']  = t_obj.realized_pips if t_obj else result.get('pips', 0)

                is_win = rec['pnl_pips'] > 0
                self.total_closed += 1
                if is_win:
                    self.successful += 1

                win_rate = self.successful / max(1, self.total_closed) * 100
                label    = "✅ WIN" if is_win else "❌ LOSS"
                if 'emergency' in action:
                    label = "🚨 EMERGENCY"

                self.logger.info(
                    f"\n{label} #{sig_id} {dirn}"
                    f" | سبب: {result.get('reason', action)}"
                    f" | P&L: {rec['pnl_pips']:+.1f} Pips"
                    f" | WinRate: {win_rate:.1f}%"
                    f" | MFE: {rec['max_favorable']:.1f}"
                    f" | MAE: {rec['max_adverse']:.1f}"
                    f" | WallRate: {ctx_f.get('wall_consumption_rate', 0):.1%}"
                )

                # حفظ ملخص context_features النهائي
                rec['final_context'] = ctx_f
                rec['avg_wall_consumption'] = (
                    float(np.mean([
                        cf.get('wall_consumption_rate', 0)
                        for cf in rec['context_features_history']
                    ])) if rec['context_features_history'] else 0.0
                )

                # تنظيف الكائنات الثقيلة
                rec.pop('manager',   None)
                rec.pop('trade_obj', None)

                closed.append(rec)
                del self.active_signals[sig_id]

        return closed

    # ══════════════════════════════════════════════════════════════════════
    # خصائص ومساعدات
    # ══════════════════════════════════════════════════════════════════════

    @property
    def open_count(self) -> int:
        return len(self.active_signals)

    @property
    def win_rate(self) -> float:
        return self.successful / max(1, self.total_closed)

    def summary(self) -> dict:
        return {
            'total_closed': self.total_closed,
            'successful':   self.successful,
            'win_rate':     round(self.win_rate, 4),
            'open_trades':  self.open_count,
        }


# ══════════════════════════════════════════════════════════════════════════
# مثال استخدام
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import numpy as np

    tracker = QuantSignalTracker(tick_size=0.0001)

    # صف MBP10 وهمي
    fake_book = {f'bid_px_{i:02d}': 1.2000 - i * 0.0001 for i in range(10)}
    fake_book.update({f'bid_sz_{i:02d}': 100 + i * 10 for i in range(10)})
    fake_book.update({f'ask_px_{i:02d}': 1.2001 + i * 0.0001 for i in range(10)})
    fake_book.update({f'ask_sz_{i:02d}': 90 + i * 5 for i in range(10)})
    # إضافة حائط كبير على bid_sz_02
    fake_book['bid_sz_02'] = 5000

    signal = {
        'bias': 'LONG', 'price': 1.2001,
        'cvd_delta': 150.0, 'confidence': 0.72,
        'tradeable': True,
    }
    context = {
        'remaining_fuel': 0.0060,   # FIX: 60pip بدل 8pip الهاردكود القديم
        'adr_pips':       80.0,     # FIX: يُحدَّث من DailyContextEngine في الإنتاج
        'tick_size':      0.0001,
    }

    opened = tracker.evaluate_and_open_virtual_trade(signal, fake_book, context)
    print(f"Opened: {opened}")

    # محاكاة 10 ticks
    for step, price in enumerate(np.linspace(1.2001, 1.2025, 10)):
        closed = tracker.update_and_get_closed_trades(
            current_price   = float(price),
            current_cvd     = 150.0 + step * 10,
            order_book_row  = fake_book,
            vwap_zscore     = 0.5,
            regime_tradeable= True,
        )
        if closed:
            print(f"Closed: {[c['pnl_pips'] for c in closed]}")

    print(tracker.summary())
