"""
prepare_training_data.py — V19 Data Refinery
═══════════════════════════════════════════════════════════════════════
يبني dataset موحد لـ QuantSystem V19:
  ✅ استخراج features من MBO و MBP10
  ✅ إضافة rolling/session/context features
  ✅ توليد autoencoder embeddings
  ✅ بناء LOB tensors للـ DeepLOB branch
  ✅ توليد causal labels الخاصة بـ V19
═══════════════════════════════════════════════════════════════════════
"""

# QuantSystem V19
import argparse, datetime, os, sys, multiprocessing, json
import numpy as np
import pandas as pd
import gc

from modules.microstructure          import (FastMicrostructureEngine,
                                              AbsorptionIntensityEngine, CancelRatioEngine,
                                              FastTapeSpeedTracker, TRADE_ACTIONS)
from modules.fisher_alpha            import FastFisherAlpha
from modules.fim_anomaly             import FastFIMDetector
from modules.orderbook               import (OrderBookSnapshotEngine,
                                              SpoofingDetector, LiquidityTrapDetector)
from modules.micro_volatility        import MicroVolatilityEngine
from modules.context_features        import (MomentumContextEngine,
                                              LiquiditySweepDetector,
                                              compute_daily_weekly_levels,
                                              LiquidityWallsEngine, DailyContextEngine) # 🔴 إضافة كلاسات السياق
from modules.auto_calibrator         import AutoCalibrator
from modules.market_research_features import (KylesLambdaEngine,
                                               HawkesIntensityEngine,
                                               LiquidityGapsEngine,
                                               VNETEngine)
from modules.fractional_diff         import apply_fractional_diff
from modules.dynamic_labels         import (DIR_NEUTRAL,
                                              OrderWallScanner,
                                              QUALITY_STRONG,
                                              QUALITY_WEAK)
from modules.purging_embargo         import (spearman_redundancy_filter,
                                              mrmr_selection,
                                              walk_forward_expanding)
from modules.regime_classifier       import RegimeClassifier, REGIME_META_SCORE_COLS
from modules.slippage_model          import SlippageModel
from modules.session_features        import add_session_features, SESSION_FEATURE_COLS, SessionVWAPEngine # 🔴 إضافة VWAP
from modules.gpu_config              import (detect_gpu, get_multiprocessing_workers,
                                              print_gpu_report, N_WORKERS)
# ── V19: DeepLOB Tensor Builder ──────────────────────────────────
DEEPLOB_AVAILABLE = False
LOBTensorBuilder = None
build_lob_tensor_dataset = None
N_TIME_STEPS = 0
N_PRICE_LEVELS = 0
N_CHANNELS = 0
_DEEPLOB_IMPORT_ATTEMPTED = False

DEEPLOB_MAX_EVENTS_DEFAULT = 5_000_000
DEEPLOB_MAX_TENSORS_DEFAULT = 25_000
DEEPLOB_MAX_GB_DEFAULT = 0.30
LOB_EVENT_SAMPLE_DEFAULT = 100_000

try:
    from modules.labels_v22 import build_causal_event_labels
    V19_LABELS_AVAILABLE = True
    V19_LABELS_IMPORT_ERROR = None
    V19_LABELS_SOURCE = 'modules.labels_v22'
except ImportError:
    try:
        from modules.labels_v19 import build_causal_event_labels
        V19_LABELS_AVAILABLE = True
        V19_LABELS_IMPORT_ERROR = None
        V19_LABELS_SOURCE = 'modules.labels_v19'
    except ImportError as exc:
        V19_LABELS_AVAILABLE = False
        V19_LABELS_IMPORT_ERROR = exc
        V19_LABELS_SOURCE = None
        build_causal_event_labels = None

try:
    from tqdm import tqdm
    def _prog(it, desc='', total=None, **kw): return tqdm(it, desc=desc, total=total)
except ImportError:
    def _prog(it, desc='', total=None, **kw):
        it=list(it); n=len(it)
        for i,x in enumerate(it):
            if (i+1)%200_000==0: print(f"  {desc}: {i+1:,}/{n:,}", flush=True)
            yield x

# 🔴 تحديث قائمة الميزات لتشمل حواس الماكرو الجديدة والـ Embeddings
EMBEDDINGS_DIM = 8
EMBEDDING_COLS = [f'emb_{i}' for i in range(EMBEDDINGS_DIM)]

# ══════════════════════════════════════════════════════════
# ANTI-LEAKAGE DESIGN:
#
# SESSION_LEAK_COLS: أعمدة تعكس وقت الجلسة مباشرة
#   → ارتباط مباشر مع session-based labels → DATA LEAKAGE
#   → تُحفظ في CSV كـ metadata لكن لا تدخل الموديل أبداً
#
# VWAP_REALTIME_COLS: مقبولة لأنها real-time tick-by-tick
#   → لا تعكس نهاية الجلسة → لا تسريب
# ══════════════════════════════════════════════════════════
SESSION_LEAK_COLS = [
    'session_asia', 'session_london', 'session_ny',
    'session_overlap', 'session_off', 'session_hour', 'session_label',
    'session_cvd',   # CVD تراكمي يعكس اتجاه الجلسة = LEAK مع session labels
]

# ── Anti-Leakage Diagnostic Drop ─────────────────────────────────
# هذه الأعمدة قد تظهر في CSV كـ metadata لكن يجب حذفها من الذاكرة
# قبل أي تدريب. تُستخدم في مسار التدريب الحديث لتصفية metadata غير المناسبة للموديل.
# الفلسفة: الـ CSV يحفظ كل شيء، الـ training يختار ما يحتاجه فقط.
TEMPORAL_DROP_COLS = SESSION_LEAK_COLS + [
    'hour', 'minute', 'day_of_week',       # أعمدة زمنية خام
    'session', 'date',                      # metadata زمنية
    # NOTE: bias_label/setup_label/conf_label محمية في _sanitize_df
    # لا تُحذف من هنا لمنع KeyError في stage1_catboost_advisor
    'is_expansion', 'liq_score',
]

FEATURE_COLS = [
    # ── Core Statistical Surface + Dynamic Order-Book Context ───────────
    'cvd', 'obi', 'absorption_intensity', 'cancel_ratio',
    'spoofing_ratio', 'spoofing_duration', 'liquidity_trap',
    'micro_atr', 'volume_burst', 'inter_event_time',
    'micro_price', 'bid_wall_strength', 'ask_wall_strength',
    'distance_to_wall', 'gap_size', 'liquidity_density',
    'fisher_signal', 'anomaly',
    'cvd_momentum', 'cvd_price_divergence',
    'trend_strength', 'correction_depth', 'liquidity_sweep',
    'pdh', 'pdl', 'pwh', 'pwl',
    'dist_to_pdh', 'price_position',
]

MODEL_FEATURE_COLS = FEATURE_COLS + [
    # ── Extended Model Inputs ───────────────────────────
    'kyle_lambda', 'hawkes_intensity',
    'liquidity_gaps', 'vnet',
    'cvd_roc_10', 'cvd_roc_50', 'cvd_roc_200',
    'cvd_accel', 'volume_accel',
    'cvd_frac', 'kyle_frac', 'hawkes_frac', 'vnet_frac',
    'dl_anomaly_score',
    'dist_to_bid_wall', 'dist_to_ask_wall',
    'ib_status', 'remaining_fuel', 'fuel_exhausted',
    'current_vwap', 'vwap_z_score', 'vwap_slope',
] + EMBEDDING_COLS

BINARY_FEATURES = {'fisher_signal', 'anomaly', 'fuel_exhausted'}

PROTECTED_FEATURES = {
    # Microstructure core — لا تُحذف من mRMR أبداً
    'cvd', 'absorption_intensity', 'kyle_lambda', 'hawkes_intensity',
    'pdh', 'pdl', 'pwh', 'pwl', 'obi', 'spoofing_ratio', 'liquidity_trap',
    'micro_price', 'bid_wall_strength', 'ask_wall_strength',
    'distance_to_wall', 'gap_size', 'liquidity_density',
    # VWAP real-time — safe, محمية
    'vwap_z_score', 'vwap_slope', 'current_vwap',
    'dist_to_bid_wall', 'dist_to_ask_wall',
    'ib_status', 'remaining_fuel',
    # NOTE: session_overlap/london/ny أُزيلت من FEATURE_COLS تماماً (anti-leakage)
}

SESSIONS = [
    ('asia',      0,  0,  8,  0),
    ('london',    8,  0, 13,  0),
    ('ny_open',  13,  0, 16,  0),
    ('ny_main',  16,  0, 21,  0),
]

# ── V19: CatBoost Advisor feature surface (extended with dynamic labels context) ──
# هذه هي المدخلات للمرحلة الأولى (CatBoost Statistical Advisor)
CATBOOST_ADVISOR_FEATURES = [
    'cvd', 'obi', 'absorption_intensity', 'cancel_ratio',
    'spoofing_ratio', 'spoofing_duration', 'liquidity_trap',
    'micro_atr', 'volume_burst', 'inter_event_time',
    'micro_price', 'bid_wall_strength', 'ask_wall_strength',
    'distance_to_wall', 'gap_size', 'liquidity_density',
    'fisher_signal', 'anomaly',
    'cvd_momentum', 'cvd_price_divergence',
    'trend_strength', 'correction_depth', 'liquidity_sweep',
    'pdh', 'pdl', 'dist_to_pdh', 'price_position',
    'kyle_lambda', 'hawkes_intensity', 'vnet',
    'vwap_z_score',
]  # N = 31

RAW_STAT_PREFIX = 'raw__'
RAW_STAT_FEATURE_COLS = [f'{RAW_STAT_PREFIX}{col}' for col in CATBOOST_ADVISOR_FEATURES]

# V19 Meta-Features — مخرجات CatBoost تُضاف للـ LSTM
META_FEATURE_COLS = [
    'cb_prob_long', 'cb_prob_short',                       # 2 احتمالات CatBoost
    'cluster_0', 'cluster_1', 'cluster_2', 'cluster_3',   # 4 One-Hot Cluster
    *list(REGIME_META_SCORE_COLS),                        # 3 Soft regime scores
]  # N = 9

# V19 Visual Features — مخرجات CNN
VISUAL_EMB_COLS = [f'vis_emb_{i}' for i in range(8)]  # N = 8

BIAS_LONG    = 0
BIAS_SHORT   = 1
BIAS_NEUTRAL = 2

SETUP_ABSORPTION = 0
SETUP_SPOOFING   = 1
SETUP_OBI        = 2
SETUP_MIXED      = 3

def _normalize_databento_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'ts_event' not in df.columns and 'ts_recv' in df.columns:
        df['ts_event'] = df['ts_recv']

    if 'ts_event' in df.columns:
        df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True, errors='coerce').dt.tz_localize(None)
        df = df[df['ts_event'].notna()].reset_index(drop=True)

    if 'action' in df.columns:
        df['action'] = df['action'].astype(str).str.strip().str.upper()
        df = df[df['action'] != 'R'].reset_index(drop=True)
    else:
        df['action'] = 'A'

    if 'side' in df.columns:
        df['side'] = df['side'].astype(str).str.strip().str.upper().replace('N', 'A')
    else:
        df['side'] = 'B'

    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df[df['price'].notna() & (df['price'] > 0)].reset_index(drop=True)
        df['price'] = df['price'].astype('float32')
    else:
        bid0 = pd.to_numeric(df['bid_px_00'], errors='coerce') if 'bid_px_00' in df.columns else None
        ask0 = pd.to_numeric(df['ask_px_00'], errors='coerce') if 'ask_px_00' in df.columns else None

        if bid0 is not None or ask0 is not None:
            if bid0 is not None and ask0 is not None:
                df['price'] = ((bid0.where(bid0 > 0, np.nan) + ask0.where(ask0 > 0, np.nan)) / 2.0)
                df['price'] = df['price'].fillna(bid0.where(bid0 > 0, np.nan)).fillna(ask0.where(ask0 > 0, np.nan))
            elif bid0 is not None:
                df['price'] = bid0.where(bid0 > 0, np.nan)
            else:
                df['price'] = ask0.where(ask0 > 0, np.nan)

            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df = df[df['price'].notna() & (df['price'] > 0)].reset_index(drop=True)
            df['price'] = df['price'].astype('float32')
        else:
            print('  ⚠️ لا يوجد عمود price!')

    if 'size' not in df.columns:
        for alt in ['qty', 'quantity', 'volume']:
            if alt in df.columns:
                df['size'] = df[alt]
                break
    if 'size' in df.columns:
        df['size'] = pd.to_numeric(df['size'], errors='coerce').fillna(1).astype(int)

    if 'order_id' not in df.columns:
        df['order_id'] = range(len(df))

    if 'symbol' not in df.columns:
        df['symbol'] = df['instrument_id'].astype(str) if 'instrument_id' in df.columns else 'UNKNOWN'

    bid_px_cols = [c for c in df.columns if c.startswith('bid_px_')]
    ask_px_cols = [c for c in df.columns if c.startswith('ask_px_')]
    bid_sz_cols = [c for c in df.columns if c.startswith('bid_sz_')]
    ask_sz_cols = [c for c in df.columns if c.startswith('ask_sz_')]

    for col in bid_px_cols + ask_px_cols:
        vals = pd.to_numeric(df[col], errors='coerce').fillna(0)
        if vals[vals > 0].count() > 10:
            q1, q3 = vals[vals > 0].quantile(0.25), vals[vals > 0].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                hi = q3 + 5 * iqr
                vals = vals.where(vals <= hi, 0)
        df[col] = vals.astype('float32')
        
    for col in bid_sz_cols + ask_sz_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    n_final = len(df)
    n_acts  = df['action'].value_counts().to_dict() if 'action' in df.columns else {}
    print(f"  ✅ Normalize: {n_final:,} صف | actions={n_acts}")
    return df

def _process_mbo_chunk(args):
    df_chunk, cal_params = args

    # إعادة بناء الـ Engines داخل الـ Worker
    from modules.microstructure   import FastMicrostructureEngine, FastTapeSpeedTracker, TRADE_ACTIONS, AbsorptionIntensityEngine, CancelRatioEngine
    from modules.micro_volatility import MicroVolatilityEngine
    from modules.market_research_features import KylesLambdaEngine, HawkesIntensityEngine, VNETEngine
    from modules.fisher_alpha     import FastFisherAlpha
    from modules.fim_anomaly      import FastFIMDetector
    from modules.context_features import MomentumContextEngine, LiquiditySweepDetector, DailyContextEngine
    from modules.session_features import SessionVWAPEngine
    
    micro      = FastMicrostructureEngine()
    tape       = FastTapeSpeedTracker()
    mv         = MicroVolatilityEngine()
    kyle       = KylesLambdaEngine(window=50)
    hawkes     = HawkesIntensityEngine()
    vnet       = VNETEngine()
    fisher     = FastFisherAlpha()
    fim        = FastFIMDetector()

    sweep_thresh   = cal_params.get('sweep_thresh', 0.05)
    min_price_move = cal_params.get('min_price_move', 0.25)
    
    absorb     = AbsorptionIntensityEngine(min_price_move=min_price_move)
    cancel_eng = CancelRatioEngine()
    ctx        = MomentumContextEngine()
    sweep      = LiquiditySweepDetector(sweep_threshold=sweep_thresh)
    
    # 🔴 محركات السياق الجديدة
    daily_ctx  = DailyContextEngine(default_adr=80.0 * cal_params.get('tick_size', 0.0001))
    vwap_eng   = SessionVWAPEngine()

    cvd = 0; last_cancel_ratio = 0.0; out = []

    for row in df_chunk.itertuples(index=False):
        price  = float(getattr(row,'price',0) or 0)
        size   = int(getattr(row,'size', getattr(row,'qty', getattr(row,'volume',0))) or 0)
        action = str(getattr(row,'action','')).strip().upper()
        side   = str(getattr(row,'side','')).strip().upper()
        oid    = getattr(row,'order_id', None)
        raw_t  = getattr(row,'ts_event', None)
        try:    ts=pd.to_datetime(raw_t); ts_ns=int(ts.value)
        except: ts=pd.Timestamp.now();   ts_ns=int(ts.value)

        micro.process_mbo_tick(action, oid, side, size, price, ts_ns)
        _cr = cancel_eng.process_tick(action, oid, size)
        if _cr != 0.0: last_cancel_ratio = _cr
        cancel_ratio = last_cancel_ratio
        tape_speed   = tape.update_and_get_speed(ts, action)
        mu_atr,vbi,iet = mv.process_tick(action, price, size, ts_ns)
        lsweep       = sweep.update(price)
        hawkes_val   = hawkes.update(ts_ns, action)

        # تحديث سياق اليوم (حتى بدون تنفيذ Trade)
        ib_stat, r_fuel, f_exh = daily_ctx.update(ts, price)

        if action in TRADE_ACTIONS:
            is_buy = False
            if side in ('B','BID'):   
                cvd += size
                is_buy = True
            elif side in ('A','ASK'): 
                cvd -= size
            
            aii                      = absorb.update(price, cvd)
            fs                       = fisher.update_and_get_signal(price, cvd)
            an                       = fim.detect_stop_hunts(price)
            cvd_mom, cvd_div, t_str, corr_d = ctx.update(price, cvd)
            kyle_val                 = kyle.update(price, size)
            vnet_val                 = vnet.update(price, size, side)
            
            # 🔴 تحديث الـ VWAP
            cur_vwap, v_zscore, v_slope, sess_cvd = vwap_eng.update(ts, price, float(size), is_buy)
            
            out.append({
                'ts_event':ts, 'price':price, 'size':size, 'action':action, 'side':side,
                'cvd':cvd,
                'absorption_intensity':aii, 'cancel_ratio':cancel_ratio,
                'micro_atr':mu_atr, 'volume_burst':vbi,
                'inter_event_time':iet, 'fisher_signal':fs,
                'anomaly':an, 'tape_speed':tape_speed,
                'cvd_momentum':cvd_mom, 'cvd_price_divergence':cvd_div,
                'trend_strength':t_str, 'correction_depth':corr_d,
                'liquidity_sweep':lsweep,
                'kyle_lambda':kyle_val, 'hawkes_intensity':hawkes_val,
                'vnet':vnet_val, 'liquidity_gaps':0.0,
                # الفيتشرز الجديدة
                'ib_status': ib_stat, 'remaining_fuel': r_fuel, 'fuel_exhausted': f_exh,
                'current_vwap': cur_vwap, 'vwap_z_score': v_zscore, 
                'vwap_slope': v_slope, 'session_cvd': sess_cvd
            })

    return pd.DataFrame(out)

def _process_mbo(df_mbo, n_workers: int = None):
    print("  🔧 Auto-Calibration...")
    cal     = AutoCalibrator(n_ticks=2000).fit(df_mbo)
    engines = cal.build_engines(include_extended=True)

    cal_params = {
        'tick_size':      cal.tick_size,
        'min_price_move': cal.min_price_move,
        'typical_size':   cal.typical_size,
        'volatility':     cal.volatility,
        'sweep_thresh':   getattr(engines.get('sweep', object()), 'threshold', 0.05),
    }

    if n_workers is None:
        n_workers = get_multiprocessing_workers()

    if n_workers <= 1 or len(df_mbo) < 10_000:
        return _process_mbo_sequential(df_mbo, engines, cal_params)

    print(f"  ⚡ Multiprocessing: {n_workers} workers على {len(df_mbo):,} events...")

    chunk_size = max(1000, len(df_mbo) // n_workers)
    chunks = [(df_mbo.iloc[i:i+chunk_size].copy(), cal_params) for i in range(0, len(df_mbo), chunk_size)]

    try:
        ctx_mp = multiprocessing.get_context('spawn')
        with ctx_mp.Pool(processes=min(n_workers, len(chunks))) as pool:
            results = pool.map(_process_mbo_chunk, chunks)

        df_out = pd.concat(results, ignore_index=True)
        df_out = df_out.sort_values('ts_event').reset_index(drop=True)

        # ══════════════════════════════════════════════════════════
        # CVD FIX: إعادة حساب CVD الحقيقي من side/action مباشرة
        # diff().cumsum() خاطئ لأن كل chunk بدأ من صفر
        # الحل: true_delta = +size (Buy) أو -size (Sell) ثم cumsum
        # ══════════════════════════════════════════════════════════
        is_trade = df_out['action'].astype(str).str.upper().isin({'T','F','TRADE','EXECUTE','0'})
        is_buy   = df_out['side'].astype(str).str.upper().isin({'B','BID'})
        is_sell  = df_out['side'].astype(str).str.upper().isin({'A','ASK','S','SELL'})

        true_trade_vol = np.zeros(len(df_out), dtype=np.float64)
        buy_mask  = is_trade & is_buy
        sell_mask = is_trade & is_sell
        true_trade_vol[buy_mask.values]  =  df_out.loc[buy_mask,  'size'].values
        true_trade_vol[sell_mask.values] = -df_out.loc[sell_mask, 'size'].values

        df_out['cvd'] = true_trade_vol.cumsum()

        print(f"  ✅ Multiprocessing: {len(df_out):,} trades")
        return df_out

    except Exception as e:
        print(f"  ⚠️ Multiprocessing فشل ({e}) — sequential fallback")
        return _process_mbo_sequential(df_mbo, engines, cal_params)

def _process_mbo_sequential(df_mbo, engines, cal_params):
    micro      = FastMicrostructureEngine()
    absorb     = engines['absorb']
    cancel_eng = engines['cancel']
    ctx        = engines['momentum']
    sweep      = engines['sweep']
    fisher     = engines['fisher']
    fim        = engines['fim']
    kyle       = engines['kyle']
    hawkes     = engines['hawkes']
    vnet       = engines['vnet']
    tape       = FastTapeSpeedTracker()
    mv         = MicroVolatilityEngine()
    
    # 🔴 محركات السياق الجديدة
    daily_ctx  = DailyContextEngine(default_adr=80.0 * cal_params.get('tick_size', 0.0001))
    vwap_eng   = SessionVWAPEngine()
    
    cvd        = 0; last_cancel_ratio = 0.0; out = []

    for row in _prog(df_mbo.itertuples(index=False), desc='MBO', total=len(df_mbo)):
        price  = float(getattr(row,'price',0) or 0)
        size   = int(getattr(row,'size', getattr(row,'qty', getattr(row,'volume',0))) or 0)
        action = str(getattr(row,'action','')).strip().upper()
        side   = str(getattr(row,'side','')).strip().upper()
        oid    = getattr(row,'order_id', None)
        raw_t  = getattr(row,'ts_event', getattr(row,'timestamp', None))
        try:    ts=pd.to_datetime(raw_t); ts_ns=int(ts.value)
        except: ts=pd.Timestamp.now();   ts_ns=int(ts.value)

        micro.process_mbo_tick(action, oid, side, size, price, ts_ns)
        _cr = cancel_eng.process_tick(action, oid, size)
        if _cr != 0.0: last_cancel_ratio = _cr
        cancel_ratio = last_cancel_ratio
        tape_speed   = tape.update_and_get_speed(ts, action)
        mu_atr,vbi,iet = mv.process_tick(action, price, size, ts_ns)
        lsweep       = sweep.update(price)
        hawkes_val   = hawkes.update(ts_ns, action)

        ib_stat, r_fuel, f_exh = daily_ctx.update(ts, price)

        if action in TRADE_ACTIONS:
            is_buy = False
            if side in ('B','BID'):   
                cvd += size
                is_buy = True
            elif side in ('A','ASK'): 
                cvd -= size
            
            aii                      = absorb.update(price, cvd)
            fs                       = fisher.update_and_get_signal(price, cvd)
            an                       = fim.detect_stop_hunts(price)
            cvd_mom, cvd_div, t_str, corr_d = ctx.update(price, cvd)
            kyle_val                 = kyle.update(price, size)
            vnet_val                 = vnet.update(price, size, side)
            
            cur_vwap, v_zscore, v_slope, sess_cvd = vwap_eng.update(ts, price, float(size), is_buy)
            
            out.append({
                'ts_event':ts, 'price':price, 'size':size, 'action':action, 'side':side,
                'cvd':cvd,
                'absorption_intensity':aii, 'cancel_ratio':cancel_ratio,
                'micro_atr':mu_atr, 'volume_burst':vbi,
                'inter_event_time':iet, 'fisher_signal':fs,
                'anomaly':an, 'tape_speed':tape_speed,
                'cvd_momentum':cvd_mom, 'cvd_price_divergence':cvd_div,
                'trend_strength':t_str, 'correction_depth':corr_d,
                'liquidity_sweep':lsweep,
                'kyle_lambda':kyle_val, 'hawkes_intensity':hawkes_val,
                'vnet':vnet_val, 'liquidity_gaps':0.0,
                'ib_status': ib_stat, 'remaining_fuel': r_fuel, 'fuel_exhausted': f_exh,
                'current_vwap': cur_vwap, 'vwap_z_score': v_zscore, 
                'vwap_slope': v_slope, 'session_cvd': sess_cvd
            })

    return pd.DataFrame(out)

def _add_rolling_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['cvd_roc_10']  = df['cvd'].diff(10).fillna(0)
    df['cvd_roc_50']  = df['cvd'].diff(50).fillna(0)
    df['cvd_roc_200'] = df['cvd'].diff(200).fillna(0)
    df['cvd_accel'] = df['cvd_roc_10'] - df['cvd_roc_50']
    df['volume_accel'] = df['volume_burst'].diff(10).fillna(0)
    return df

def _process_mbp10(df_mbp, tick_size: float = 0.0001):
    ob   = OrderBookSnapshotEngine()
    sp   = SpoofingDetector(large_mult=1.5)
    gaps = LiquidityGapsEngine(levels=10, gap_threshold=1.0)
    scanner = OrderWallScanner(wall_mult=3.0, gap_mult=2.0, levels=10)
    
    # 🔴 محرك حيطان السيولة
    walls_eng = LiquidityWallsEngine(depth_levels=10, wall_threshold_multiplier=3.0)
    
    out  = []

    for row in _prog(df_mbp.itertuples(index=False), desc='MBP10', total=len(df_mbp)):
        action  = str(getattr(row,'action','')).strip().upper()
        price   = float(getattr(row,'price', 0) or 0) # السعر التقريبي من الـ MBP
        raw_t   = getattr(row,'ts_event', getattr(row,'timestamp', None))
        try:    ts=pd.to_datetime(raw_t)
        except: ts=pd.Timestamp.now()
        
        row_d   = row._asdict()
        if price <= 0:
            bid0 = float(row_d.get('bid_px_00', 0) or 0)
            ask0 = float(row_d.get('ask_px_00', 0) or 0)
            if bid0 > 0 and ask0 > 0:
                price = (bid0 + ask0) / 2.0
            elif bid0 > 0:
                price = bid0
            elif ask0 > 0:
                price = ask0

        obi     = ob.compute_obi(row_d)
        sr, sd  = sp.process_snapshot(row_d, action)
        gap_val = gaps.update(row_d, tick_size=tick_size)
        
        # استخراج العروض والطلبات لحساب حيطان السيولة
        bids = [[row_d.get(f'bid_px_{i:02d}',0), row_d.get(f'bid_sz_{i:02d}',0)] for i in range(10)]
        asks = [[row_d.get(f'ask_px_{i:02d}',0), row_d.get(f'ask_sz_{i:02d}',0)] for i in range(10)]
        
        dist_bid_wall, dist_ask_wall = walls_eng.update(price, bids, asks)
        
        scan = scanner.scan(row_d, tick_size=tick_size, cvd=0.0, volume=0.0)

        out.append({
            'ts_event': ts,
            'obi': obi,
            'spoofing_ratio': sr,
            'spoofing_duration': sd,
            'liquidity_gaps': gap_val,
            'dist_to_bid_wall': dist_bid_wall,
            'dist_to_ask_wall': dist_ask_wall,
            'mid_price': scan.get('mid_price', price),
            'micro_price': scan.get('micro_price', price),
            'spread': scan.get('spread', 0.0),
            'bid_wall_px': scan.get('bid_wall_px'),
            'ask_wall_px': scan.get('ask_wall_px'),
            'bid_gap_px': scan.get('bid_gap_px'),
            'ask_gap_px': scan.get('ask_gap_px'),
            'bid_gap_size': scan.get('bid_gap_size', 0.0),
            'ask_gap_size': scan.get('ask_gap_size', 0.0),
            'bid_wall_strength': scan.get('bid_wall_strength', scan.get('bid_wall_str', 0.0)),
            'ask_wall_strength': scan.get('ask_wall_strength', scan.get('ask_wall_str', 0.0)),
            'distance_to_wall': scan.get('distance_to_wall', 0.0),
            'gap_size': scan.get('gap_size', 0.0),
            'liquidity_density': scan.get('liquidity_density', 0.0),
        })

    return pd.DataFrame(out)

def _merge(df_mbo, df_mbp):
    print("  🔗 merge_asof MBO + MBP10...")
    df_mbo = df_mbo.copy()
    df_mbp = df_mbp.copy()

    df_mbo['ts_event'] = pd.to_datetime(df_mbo.get('ts_event'), errors='coerce')
    df_mbp['ts_event'] = pd.to_datetime(df_mbp.get('ts_event'), errors='coerce')

    df_mbo = df_mbo[df_mbo['ts_event'].notna()].sort_values('ts_event').reset_index(drop=True)
    df_mbp = df_mbp[df_mbp['ts_event'].notna()].sort_values('ts_event').reset_index(drop=True)

    if len(df_mbo) == 0:
        raise ValueError('MBO data has no valid ts_event rows after normalization')

    if len(df_mbp) == 0:
        df = df_mbo.copy()
        for c in [
            'obi', 'spoofing_ratio', 'spoofing_duration', 'dist_to_bid_wall', 'dist_to_ask_wall',
            'mid_price', 'micro_price', 'spread',
            'bid_gap_size', 'ask_gap_size', 'bid_wall_strength', 'ask_wall_strength',
            'distance_to_wall', 'gap_size', 'liquidity_density',
        ]:
            df[c] = 0.0
    else:
        df = pd.merge_asof(df_mbo, df_mbp, on='ts_event',
                           direction='backward', tolerance=pd.Timedelta('5s'))
                       
    for c in [
        'obi', 'spoofing_ratio', 'spoofing_duration', 'dist_to_bid_wall', 'dist_to_ask_wall',
        'mid_price', 'micro_price', 'spread',
        'bid_gap_size', 'ask_gap_size', 'bid_wall_strength', 'ask_wall_strength',
        'distance_to_wall', 'gap_size', 'liquidity_density',
    ]:
        df[c] = df[c].fillna(0.0)

    liq = LiquidityTrapDetector()
    df['liquidity_trap'] = [
        liq.update(float(r.get('price',0) or 0), float(r.get('obi',0) or 0))
        for _, r in df.iterrows()
    ]

    for col in df.select_dtypes(include='float64').columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include='int64').columns:
        df[col] = df[col].astype('int32')
        
    gc.collect()
    return df

def _get_session(ts):
    h, m = ts.hour, ts.minute
    t = h * 60 + m
    for name, h1, m1, h2, m2 in SESSIONS:
        s1 = h1*60+m1; s2 = h2*60+m2
        if s1 <= t < s2:
            return name
    return None

def _compute_liquidity_score(group):
    vb  = float(group['volume_burst'].mean()) if 'volume_burst' in group else 1.0
    aii = float(group['absorption_intensity'].abs().mean()) if 'absorption_intensity' in group else 0.0
    obi = float(group['obi'].abs().mean() + 0.01) if 'obi' in group else 0.01
    return vb * aii * obi

def _compute_volatility_barriers(group, ema_alpha=0.1):
    prices = group['price'].values.astype(np.float64)
    if len(prices) < 3: return 0.001, 0.001
    returns = np.abs(np.diff(prices))
    ema_vol = returns[0]
    for r in returns[1:]: ema_vol = ema_alpha * r + (1 - ema_alpha) * ema_vol
    ema_vol = max(ema_vol, 1e-8)
    # ══════════════════════════════════════════════════════════
    # RR FIX: TP=SL=EMA_vol (1:1) بدل 2:1 الذي يُحيّز labels نحو LONG
    # في الواقع: الـ label يعتمد على من يُضرب أولاً (TP أو SL)
    # نسبة 1:1 تُعطي توزيعاً أكثر واقعية ومتوازناً
    # ══════════════════════════════════════════════════════════
    return ema_vol * 1.5, ema_vol * 1.0  # TP=1.5x, SL=1.0x (RR=1.5:1 معقول)

def _compute_bias(group):
    if len(group) < 2: return BIAS_NEUTRAL
    prices = group['price'].values.astype(np.float64)
    open_p = prices[0]
    tp, sl = _compute_volatility_barriers(group)
    upper, lower = open_p + tp, open_p - sl

    for p in prices[1:]:
        if p >= upper: return BIAS_LONG
        if p <= lower: return BIAS_SHORT

    move = prices[-1] - open_p
    if move > tp * 0.5: return BIAS_LONG
    elif move < -sl * 0.5: return BIAS_SHORT
    return BIAS_NEUTRAL

def _compute_setup_threshold(group):
    if len(group) == 0: return SETUP_MIXED
    signals = []
    if 'absorption_intensity' in group.columns and float(group['absorption_intensity'].mean()) > 0.3: signals.append('absorption')
    if 'spoofing_ratio' in group.columns and float(group['spoofing_ratio'].mean()) > 0.2: signals.append('spoofing')
    if 'obi' in group.columns and float(group['obi'].abs().mean()) > 0.3: signals.append('obi')
    
    if len(signals) > 1: return SETUP_MIXED
    elif 'absorption' in signals: return SETUP_ABSORPTION
    elif 'spoofing' in signals: return SETUP_SPOOFING
    elif 'obi' in signals: return SETUP_OBI
    return SETUP_MIXED

def _compute_setup(group, setup_window=900):
    if len(group) == 0: return SETUP_MIXED
    start  = group['ts_event'].iloc[0]
    window = group[group['ts_event'] <= start + pd.Timedelta(seconds=setup_window)]
    if len(window) == 0: window = group
    return _compute_setup_threshold(window)

def _label_sessions(df):
    print("\n📅 Session Labeling...")
    df = df.copy()
    df['date']    = df['ts_event'].dt.date
    df['session'] = df['ts_event'].apply(_get_session)
    df['session'] = df['session'].fillna('off')

    df['bias_label']   = BIAS_NEUTRAL
    df['setup_label']  = SETUP_MIXED
    df['is_expansion'] = 0
    df['liq_score']    = 0.0

    days = df['date'].unique()
    print(f"  أيام: {len(days)}  |  sessions لكل يوم: 3")

    for day in days:
        day_df = df[df['date'] == day]
        session_scores = {}

        for sess_name, *_ in SESSIONS:
            sess_df = day_df[day_df['session'] == sess_name]
            if len(sess_df) >= 50: session_scores[sess_name] = _compute_liquidity_score(sess_df)

        if not session_scores: continue

        max_sess  = max(session_scores, key=session_scores.get)

        for sess_name, *_ in SESSIONS:
            sess_mask = (df['date']==day) & (df['session']==sess_name)
            sess_df   = df[sess_mask]
            if len(sess_df) == 0: continue

            df.loc[sess_mask, 'liq_score'] = session_scores.get(sess_name, 0.0)

            if sess_name == max_sess:
                df.loc[sess_mask, 'bias_label']   = _compute_bias(sess_df)
                df.loc[sess_mask, 'setup_label']  = _compute_setup(sess_df)
                df.loc[sess_mask, 'is_expansion'] = 1
            else:
                df.loc[sess_mask, 'bias_label']   = BIAS_NEUTRAL
                df.loc[sess_mask, 'setup_label']  = SETUP_MIXED
                df.loc[sess_mask, 'is_expansion'] = 0

    df['conf_label'] = ((df['is_expansion'] == 1) & (df['bias_label'] != BIAS_NEUTRAL)).astype(np.float32)
    return df

def _compute_rolling_stats(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    print(f"\n📊 Rolling Stats (window={window} ticks)...")
    df = df.copy()
    roll_cols = []
    
    # لا نقوم بعمل Rolling للـ Embeddings
    base_features = [c for c in MODEL_FEATURE_COLS if not c.startswith('emb_')]
    
    new_roll_dfs = []
    for col in base_features:
        if col not in df.columns: df[col] = 0.0
        s = df[col].astype(np.float32)
        # past-only rolling: the current row must not peek into itself or future rows
        s_past = s.shift(1)
        mean_col = f'{col}_mean{window}'
        std_col  = f'{col}_std{window}'
        new_roll_dfs.append(pd.DataFrame({
            mean_col: s_past.rolling(window=window, min_periods=1).mean().fillna(0).astype(np.float32),
            std_col:  s_past.rolling(window=window, min_periods=1).std().fillna(0).astype(np.float32),
        }))
        roll_cols.extend([mean_col, std_col])
    if new_roll_dfs:
        df = pd.concat([df] + new_roll_dfs, axis=1)

    print(f"  ✅ أضاف {len(roll_cols)} عمود (mean+std لكل feature)")
    return df, roll_cols

def _load_scaler_params_from_path(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _deeplob_runtime_limits(lob_event_sample: int = LOB_EVENT_SAMPLE_DEFAULT) -> dict:
    max_events = int(os.environ.get('QUANTSYSTEM_MAX_LOB_EVENTS', DEEPLOB_MAX_EVENTS_DEFAULT))
    max_tensors = int(os.environ.get('QUANTSYSTEM_MAX_LOB_TENSORS', DEEPLOB_MAX_TENSORS_DEFAULT))
    max_gb = float(os.environ.get('QUANTSYSTEM_MAX_LOB_GB', DEEPLOB_MAX_GB_DEFAULT))
    force = os.environ.get('QUANTSYSTEM_FORCE_LOB', '').strip() == '1'
    return {
        'max_events': max_events,
        'max_tensors': max_tensors,
        'max_bytes': int(max_gb * (1024 ** 3)),
        'max_gb': max_gb,
        'force': force,
        'lob_event_sample': int(max(lob_event_sample, 1)),
    }


def _lob_source_frame(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    if kind == 'mbo':
        cols = ['ts_event', 'action', 'price', 'size', 'side']
    else:
        cols = ['ts_event']
        for i in range(10):
            cols.extend([f'bid_px_{i:02d}', f'bid_sz_{i:02d}', f'ask_px_{i:02d}', f'ask_sz_{i:02d}'])
    cols = [c for c in cols if c in df.columns]
    return df.loc[:, cols].copy(deep=False)


def _select_event_rich_lob_emit_positions(
    df_labeled: pd.DataFrame,
    lob_mbp_src: pd.DataFrame,
    max_events: int = LOB_EVENT_SAMPLE_DEFAULT,
) -> tuple[np.ndarray, dict]:
    if (
        df_labeled is None or len(df_labeled) == 0 or
        lob_mbp_src is None or len(lob_mbp_src) == 0
    ):
        return np.array([], dtype=np.int32), {'selected_events': 0, 'selected_emit_positions': 0}

    event_col = 'train_event_flag' if 'train_event_flag' in df_labeled.columns else 'event_flag'
    label_cols = ['ts_event', 'event_flag', 'train_event_flag', 'bias_label', 'signal_quality']
    candidates = df_labeled.loc[:, [c for c in label_cols if c in df_labeled.columns]].copy()
    if len(candidates) == 0 or 'ts_event' not in candidates.columns:
        return np.array([], dtype=np.int32), {'selected_events': 0, 'selected_emit_positions': 0}

    candidates['ts_event'] = pd.to_datetime(candidates['ts_event'], utc=True, errors='coerce').dt.tz_localize(None)
    candidates = candidates.dropna(subset=['ts_event']).sort_values('ts_event').reset_index(drop=True)
    candidates['event_flag'] = pd.to_numeric(candidates.get('event_flag', 0), errors='coerce').fillna(0).astype(np.int8)
    candidates['train_event_flag'] = pd.to_numeric(candidates.get('train_event_flag', candidates['event_flag']), errors='coerce').fillna(0).astype(np.int8)
    candidates['bias_label'] = pd.to_numeric(candidates.get('bias_label', DIR_NEUTRAL), errors='coerce').fillna(DIR_NEUTRAL).astype(np.int8)
    candidates['signal_quality'] = pd.to_numeric(candidates.get('signal_quality', 0), errors='coerce').fillna(0).astype(np.int8)
    candidates = candidates[
        (candidates[event_col] == 1) &
        (candidates['bias_label'] != DIR_NEUTRAL)
    ].reset_index(drop=True)
    if len(candidates) == 0:
        return np.array([], dtype=np.int32), {'selected_events': 0, 'selected_emit_positions': 0}

    max_events = int(max(max_events, 1))
    strong = candidates[candidates['signal_quality'] == QUALITY_STRONG].copy()
    weak = candidates[candidates['signal_quality'] == QUALITY_WEAK].copy()

    selected_parts = []
    if len(strong):
        selected_parts.append(strong.iloc[:max_events].copy())
    remaining = max_events - sum(len(part) for part in selected_parts)
    if remaining > 0 and len(weak):
        selected_parts.append(weak.iloc[:remaining].copy())
    if not selected_parts:
        return np.array([], dtype=np.int32), {'selected_events': 0, 'selected_emit_positions': 0}

    selected = pd.concat(selected_parts, axis=0, ignore_index=True).sort_values('ts_event').drop_duplicates('ts_event')
    mbp_df = lob_mbp_src[['ts_event']].copy()
    mbp_df['ts_event'] = pd.to_datetime(mbp_df['ts_event'], utc=True, errors='coerce').dt.tz_localize(None)
    mbp_df = mbp_df.dropna(subset=['ts_event']).sort_values('ts_event').reset_index(drop=True)
    mbp_df['mbp_pos'] = np.arange(len(mbp_df), dtype=np.int32)
    if len(mbp_df) == 0:
        return np.array([], dtype=np.int32), {'selected_events': int(len(selected)), 'selected_emit_positions': 0}

    aligned = pd.merge_asof(
        selected[['ts_event']].sort_values('ts_event'),
        mbp_df[['ts_event', 'mbp_pos']],
        on='ts_event',
        direction='backward',
    ).dropna(subset=['mbp_pos'])
    emit_positions = aligned['mbp_pos'].astype(np.int32).drop_duplicates().to_numpy()
    meta = {
        'directional_events_available': int(len(candidates)),
        'event_col': event_col,
        'strong_available': int(len(strong)),
        'weak_available': int(len(weak)),
        'selected_events': int(len(selected)),
        'selected_emit_positions': int(len(emit_positions)),
        'selected_strong': int(sum(len(part) for part in selected_parts if 'signal_quality' in part.columns and int(part['signal_quality'].iloc[0]) == QUALITY_STRONG) if selected_parts else 0),
        'selected_weak': int(sum(len(part) for part in selected_parts if 'signal_quality' in part.columns and int(part['signal_quality'].iloc[0]) == QUALITY_WEAK) if selected_parts else 0),
    }
    return emit_positions, meta


def _build_refinery_split_context(
    df: pd.DataFrame,
    train_frac: float = 0.80,
) -> dict:
    n = len(df)
    if n == 0:
        return {
            'split_idx': 0,
            'split_time': pd.NaT,
            'label_end': pd.Series(dtype='datetime64[ns]'),
            'train_row_ok': np.array([], dtype=bool),
            'holdout_row_ok': np.array([], dtype=bool),
            'purged_row_ok': np.array([], dtype=bool),
            'train_idx': np.array([], dtype=np.int32),
        }

    ts = pd.to_datetime(df.get('ts_event', pd.Series(pd.RangeIndex(n))), utc=True, errors='coerce').dt.tz_localize(None)
    ts = ts.ffill()
    if ts.isna().any():
        base = pd.Timestamp('2026-01-01')
        ts = pd.Series([base + pd.Timedelta(seconds=i) for i in range(n)], index=df.index)

    split_idx = min(max(int(n * train_frac), 1), max(n - 1, 1))
    split_time = ts.iloc[min(split_idx, n - 1)]
    label_end = pd.to_datetime(df.get('label_end_ts', ts), utc=True, errors='coerce').dt.tz_localize(None).fillna(ts)

    row_ids = np.arange(n)
    train_row_ok = (row_ids < split_idx) & (label_end.values < split_time.to_datetime64())
    holdout_row_ok = row_ids >= split_idx
    purged_row_ok = (~train_row_ok) & (~holdout_row_ok)
    train_idx = np.flatnonzero(train_row_ok)

    if train_idx.size == 0:
        train_idx = np.arange(min(split_idx, n), dtype=np.int32)
        train_row_ok = np.zeros(n, dtype=bool)
        train_row_ok[train_idx] = True
        purged_row_ok = (~train_row_ok) & (~holdout_row_ok)

    return {
        'split_idx': int(split_idx),
        'split_time': split_time,
        'label_end': label_end,
        'train_row_ok': train_row_ok.astype(bool),
        'holdout_row_ok': holdout_row_ok.astype(bool),
        'purged_row_ok': purged_row_ok.astype(bool),
        'train_idx': train_idx.astype(np.int32),
    }


def _normalize_and_save(
    df,
    output_dir,
    rolling_window: int = 50,
    external_scaler_path: str | None = None,
    fit_aux_models: bool = True,
):
    print("\n📐 Normalization (RobustScaler — train-only fit)...")
    df = df.copy()
    fit_aux_models = fit_aux_models and os.environ.get('QUANTSYSTEM_SKIP_HEAVY_ML', '').strip() != '1'

    # Preserve raw model inputs so train_v19.py can fit fold-specific scalers
    # instead of inheriting a globally pre-scaled CSV.
    for col in CATBOOST_ADVISOR_FEATURES:
        raw_col = f'{RAW_STAT_PREFIX}{col}'
        if col not in df.columns:
            df[col] = 0.0
        df[raw_col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(np.float32)

    # ══════════════════════════════════════════════════════════
    # ANTI-LEAKAGE FIX: FIT scaler على train split فقط (80%)
    # TRANSFORM على كامل الداتا (train + val)
    # احفظ params للـ live trading لضمان consistency
    # ══════════════════════════════════════════════════════════
    split_ctx = _build_refinery_split_context(df, train_frac=0.80)
    train_mask = split_ctx['train_row_ok']
    train_idx = split_ctx['train_idx']
    split_time = split_ctx['split_time']
    n_total = len(df)

    df['is_train_slice'] = train_mask.astype(np.int8)
    df['is_holdout_slice'] = split_ctx['holdout_row_ok'].astype(np.int8)
    df['is_purged_slice'] = split_ctx['purged_row_ok'].astype(np.int8)
    df['dataset_slice'] = np.where(
        split_ctx['holdout_row_ok'],
        'holdout',
        np.where(train_mask, 'train', 'purged')
    )

    split_meta = {
        'split_idx': int(split_ctx['split_idx']),
        'split_time': None if pd.isna(split_time) else str(split_time),
        'train_rows': int(train_mask.sum()),
        'holdout_rows': int(split_ctx['holdout_row_ok'].sum()),
        'purged_rows': int(split_ctx['purged_row_ok'].sum()),
    }
    with open(os.path.join(output_dir, 'refinery_split.json'), 'w') as f:
        json.dump(split_meta, f, indent=2)
    print(
        "  ✅ Split Context: "
        f"train={split_meta['train_rows']:,} | "
        f"holdout={split_meta['holdout_rows']:,} | "
        f"purged={split_meta['purged_rows']:,} | "
        f"split_time={split_meta['split_time']}"
    )

    scaler_params = {}  # يُحفظ لاحقاً للـ live trading
    using_external_scaler = bool(external_scaler_path and os.path.exists(external_scaler_path))
    if using_external_scaler:
        scaler_params = _load_scaler_params_from_path(external_scaler_path)
        print(f"  ✅ Using external scaler params: {external_scaler_path}")

    base_features = [c for c in MODEL_FEATURE_COLS if not c.startswith('emb_')]
    for col in base_features:
        if col not in df.columns:
            df[col] = 0.0; continue
        s = df[col].astype(np.float32).fillna(0.0)

        if using_external_scaler:
            p = scaler_params.get(col, {'type': 'zero'})
            typ = p.get('type', 'zero')
            if typ == 'binary':
                df[col] = s
            elif typ == 'robust':
                df[col] = ((s - float(p.get('median', 0.0))) / max(float(p.get('iqr', 0.0)), 1e-8)).clip(-10, 10)
            elif typ == 'minmax':
                smin = float(p.get('min', 0.0)); smax = float(p.get('max', 0.0)); rng = max(smax - smin, 1e-8)
                df[col] = ((s - smin) / rng * 2 - 1).clip(-10, 10)
            else:
                df[col] = 0.0
            continue
        else:
            if col in BINARY_FEATURES:
                df[col] = s
                scaler_params[col] = {'type': 'binary'}
                continue

            # FIT على train فقط
            s_train = s.iloc[train_idx]
            median_  = float(s_train.median())
            q1, q3   = float(s_train.quantile(0.25)), float(s_train.quantile(0.75))
            iqr      = q3 - q1

            if iqr > 1e-8:
                df[col] = ((s - median_) / iqr).clip(-10, 10)
                scaler_params[col] = {'type': 'robust', 'median': median_, 'iqr': iqr}
            elif s_train.abs().max() > 1e-8:
                smin = float(s_train.min()); smax = float(s_train.max()); rng = smax - smin
                if rng > 1e-8:
                    df[col] = ((s - smin) / rng * 2 - 1).clip(-10, 10)
                    scaler_params[col] = {'type': 'minmax', 'min': smin, 'max': smax}
                else:
                    df[col] = 0.0
                    scaler_params[col] = {'type': 'zero'}
            else:
                df[col] = 0.0
                scaler_params[col] = {'type': 'zero'}

    # حفظ params للـ live trading
    scaler_path = os.path.join(output_dir, 'scaler_params.json')
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    print(f"  ✅ Scaler params محفوظة: {scaler_path} ({len(scaler_params)} features)")

    # 2. Compute Rolling Stats
    df, roll_cols = _compute_rolling_stats(df, window=rolling_window)
    roll_col_list = [c for c in roll_cols if c in df.columns]

    # ══════════════════════════════════════════════════════════
    # ANTI-LEAKAGE FIX [AE]: train AE على train split فقط
    # ══════════════════════════════════════════════════════════
    print(f"\n🧠 Deep Autoencoder (Extracting {EMBEDDINGS_DIM} Embeddings — train-only fit)...")
    if fit_aux_models and len(roll_col_list) >= 10:
        from modules.autoencoder_extractor import AutoencoderExtractor

        X_roll = df[roll_col_list].fillna(0).values.astype('float32')
        n_ae = len(X_roll)
        X_roll_train = X_roll[train_idx]
        if len(X_roll_train) == 0:
            X_roll_train = X_roll[:max(1, min(n_ae, split_ctx['split_idx']))]
        ae_train_end = len(X_roll_train)

        ae = AutoencoderExtractor(input_dim=len(roll_col_list), bottleneck=EMBEDDINGS_DIM)
        ae.fit(X_roll_train, epochs=40, batch_size=512, output_dir=output_dir)

        # transform على كامل الداتا (train + val)
        df['dl_anomaly_score'] = ae.score(X_roll).astype('float32')
        embeddings = ae.get_embeddings(X_roll)
        for i in range(EMBEDDINGS_DIM):
            df[f'emb_{i}'] = embeddings[:, i].astype('float32')

        print(f"  ✅ AE: fit على {ae_train_end:,} | transform على {n_ae:,} | emb_0→emb_{EMBEDDINGS_DIM-1}")
    else:
        df['dl_anomaly_score'] = 0.0
        for i in range(EMBEDDINGS_DIM): df[f'emb_{i}'] = 0.0

    # ══════════════════════════════════════════════════════════
    # ANTI-LEAKAGE FIX [mRMR]: feature selection على train split فقط
    # ══════════════════════════════════════════════════════════
    print("\n🔍 mRMR Feature Selection (train-only)...")
    feat_cols_available = [c for c in MODEL_FEATURE_COLS if c in df.columns]
    df_train  = df.iloc[train_idx].copy()
    if df_train.empty:
        df_train = df.iloc[:max(1, min(len(df), split_ctx['split_idx']))].copy()
    target    = df_train.get('bias_label', pd.Series(np.zeros(len(df_train))))

    try:
        if fit_aux_models:
            protected_in_data = [f for f in PROTECTED_FEATURES if f in feat_cols_available] + EMBEDDING_COLS
            filtered = spearman_redundancy_filter(df_train[feat_cols_available], threshold=0.85, target=target, protected=set(protected_in_data))
            if len(feat_cols_available) - len(filtered) > 0: print(f"  Spearman: حذف {len(feat_cols_available) - len(filtered)} feature متكررة")

            selected = mrmr_selection(df_train[filtered], target, n_features=min(40, len(filtered)), protected=set(protected_in_data))
            print(f"  mRMR: {len(feat_cols_available)} → {len(selected)} feature (على {len(df_train):,} train rows)")
        else:
            selected = feat_cols_available
            print("  ✅ mRMR skipped intentionally (fit_aux_models=False)")

        with open(os.path.join(output_dir, 'selected_features.txt'), 'w') as f: f.write('\n'.join(selected))

    except Exception as e:
        print(f"  ⚠️ mRMR skipped: {e}")
        selected = feat_cols_available

    # ══════════════════════════════════════════════════════════
    # ANTI-LEAKAGE FIX [Regime]: fit على train split فقط
    # ══════════════════════════════════════════════════════════
    print("\n🎯 Regime Classification (train-only fit)...")
    try:
        if fit_aux_models:
            train_regime_df = df.iloc[train_idx].copy()
            if train_regime_df.empty:
                train_regime_df = df.iloc[:max(1, min(len(df), split_ctx['split_idx']))].copy()
            regime_clf = RegimeClassifier(n_regimes=4)
            regime_clf.fit(train_regime_df, output_dir=output_dir)
            df['regime_cluster'] = regime_clf.predict(df).astype(np.int8)
            if 'regime_label' not in df.columns:
                df['regime_label'] = df['regime_cluster'].astype(np.int8)
            print(f"  ✅ Regime: fit على {len(train_regime_df):,} | predict على {len(df):,}")
        else:
            df['regime_cluster'] = 0
            if 'regime_label' not in df.columns:
                df['regime_label'] = 0
            print("  ✅ Regime skipped intentionally for evaluation slice")
    except Exception as e:
        df['regime_cluster'] = 0
        if 'regime_label' not in df.columns:
            df['regime_label'] = 0
        print(f"  ⚠️ Regime skipped: {e}")

    # SESSION_LEAK_COLS تُحفظ كـ metadata للتحليل لكن لا تدخل FEATURE_COLS
    session_meta = [c for c in SESSION_LEAK_COLS if c in df.columns]
    meta_cols = ['price', 'size', 'bias_label', 'setup_label', 'conf_label', 'signal_quality',
                 'is_expansion', 'event_flag', 'train_event_flag', 'event_score', 'event_trigger_count',
                 'session', 'liq_score', 'regime_label', 'regime_cluster',
                 'ts_event', 'label_end_ts', 'forward_return', 'label_horizon_steps',
                 'is_train_slice', 'is_holdout_slice', 'is_purged_slice', 'dataset_slice'] + session_meta
    raw_stat_cols = [c for c in RAW_STAT_FEATURE_COLS if c in df.columns]
    out_cols  = [c for c in meta_cols + raw_stat_cols + MODEL_FEATURE_COLS + roll_cols if c in df.columns]
    
    path = os.path.join(output_dir, 'training_features_ready.csv')
    df[out_cols].to_csv(path, index=False)

    print(f"  ✅ CSV: {len(MODEL_FEATURE_COLS)} raw + {len(roll_cols)} rolling + meta")
    return df, path, roll_cols

def _report(df, mbo_p, mbp_p, elapsed, output_dir):
    total = len(df)
    lines = []
    def L(x=''): print(x); lines.append(x)

    L(); L("="*70); L("📊 [REFINERY REPORT V19] — Data Refinery & Embeddings")
    L(f"🕐 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ⏱ {elapsed:.1f}s")
    L("="*70); L(f"  MBO  : {mbo_p}"); L(f"  MBP10: {mbp_p}")
    L(); L("─"*70); L("📊 حالة الـ Features (عينة):"); L("─"*70)
    
    sample_cols = [c for c in MODEL_FEATURE_COLS if c in df.columns and (not c.startswith('emb_') or c == 'emb_0')]
    for col in sample_cols[:25]:
        nz=int((df[col]!=0).sum()); pct=nz/max(total,1)*100
        icon="✅" if nz>0 else "🔴 DEAD"
        L(f"  {icon}  {col:<28} non-zero={nz:>8,} ({pct:5.1f}%)  [{df[col].min():.3f} → {df[col].max():.3f}]")

    if 'emb_0' in df.columns:
        L(f"  🌟  تم تضمين جميع الـ Embeddings بنجاح (emb_0 → emb_{EMBEDDINGS_DIM-1})")

    L(); L("─"*70)
    out_csv=os.path.join(output_dir,'training_features_ready.csv')
    L(f"🚀 python train_v19.py --csv {out_csv} --output {output_dir}")
    L("="*70)

    rep=os.path.join(output_dir,'refinery_report.txt')
    with open(rep,'w',encoding='utf-8') as f: f.write('\n'.join(lines))
    print(f"\n📄 {rep}")


def apply_scaler_params(df: pd.DataFrame, scaler_path: str) -> pd.DataFrame:
    """
    يُطبَّق في الـ Live Trading لتطبيع features بنفس معاملات التدريب.
    
    الاستخدام:
        df_live = apply_scaler_params(df_live, 'outputs/scaler_params.json')
    """
    import json
    if not os.path.exists(scaler_path):
        print(f"  ⚠️ scaler_params.json غير موجود: {scaler_path}")
        return df
    
    with open(scaler_path) as f:
        params = json.load(f)
    
    df = df.copy()
    for col, p in params.items():
        if col not in df.columns:
            df[col] = 0.0
            continue
        s = df[col].astype('float32').fillna(0.0)
        t = p.get('type', 'zero')
        
        if t == 'binary':
            df[col] = s
        elif t == 'robust':
            df[col] = ((s - p['median']) / max(p['iqr'], 1e-8)).clip(-10, 10)
        elif t == 'minmax':
            rng = max(p['max'] - p['min'], 1e-8)
            df[col] = ((s - p['min']) / rng * 2 - 1).clip(-10, 10)
        else:
            df[col] = 0.0
    
    return df


def _require_causal_label_runtime(label_mode: str) -> None:
    requested = str(label_mode or '').strip().lower()
    if requested not in {'v19', 'v22'}:
        return
    if V19_LABELS_AVAILABLE:
        return
    if os.environ.get('QUANTSYSTEM_ALLOW_FALLBACK_SESSION_LABELS', '').strip() == '1':
        print("  ⚠️ Fallback session labeling override enabled via QUANTSYSTEM_ALLOW_FALLBACK_SESSION_LABELS=1")
        return

    detail = f" Import error: {V19_LABELS_IMPORT_ERROR}" if V19_LABELS_IMPORT_ERROR is not None else ''
    raise RuntimeError(
        "❌ Causal v19/v22 labels are unavailable, and the unsafe session-label fallback is disabled by default."
        " Fix the label runtime or set QUANTSYSTEM_ALLOW_FALLBACK_SESSION_LABELS=1 to override."
        f"{detail}"
    )

def run_refinery(
    mbo_path,
    mbp_path,
    symbol='',
    output_dir='outputs',
    chunksize=300_000,
    label_mode='v19',
    n_workers=None,
    target_bars=500,
    label_horizon: int = 150,          # FIX: 50 → 150 (يتوافق مع شمعة 5 دقائق)
    event_roll_window: int = 50,
    direction_threshold_ticks: float = 1.0,  # إعداد هجومي: 2.0 → 1.0
    lob_event_sample: int = LOB_EVENT_SAMPLE_DEFAULT,
    external_scaler_path: str | None = None,
    fit_aux_models: bool = True,
    tp_mult: float = 1.2,
    sl_mult: float = 1.0,
    kalman_slope_threshold: float = 0.05,   # FIX: 1e-5 → 0.05
    trend_strength_min: float = 0.05,
):
    os.makedirs(output_dir, exist_ok=True)
    t0 = datetime.datetime.now()

    print_gpu_report()

    print("="*70)
    print(f"🚀 [REFINERY V19] label={label_mode.upper()} workers={n_workers or 'auto'}")
    print("="*70)
    mode = f"Chunked ({chunksize:,}/دفعة)" if chunksize else "Full Load"
    print(f"  ⚡ {mode}")

    def _read(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.parquet', '.pq', '.snappy'):
            return pd.read_parquet(path)
        elif ext in ('.zst', '.gz'):
            if chunksize: return pd.concat(list(pd.read_csv(path, chunksize=chunksize, low_memory=False, compression='infer')), ignore_index=True)
            return pd.read_csv(path, low_memory=False, compression='infer')
        else:
            if chunksize: return pd.concat(list(pd.read_csv(path, chunksize=chunksize, low_memory=False)), ignore_index=True)
            return pd.read_csv(path, low_memory=False)

    print(f"\n📥 قراءة MBO: {mbo_path}")
    df_mbo = _read(mbo_path)
    print(f"  {len(df_mbo):,} صف")
    if len(df_mbo) == 0: sys.exit(1)

    mbp_exists = os.path.exists(mbp_path) and os.path.getsize(mbp_path) > 0
    df_mbp = None
    if mbp_exists:
        df_mbp = _read(mbp_path)

    print("\n⚙️  Step 1 — MBO (Context & Microstructure)...")
    df_mbo = _normalize_databento_columns(df_mbo)
    deeplob_enabled = _load_deeplob_components() and mbp_exists
    lob_limits = _deeplob_runtime_limits(lob_event_sample=lob_event_sample)
    lob_mbo_src = _lob_source_frame(df_mbo, 'mbo') if deeplob_enabled else None
    lob_mbp_src = None
    df_mbo_p = _process_mbo(df_mbo, n_workers=n_workers)

    _tick = float(AutoCalibrator(200).fit(df_mbo).tick_size) if len(df_mbo) > 0 else 0.0001
    del df_mbo

    if df_mbo_p is None or len(df_mbo_p) == 0:
        print("\n❌ مفيش trades في MBO — تأكد أن الملف يحتوي action=T أو F")
        import sys; sys.exit(1)

    if mbp_exists and df_mbp is not None and len(df_mbp) > 0:
        print("\n⚙️  Step 2 — MBP10 (Liquidity Walls)...")
        df_mbp = _normalize_databento_columns(df_mbp)
        if deeplob_enabled:
            lob_mbp_src = _lob_source_frame(df_mbp, 'mbp')
        df_mbp_p = _process_mbp10(df_mbp, tick_size=_tick)
        del df_mbp

        print("\n⚙️  Step 3 — Merge MBO & MBP...")
        df_merged = _merge(df_mbo_p, df_mbp_p)
        del df_mbo_p, df_mbp_p
    else:
        df_merged = df_mbo_p.copy()
        del df_mbo_p
        for c in ['obi','spoofing_ratio','spoofing_duration','liquidity_trap','liquidity_gaps','dist_to_bid_wall','dist_to_ask_wall']: df_merged[c] = 0.0

    print("\n⚙️  Step 3b — Rolling Context Features...")
    df_merged = _add_rolling_context(df_merged)

    print("\n⚙️  Step 3b-ii — Session Zone Features (metadata only)...")
    df_merged = add_session_features(df_merged, ts_col='ts_event')

    print("\n⚙️  Step 3c — Fractional Differentiation...")
    df_merged, frac_cols = apply_fractional_diff(df_merged, d=0.4)

    print("\n⚙️  Step 3d — Daily/Weekly Levels...")
    df_merged = compute_daily_weekly_levels(df_merged)

    _require_causal_label_runtime(label_mode)

    if label_mode in {'v19', 'v22'} and V19_LABELS_AVAILABLE:
        label_runtime = 'V22' if V19_LABELS_SOURCE == 'modules.labels_v22' else 'V19'
        print(f"\n⚙️  Step 4 — {label_runtime} Causal Event Labels...")
        df_labeled = build_causal_event_labels(
            df_merged,
            horizon=label_horizon,
            event_roll_window=event_roll_window,
            direction_threshold_ticks=direction_threshold_ticks,
            tp_mult=tp_mult,
            sl_mult=sl_mult,
            neutral_mult=0.45,
            tick_size=_tick,
            kalman_slope_threshold=kalman_slope_threshold,
            trend_strength_min=trend_strength_min,
        )
    else:
        print("\n⚙️  Step 4 — Fallback Session Labeling (override enabled)...")
        if V19_LABELS_IMPORT_ERROR is not None:
            print(f"  ⚠️ V19 labels import failed: {V19_LABELS_IMPORT_ERROR}")
        df_labeled = _label_sessions(df_merged)

    # ── V19 Step 3e: LOB Tensor Dataset (event-rich emit positions) ─────
    if deeplob_enabled and lob_mbp_src is not None and len(lob_mbp_src) > 0:
        print("\n⚙️  Step 3e — Building Event-Rich LOB Tensor Dataset (V19)...")
        try:
            lob_path = os.path.join(output_dir, 'lob_tensors.npy')
            ts_path = os.path.join(output_dir, 'lob_tensor_timestamps.npy')
            empty_lob = np.zeros((0, N_TIME_STEPS, N_PRICE_LEVELS, N_CHANNELS), dtype=np.float32)
            empty_ts = np.array([], dtype=np.int64)
            mbo_rows = 0 if lob_mbo_src is None else int(len(lob_mbo_src))
            mbp_rows = 0 if lob_mbp_src is None else int(len(lob_mbp_src))
            total_lob_events = mbo_rows + mbp_rows
            tensor_bytes = max(1, N_TIME_STEPS * N_PRICE_LEVELS * N_CHANNELS * np.dtype(np.float32).itemsize)
            max_tensors_by_bytes = max(0, lob_limits['max_bytes'] // tensor_bytes)
            effective_max_tensors = lob_limits['max_tensors']
            if max_tensors_by_bytes > 0:
                effective_max_tensors = min(effective_max_tensors, max_tensors_by_bytes)
            sample_cap = int(min(lob_limits['lob_event_sample'], max(effective_max_tensors, 0) or lob_limits['lob_event_sample']))
            emit_positions, emit_meta = _select_event_rich_lob_emit_positions(
                df_labeled,
                lob_mbp_src,
                max_events=sample_cap,
            )
            build_plan = {
                'force': lob_limits['force'],
                'max_events': lob_limits['max_events'],
                'max_tensors': lob_limits['max_tensors'],
                'max_tensors_by_bytes': int(max_tensors_by_bytes),
                'effective_max_tensors': int(effective_max_tensors),
                'max_gb': lob_limits['max_gb'],
                'lob_event_sample': int(lob_limits['lob_event_sample']),
                'mbo_rows': mbo_rows,
                'mbp_rows': mbp_rows,
                'total_events': total_lob_events,
                **emit_meta,
            }
            skip_large = (
                (total_lob_events > lob_limits['max_events']) or
                (effective_max_tensors <= 0)
            ) and not lob_limits['force']
            with open(os.path.join(output_dir, 'lob_build_plan.json'), 'w') as f:
                json.dump(build_plan, f, indent=2)

            if skip_large:
                np.save(lob_path, empty_lob)
                np.save(ts_path, empty_ts)
                with open(os.path.join(output_dir, 'lob_build_meta.json'), 'w') as f:
                    json.dump({
                        **build_plan,
                        'status': 'skipped',
                        'built_tensors': 0,
                        'reason': 'auto_skip_large_step3e',
                    }, f, indent=2)
                print(
                    "  ⚠️ Step 3e skipped تلقائيًا: "
                    f"events={total_lob_events:,}, budget_tensors={effective_max_tensors:,}. "
                    "استخدم QUANTSYSTEM_FORCE_LOB=1 لو أردت تشغيله يدويًا."
                )
            elif len(emit_positions) == 0:
                np.save(lob_path, empty_lob)
                np.save(ts_path, empty_ts)
                with open(os.path.join(output_dir, 'lob_build_meta.json'), 'w') as f:
                    json.dump({
                        **build_plan,
                        'status': 'empty_event_sample',
                        'built_tensors': 0,
                        'reason': 'no_event_emit_positions',
                    }, f, indent=2)
                print("  ⚠️ لا توجد event-rich positions كافية لبناء LOB tensors")
            else:
                lob_meta = build_lob_tensor_dataset(
                    lob_mbo_src,
                    lob_mbp_src,
                    time_steps=N_TIME_STEPS,
                    output_path=lob_path,
                    timestamps_path=ts_path,
                    emit_positions=emit_positions,
                )
                lob_meta = {**build_plan, **lob_meta}
                with open(os.path.join(output_dir, 'lob_build_meta.json'), 'w') as f:
                    json.dump(lob_meta, f, indent=2)

                if int(lob_meta.get('built_tensors', 0)) > 0:
                    est_gb = lob_meta.get('estimated_tensor_bytes', 0) / (1024 ** 3)
                    print(
                        "  ✅ LOB Tensors built "
                        f"({lob_meta.get('built_tensors', 0):,} tensors, "
                        f"selected_emit_positions={len(emit_positions):,}, "
                        f"~{est_gb:.2f} GB on disk)"
                    )
                    print(f"  ✅ LOB Tensor file: {lob_path}")
                    print(f"  ✅ LOB Timestamp file: {ts_path}")
                else:
                    print("  ⚠️ LOB Tensors فارغة بعد event sampling — visual embeddings ستعود للصفر")
        except Exception as e:
            np.save(os.path.join(output_dir, 'lob_tensors.npy'), np.zeros((0, N_TIME_STEPS, N_PRICE_LEVELS, N_CHANNELS), dtype=np.float32))
            np.save(os.path.join(output_dir, 'lob_tensor_timestamps.npy'), np.array([], dtype=np.int64))
            with open(os.path.join(output_dir, 'lob_build_meta.json'), 'w') as f:
                json.dump({'status': 'failed', 'built_tensors': 0, 'error': str(e)}, f, indent=2)
            print(f"  ⚠️ LOB Tensor build failed: {e}")
    del lob_mbo_src, lob_mbp_src

    # ⑦ FIX: del df_merged بعد انتهاء كل المسارات
    del df_merged

    print("\n⚙️  Step 5 — Deep Autoencoder + Normalize + Save...")
    df_final, out_path, roll_cols = _normalize_and_save(
        df_labeled,
        output_dir,
        external_scaler_path=external_scaler_path,
        fit_aux_models=fit_aux_models,
    )
    del df_labeled

    elapsed = (datetime.datetime.now()-t0).total_seconds()
    _report(df_final, mbo_path, mbp_path, elapsed, output_dir)
    return df_final


def _load_deeplob_components() -> bool:
    global DEEPLOB_AVAILABLE, LOBTensorBuilder, build_lob_tensor_dataset
    global N_TIME_STEPS, N_PRICE_LEVELS, N_CHANNELS, _DEEPLOB_IMPORT_ATTEMPTED

    if os.environ.get('QUANTSYSTEM_SKIP_HEAVY_ML', '').strip() == '1':
        return False

    if _DEEPLOB_IMPORT_ATTEMPTED:
        return DEEPLOB_AVAILABLE

    _DEEPLOB_IMPORT_ATTEMPTED = True
    try:
        from modules.deeplob_cnn import (
            LOBTensorBuilder as _LOBTensorBuilder,
            build_lob_tensor_dataset as _build_lob_tensor_dataset,
            N_TIME_STEPS as _N_TIME_STEPS,
            N_PRICE_LEVELS as _N_PRICE_LEVELS,
            N_CHANNELS as _N_CHANNELS,
        )
        LOBTensorBuilder = _LOBTensorBuilder
        build_lob_tensor_dataset = _build_lob_tensor_dataset
        N_TIME_STEPS = _N_TIME_STEPS
        N_PRICE_LEVELS = _N_PRICE_LEVELS
        N_CHANNELS = _N_CHANNELS
        DEEPLOB_AVAILABLE = True
    except Exception as exc:
        DEEPLOB_AVAILABLE = False
        print(f"  ⚠️ DeepLOB module غير متاح — {exc}")

    return DEEPLOB_AVAILABLE

if __name__=='__main__':
    p = argparse.ArgumentParser(description='QuantSystem V19 Data Refinery')
    p.add_argument('--mbo',        required=True)
    p.add_argument('--mbp',        required=True)
    p.add_argument('--symbol',     default='')
    p.add_argument('--output',     default='outputs')
    p.add_argument('--chunksize',  type=int, default=300_000)
    p.add_argument('--label_mode', choices=['v19', 'v22'], default='v19')
    p.add_argument('--n_workers',   type=int, default=None)
    p.add_argument('--target_bars', type=int, default=500,
                   help='عدد الـ Volume Bars لكل session (500=swing, 200=scalp, 1000=position)')
    p.add_argument('--label_horizon', type=int, default=150,
                   help='Base forward horizon for causal labels (default: 150)')
    p.add_argument('--event_roll_window', type=int, default=50,
                   help='Rolling window for event filter (default: 50)')
    p.add_argument('--direction_threshold_ticks', type=float, default=1.0,
                   help='Directional threshold floor in ticks (default: 1.0)')
    p.add_argument('--lob_event_sample', type=int, default=LOB_EVENT_SAMPLE_DEFAULT,
                   help='Max event-rich emit positions for LOB tensors')
    p.add_argument('--tp_mult', type=float, default=1.2,
                   help='TP multiplier applied to dynamic threshold (default: 1.2)')
    p.add_argument('--sl_mult', type=float, default=1.0,
                   help='SL multiplier applied to dynamic threshold (default: 1.0)')
    p.add_argument('--kalman_slope_threshold', type=float, default=0.05,
                   help='Kalman slope threshold for trend direction (default: 0.05)')
    p.add_argument('--trend_strength_min', type=float, default=0.05,
                   help='Minimum opposite-trend strength required to veto directional labels (default: 0.05)')
    a  = p.parse_args()
    cs = None if a.chunksize == 0 else a.chunksize
    run_refinery(a.mbo, a.mbp, a.symbol, a.output,
                 chunksize=cs, label_mode=a.label_mode,
                 n_workers=a.n_workers, target_bars=a.target_bars,
                 label_horizon=a.label_horizon,
                 event_roll_window=a.event_roll_window,
                 direction_threshold_ticks=a.direction_threshold_ticks,
                 lob_event_sample=a.lob_event_sample,
                 tp_mult=a.tp_mult,
                 sl_mult=a.sl_mult,
                 kalman_slope_threshold=a.kalman_slope_threshold,
                 trend_strength_min=a.trend_strength_min)
