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
import argparse, datetime, os, sys, multiprocessing, json, inspect, hashlib
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
from modules.regime_classifier       import RegimeClassifier
from modules.regime_classifier       import WassersteinRegimeClassifier  # التعديل 4
from modules.slippage_model          import SlippageModel
from modules.session_features        import add_session_features, SESSION_FEATURE_COLS, SessionVWAPEngine # 🔴 إضافة VWAP
from modules.session_features        import add_cyclical_session_features, CYCLICAL_SESSION_COLS          # التعديل 1
from modules.context_features        import GARCHVolatilityProxy                                         # التعديل 2
from modules.gpu_config              import (detect_gpu, get_multiprocessing_workers,
                                              print_gpu_report, N_WORKERS)
from modules.manifest_v19            import write_manifest
from modules.feature_artifact_v19    import (
    FINAL_FEATURE_DIR,
    load_feature_artifact,
    iter_table_chunks,
    parquet_shard_paths,
    read_table,
    write_checkpoint,
    write_parquet_shards,
    write_table,
)
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
    from modules.labels_v22 import (
        DEFAULT_V22_DIRECTION_THRESHOLD_TICKS,
        DEFAULT_V22_TP_MULT,
        build_causal_event_labels,
    )
    V19_LABELS_AVAILABLE = True
    V19_LABELS_IMPORT_ERROR = None
    V19_LABELS_SOURCE = 'modules.labels_v22'
except ImportError:
    try:
        from modules.labels_v19 import build_causal_event_labels
        DEFAULT_V22_DIRECTION_THRESHOLD_TICKS = 1.5
        DEFAULT_V22_TP_MULT = 1.5
        V19_LABELS_AVAILABLE = True
        V19_LABELS_IMPORT_ERROR = None
        V19_LABELS_SOURCE = 'modules.labels_v19'
    except ImportError as exc:
        DEFAULT_V22_DIRECTION_THRESHOLD_TICKS = 1.5
        DEFAULT_V22_TP_MULT = 1.5
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


def _call_build_causal_event_labels(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Keep label-builder invocation compatible across older v19/v22 module variants.
    Some environments may have a build_causal_event_labels implementation that
    does not yet accept newer keyword arguments like kalman_slope_threshold.
    """
    if build_causal_event_labels is None:
        raise RuntimeError("build_causal_event_labels is not available")

    try:
        sig = inspect.signature(build_causal_event_labels)
    except (TypeError, ValueError):
        sig = None

    if sig is None:
        return build_causal_event_labels(df, **kwargs)

    params = sig.parameters
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kwargs:
        return build_causal_event_labels(df, **kwargs)

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in params}
    dropped = [k for k in kwargs if k not in filtered_kwargs]
    if dropped:
        print(f"  ℹ️ Label builder compatibility: ignoring unsupported args {dropped}")

    return build_causal_event_labels(df, **filtered_kwargs)


def _require_causal_label_runtime(label_mode: str) -> None:
    mode = str(label_mode or '').strip().lower()
    if mode not in {'v19', 'v22'}:
        return
    if V19_LABELS_AVAILABLE:
        return
    if os.environ.get('QUANTSYSTEM_ALLOW_FALLBACK_SESSION_LABELS', '').strip() == '1':
        print("  ⚠️ Fallback session labeling allowed by QUANTSYSTEM_ALLOW_FALLBACK_SESSION_LABELS=1")
        return
    source = V19_LABELS_SOURCE or 'unavailable'
    detail = f'{V19_LABELS_IMPORT_ERROR}' if V19_LABELS_IMPORT_ERROR is not None else 'missing runtime'
    raise RuntimeError(
        f"❌ Causal label runtime required for {mode.upper()} is unavailable "
        f"({source}: {detail}). Set QUANTSYSTEM_ALLOW_FALLBACK_SESSION_LABELS=1 "
        "only إذا كنت تقبل fallback غير سببي لأغراض التشخيص فقط."
    )


def _refinery_dataset_contract(
    df: pd.DataFrame,
    *,
    mbo_path: str,
    mbp_path: str,
    label_mode: str,
    output_dir: str,
    split_meta: dict | None,
    requested_workers: int | None,
    effective_workers: int,
    deeplob_enabled: bool,
    deterministic_stage1: bool,
    merge_tolerance_ms: int,
) -> dict:
    ts = pd.to_datetime(df.get('ts_event', pd.Series(dtype='datetime64[ns]')), utc=True, errors='coerce').dt.tz_localize(None)
    ts = ts.dropna()
    contract_seed = {
        'mbo': os.path.abspath(mbo_path),
        'mbp': os.path.abspath(mbp_path),
        'label_mode': str(label_mode),
        'rows': int(len(df)),
        'ts_min': None if ts.empty else str(ts.min()),
        'ts_max': None if ts.empty else str(ts.max()),
        'split_time': (split_meta or {}).get('split_time'),
    }
    dataset_id = hashlib.sha256(json.dumps(contract_seed, sort_keys=True).encode('utf-8')).hexdigest()[:16]
    return {
        'dataset_id': dataset_id,
        'schema_version': 'v19-event-binary',
        'label_mode': str(label_mode),
        'rows': int(len(df)),
        'ts_min': contract_seed['ts_min'],
        'ts_max': contract_seed['ts_max'],
        'split_meta': split_meta or {},
        'requested_workers': None if requested_workers is None else int(requested_workers),
        'effective_workers': int(effective_workers),
        'effective_worker_mode': 'sequential' if int(effective_workers) <= 1 else 'multiprocess',
        'deterministic_stage1': bool(deterministic_stage1),
        'deeplob_enabled': bool(deeplob_enabled),
        'merge_tolerance_ms': int(merge_tolerance_ms),
    }

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
    # التعديل 1: Cyclical Session (لا leakage)
    'hour_sin', 'hour_cos', 'london_active', 'ny_active', 'overlap_active',
    # التعديل 2: GARCH Volatility
    'garch_vol', 'garch_regime',
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
    # التعديل 1: Cyclical Session محمية دائماً
    'hour_sin', 'hour_cos', 'london_active', 'ny_active', 'overlap_active',
    # التعديل 2: GARCH محمية
    'garch_vol',
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
]  # N = 6

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


def _artifact_phase_dir(output_dir: str, *parts: str) -> str:
    path = os.path.join(output_dir, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def _shard_ts_bounds(df: pd.DataFrame, ts_col: str = 'ts_event') -> tuple[str | None, str | None]:
    if ts_col not in df.columns or len(df) == 0:
        return None, None
    ts = pd.to_datetime(df[ts_col], utc=True, errors='coerce').dt.tz_localize(None).dropna()
    if ts.empty:
        return None, None
    return str(ts.min()), str(ts.max())


def _shard_record(path: str, df: pd.DataFrame, shard_idx: int) -> dict:
    ts_min, ts_max = _shard_ts_bounds(df)
    return {
        'shard_idx': int(shard_idx),
        'path': os.path.abspath(path),
        'rows': int(len(df)),
        'ts_min': ts_min,
        'ts_max': ts_max,
    }


def _read_existing_shard_record(path: str, shard_idx: int) -> dict:
    return _shard_record(path, read_table(path), shard_idx)


def _canonicalize_input_file(
    path: str,
    *,
    kind: str,
    output_dir: str,
    chunk_rows: int,
    resume: bool = False,
) -> list[dict]:
    phase_dir = _artifact_phase_dir(output_dir, 'normalized', kind)
    records: list[dict] = []
    for shard_idx, chunk in enumerate(iter_table_chunks(path, chunk_rows)):
        shard_path = os.path.join(phase_dir, f'{kind}_{shard_idx:05d}.parquet')
        if resume and os.path.exists(shard_path):
            records.append(_read_existing_shard_record(shard_path, shard_idx))
            continue

        normalized = _normalize_databento_columns(chunk)
        if 'ts_event' in normalized.columns:
            normalized['ts_event'] = pd.to_datetime(normalized['ts_event'], utc=True, errors='coerce').dt.tz_localize(None)
            normalized = normalized.dropna(subset=['ts_event']).sort_values('ts_event').reset_index(drop=True)
        write_table(normalized, shard_path, compression='snappy')
        record = _shard_record(shard_path, normalized, shard_idx)
        records.append(record)
        write_checkpoint(
            output_dir,
            f'canonical_{kind}',
            {'kind': kind, 'processed_shards': int(shard_idx + 1), 'last_shard': record},
        )
    return records


def _load_warmup_frame(prev_path: str | None, warmup_rows: int) -> pd.DataFrame:
    if not prev_path or warmup_rows <= 0 or not os.path.exists(prev_path):
        return pd.DataFrame()
    prev_df = read_table(prev_path)
    return prev_df.tail(int(warmup_rows)).copy()


def _run_shard_tasks(tasks, worker_fn, workers: int):
    if not tasks:
        return []
    if workers <= 1:
        return [worker_fn(task) for task in tasks]
    available_methods = set(multiprocessing.get_all_start_methods())
    preferred_method = 'fork' if sys.platform != 'win32' and 'fork' in available_methods else 'spawn'
    ctx_mp = multiprocessing.get_context(preferred_method)
    with ctx_mp.Pool(processes=min(int(workers), len(tasks))) as pool:
        return pool.map(worker_fn, tasks)


def _process_mbo_shard_task(task: dict) -> dict:
    current_record = task['current']
    prev_record = task.get('prev')
    cal_params = task['cal_params']
    warmup_rows = int(task.get('warmup_rows', 0))
    out_path = task['out_path']
    resume = bool(task.get('resume', False))
    shard_idx = int(current_record['shard_idx'])

    if resume and os.path.exists(out_path):
        return _read_existing_shard_record(out_path, shard_idx)

    current_df = read_table(current_record['path'])
    current_df['__emit'] = 1
    current_df['__chunk_id'] = shard_idx
    warmup_df = _load_warmup_frame(prev_record['path'] if prev_record else None, warmup_rows)
    if len(warmup_df):
        warmup_df['__emit'] = 0
        warmup_df['__chunk_id'] = shard_idx
        work_df = pd.concat([warmup_df, current_df], ignore_index=True)
    else:
        work_df = current_df

    processed = _process_mbo_chunk((work_df, cal_params))
    processed = processed[processed.get('__emit', 1) == 1].reset_index(drop=True)
    processed = processed.drop(columns=['__emit'], errors='ignore')
    write_table(processed, out_path, compression='snappy')
    return _shard_record(out_path, processed, shard_idx)


def _process_mbp_shard_task(task: dict) -> dict:
    current_record = task['current']
    prev_record = task.get('prev')
    warmup_rows = int(task.get('warmup_rows', 0))
    tick_size = float(task.get('tick_size', 0.0001))
    out_path = task['out_path']
    resume = bool(task.get('resume', False))
    shard_idx = int(current_record['shard_idx'])

    if resume and os.path.exists(out_path):
        return _read_existing_shard_record(out_path, shard_idx)

    current_df = read_table(current_record['path'])
    current_df['__emit'] = 1
    warmup_df = _load_warmup_frame(prev_record['path'] if prev_record else None, warmup_rows)
    if len(warmup_df):
        warmup_df['__emit'] = 0
        work_df = pd.concat([warmup_df, current_df], ignore_index=True)
    else:
        work_df = current_df

    processed = _process_mbp10(work_df, tick_size=tick_size)
    processed = processed[processed.get('__emit', 1) == 1].reset_index(drop=True)
    processed = processed.drop(columns=['__emit'], errors='ignore')
    write_table(processed, out_path, compression='snappy')
    return _shard_record(out_path, processed, shard_idx)


def _load_records_frame(records: list[dict]) -> pd.DataFrame:
    frames = [read_table(record['path']) for record in sorted(records, key=lambda item: int(item.get('shard_idx', 0)))]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_records_columns(records: list[dict], columns: list[str]) -> pd.DataFrame:
    ordered = sorted(records, key=lambda item: int(item.get('shard_idx', 0)))
    frames: list[pd.DataFrame] = []
    for record in ordered:
        frame = read_table(record['path'])
        keep = [col for col in columns if col in frame.columns]
        if not keep:
            continue
        frames.append(frame.loc[:, keep].copy(deep=False))
    if not frames:
        return pd.DataFrame(columns=columns)
    out = pd.concat(frames, ignore_index=True)
    if 'ts_event' in out.columns:
        out['ts_event'] = pd.to_datetime(out['ts_event'], utc=True, errors='coerce').dt.tz_localize(None)
        out = out.dropna(subset=['ts_event']).sort_values('ts_event').reset_index(drop=True)
    return out


def _load_lob_source_from_records(records: list[dict], kind: str) -> pd.DataFrame:
    ordered = sorted(records, key=lambda item: int(item.get('shard_idx', 0)))
    frames: list[pd.DataFrame] = []
    for record in ordered:
        frame = _lob_source_frame(read_table(record['path']), kind)
        if len(frame):
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if 'ts_event' in out.columns:
        out['ts_event'] = pd.to_datetime(out['ts_event'], utc=True, errors='coerce').dt.tz_localize(None)
        out = out.dropna(subset=['ts_event']).sort_values('ts_event').reset_index(drop=True)
    return out


def _rewrite_records_from_frame(
    df: pd.DataFrame,
    *,
    records: list[dict],
    output_dir: str,
    phase_name: str,
) -> list[dict]:
    out_dir = _artifact_phase_dir(output_dir, 'features', phase_name)
    updated: list[dict] = []
    ordered_records = sorted(records, key=lambda item: int(item.get('shard_idx', 0)))
    start = 0
    for record in ordered_records:
        shard_rows = int(record.get('rows', 0))
        shard_idx = int(record.get('shard_idx', 0))
        shard = df.iloc[start:start + shard_rows].copy()
        start += shard_rows
        shard_path = os.path.join(out_dir, f'{phase_name}_{shard_idx:05d}.parquet')
        write_table(shard, shard_path, compression='snappy')
        updated.append(_shard_record(shard_path, shard, shard_idx))
    return updated


def _rebuild_trade_stateful_features(df: pd.DataFrame, cal_params: dict) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df.copy() if df is not None else pd.DataFrame()

    df = df.copy().sort_values('ts_event').reset_index(drop=True)
    absorb = AbsorptionIntensityEngine(min_price_move=float(cal_params.get('min_price_move', 0.25)))
    ctx = MomentumContextEngine()
    sweep = LiquiditySweepDetector(sweep_threshold=float(cal_params.get('sweep_thresh', 0.05)))
    fisher = FastFisherAlpha()
    fim = FastFIMDetector()
    kyle = KylesLambdaEngine(window=50)
    vnet = VNETEngine()
    tape = FastTapeSpeedTracker()
    mv = MicroVolatilityEngine()
    daily_ctx = DailyContextEngine(
        default_adr=80.0 * float(cal_params.get('tick_size', 0.0001)),
        tick_size=float(cal_params.get('tick_size', 0.0001)),
    )
    vwap_eng = SessionVWAPEngine()

    cvd = 0.0
    rebuilt = {key: [] for key in [
        'cvd', 'absorption_intensity', 'micro_atr', 'volume_burst', 'inter_event_time',
        'fisher_signal', 'anomaly', 'tape_speed', 'cvd_momentum', 'cvd_price_divergence',
        'trend_strength', 'correction_depth', 'liquidity_sweep', 'kyle_lambda', 'vnet',
        'ib_status', 'remaining_fuel', 'fuel_exhausted', 'current_vwap', 'vwap_z_score',
        'vwap_slope', 'session_cvd',
    ]}

    for row in df.itertuples(index=False):
        price = float(getattr(row, 'price', 0.0) or 0.0)
        size = int(getattr(row, 'size', 0) or 0)
        side = str(getattr(row, 'side', '')).strip().upper()
        action = str(getattr(row, 'action', 'T')).strip().upper()
        ts = pd.to_datetime(getattr(row, 'ts_event'))
        ts_ns = int(ts.value)
        is_buy = side in ('B', 'BID')
        if is_buy:
            cvd += size
        elif side in ('A', 'ASK', 'S', 'SELL'):
            cvd -= size

        aii = absorb.update(price, cvd)
        mu_atr, vbi, iet = mv.process_tick(action, price, size, ts_ns)
        tape_speed = tape.update_and_get_speed(ts, action)
        fs = fisher.update_and_get_signal(price, cvd)
        an = fim.detect_stop_hunts(price)
        cvd_mom, cvd_div, t_str, corr_d = ctx.update(price, cvd)
        lsweep = sweep.update(price)
        kyle_val = kyle.update(price, size)
        vnet_val = vnet.update(price, size, side)
        ib_stat, r_fuel, f_exh = daily_ctx.update(ts, price)[:3]
        cur_vwap, v_zscore, v_slope, sess_cvd = vwap_eng.update(ts, price, float(size), is_buy)

        rebuilt['cvd'].append(cvd)
        rebuilt['absorption_intensity'].append(aii)
        rebuilt['micro_atr'].append(mu_atr)
        rebuilt['volume_burst'].append(vbi)
        rebuilt['inter_event_time'].append(iet)
        rebuilt['fisher_signal'].append(fs)
        rebuilt['anomaly'].append(an)
        rebuilt['tape_speed'].append(tape_speed)
        rebuilt['cvd_momentum'].append(cvd_mom)
        rebuilt['cvd_price_divergence'].append(cvd_div)
        rebuilt['trend_strength'].append(t_str)
        rebuilt['correction_depth'].append(corr_d)
        rebuilt['liquidity_sweep'].append(lsweep)
        rebuilt['kyle_lambda'].append(kyle_val)
        rebuilt['vnet'].append(vnet_val)
        rebuilt['ib_status'].append(ib_stat)
        rebuilt['remaining_fuel'].append(r_fuel)
        rebuilt['fuel_exhausted'].append(f_exh)
        rebuilt['current_vwap'].append(cur_vwap)
        rebuilt['vwap_z_score'].append(v_zscore)
        rebuilt['vwap_slope'].append(v_slope)
        rebuilt['session_cvd'].append(sess_cvd)

    for col, values in rebuilt.items():
        if col in {'fisher_signal', 'anomaly', 'fuel_exhausted'}:
            df[col] = np.asarray(values, dtype=np.int8)
        else:
            df[col] = np.asarray(values, dtype=np.float32)
    return df


def _select_relevant_mbp_records(
    records: list[dict],
    *,
    ts_min: pd.Timestamp,
    ts_max: pd.Timestamp,
    tolerance_ms: int,
) -> list[dict]:
    selected = []
    lower = ts_min - pd.Timedelta(milliseconds=max(int(tolerance_ms), 1))
    upper = ts_max
    for record in records:
        rec_min = pd.to_datetime(record.get('ts_min'), errors='coerce')
        rec_max = pd.to_datetime(record.get('ts_max'), errors='coerce')
        if pd.isna(rec_min) or pd.isna(rec_max):
            selected.append(record)
            continue
        if rec_max >= lower and rec_min <= upper:
            selected.append(record)
    return selected


def _merge_mbo_mbp_chunk(
    mbo_df: pd.DataFrame,
    mbp_df: pd.DataFrame,
    *,
    tolerance_ms: int = 500,
) -> pd.DataFrame:
    mbo_df = mbo_df.copy()
    mbo_df = mbo_df.drop(columns=['liquidity_gaps'], errors='ignore')
    mbo_df['ts_event'] = pd.to_datetime(mbo_df['ts_event'], utc=True, errors='coerce').dt.tz_localize(None)
    mbo_df = mbo_df.dropna(subset=['ts_event']).sort_values('ts_event').reset_index(drop=True)
    if len(mbp_df) == 0:
        out = mbo_df.copy()
    else:
        mbp_df = mbp_df.copy()
        mbp_df['ts_event'] = pd.to_datetime(mbp_df['ts_event'], utc=True, errors='coerce').dt.tz_localize(None)
        mbp_df = mbp_df.dropna(subset=['ts_event']).sort_values('ts_event').reset_index(drop=True)
        out = pd.merge_asof(
            mbo_df,
            mbp_df,
            on='ts_event',
            direction='backward',
            tolerance=pd.Timedelta(milliseconds=max(int(tolerance_ms), 1)),
        )
    for c in [
        'obi', 'spoofing_ratio', 'spoofing_duration', 'dist_to_bid_wall', 'dist_to_ask_wall',
        'mid_price', 'micro_price', 'spread',
        'bid_gap_size', 'ask_gap_size', 'bid_wall_strength', 'ask_wall_strength',
        'distance_to_wall', 'gap_size', 'liquidity_density', 'liquidity_gaps',
    ]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = out[c].fillna(0.0)
    return out


def _finalize_merged_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values('ts_event').reset_index(drop=True)
    liq = LiquidityTrapDetector()
    df['liquidity_trap'] = [
        liq.update(float(r.get('price', 0.0) or 0.0), float(r.get('obi', 0.0) or 0.0))
        for _, r in df[['price', 'obi']].fillna(0.0).iterrows()
    ]
    for col in df.select_dtypes(include='float64').columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include='int64').columns:
        df[col] = df[col].astype('int32')
    return df


def _sample_feature_selection_frame(
    df_train: pd.DataFrame,
    *,
    label_col: str = 'bias_label',
    max_rows: int = 200_000,
    seed: int = 42,
) -> pd.DataFrame:
    if df_train is None or len(df_train) <= max_rows:
        return df_train.copy()
    if label_col not in df_train.columns:
        return df_train.sample(n=max_rows, random_state=seed).sort_index()
    sampled = (
        df_train.groupby(label_col, group_keys=False, dropna=False)
        .apply(
            lambda part: part.sample(
                n=max(
                    1,
                    int(round(max_rows * (len(part) / max(len(df_train), 1)))),
                ),
                random_state=seed,
                replace=len(part) < max(
                    1,
                    int(round(max_rows * (len(part) / max(len(df_train), 1)))),
                ),
            )
        )
        .sort_index()
    )
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=seed).sort_index()
    return sampled

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
    daily_ctx  = DailyContextEngine(
        default_adr=80.0 * cal_params.get('tick_size', 0.0001),
        tick_size=cal_params.get('tick_size', 0.0001),
    )
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
        daily_ctx_values = daily_ctx.update(ts, price)
        ib_stat, r_fuel, f_exh = daily_ctx_values[:3]

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
                '__emit': int(getattr(row, '__emit', 1) or 0),
                '__chunk_id': int(getattr(row, '__chunk_id', -1) or -1),
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

def _process_mbo(df_mbo, n_workers: int = None, allow_unsafe_multiprocessing: bool = False):
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
        n_workers = 1

    if n_workers > 1 and not allow_unsafe_multiprocessing:
        print(
            "  🛡️ Deterministic stage1 active — forcing sequential MBO processing "
            f"(requested_workers={n_workers}). "
            "استخدم --allow_unsafe_multiprocessing فقط إذا كنت تقبل حدود chunk-state الحالية."
        )
        n_workers = 1

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
    daily_ctx  = DailyContextEngine(
        default_adr=80.0 * cal_params.get('tick_size', 0.0001),
        tick_size=cal_params.get('tick_size', 0.0001),
    )
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

        daily_ctx_values = daily_ctx.update(ts, price)
        ib_stat, r_fuel, f_exh = daily_ctx_values[:3]

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
                '__emit': int(getattr(row, '__emit', 1) or 0),
                '__chunk_id': int(getattr(row, '__chunk_id', -1) or -1),
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
    if df_mbp is None or len(df_mbp) == 0:
        return pd.DataFrame(columns=[
            'ts_event', 'obi', 'spoofing_ratio', 'spoofing_duration', 'liquidity_gaps',
            'dist_to_bid_wall', 'dist_to_ask_wall', 'mid_price', 'micro_price', 'spread',
            'bid_wall_px', 'ask_wall_px', 'bid_gap_px', 'ask_gap_px', 'bid_gap_size',
            'ask_gap_size', 'bid_wall_strength', 'ask_wall_strength', 'distance_to_wall',
            'gap_size', 'liquidity_density',
        ])

    out = pd.DataFrame(index=df_mbp.index.copy())
    ts_source = (
        df_mbp['ts_event']
        if 'ts_event' in df_mbp.columns
        else df_mbp['timestamp']
        if 'timestamp' in df_mbp.columns
        else pd.Series([pd.NaT] * len(df_mbp), index=df_mbp.index)
    )
    out['ts_event'] = pd.to_datetime(ts_source, errors='coerce').fillna(pd.Timestamp.now())

    bid_px_cols = [f'bid_px_{i:02d}' for i in range(10)]
    ask_px_cols = [f'ask_px_{i:02d}' for i in range(10)]
    bid_sz_cols = [f'bid_sz_{i:02d}' for i in range(10)]
    ask_sz_cols = [f'ask_sz_{i:02d}' for i in range(10)]

    bid_px = df_mbp.reindex(columns=bid_px_cols, fill_value=0.0).astype(np.float64).to_numpy()
    ask_px = df_mbp.reindex(columns=ask_px_cols, fill_value=0.0).astype(np.float64).to_numpy()
    bid_sz = df_mbp.reindex(columns=bid_sz_cols, fill_value=0.0).astype(np.float64).to_numpy()
    ask_sz = df_mbp.reindex(columns=ask_sz_cols, fill_value=0.0).astype(np.float64).to_numpy()

    weights = np.asarray([1.0 / (i + 1) for i in range(10)], dtype=np.float64)
    w_bid = bid_sz @ weights
    w_ask = ask_sz @ weights
    total = np.maximum(w_bid + w_ask, 1e-9)
    out['obi'] = np.clip((w_bid - w_ask) / total, -1.0, 1.0).astype(np.float32)

    bid0 = bid_px[:, 0]
    ask0 = ask_px[:, 0]
    mid = np.where(
        (bid0 > 0) & (ask0 > 0),
        (bid0 + ask0) / 2.0,
        np.where(bid0 > 0, bid0, np.where(ask0 > 0, ask0, 0.0)),
    )
    raw_price_source = (
        df_mbp['price']
        if 'price' in df_mbp.columns
        else pd.Series(np.zeros(len(df_mbp), dtype=np.float64), index=df_mbp.index)
    )
    raw_price = pd.to_numeric(raw_price_source, errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
    price = np.where(raw_price > 0, raw_price, mid)
    spread = np.where((bid0 > 0) & (ask0 > bid0), ask0 - bid0, 0.0)
    micro_price = (bid0 * bid_sz[:, 0] + ask0 * ask_sz[:, 0]) / np.maximum(bid_sz[:, 0] + ask_sz[:, 0], 1e-9)
    micro_price = np.where(np.isfinite(micro_price), micro_price, price)

    out['mid_price'] = mid.astype(np.float32)
    out['micro_price'] = micro_price.astype(np.float32)
    out['spread'] = spread.astype(np.float32)

    action = df_mbp.get('action', pd.Series('', index=df_mbp.index)).astype(str).str.strip().str.upper()
    top_bid_vol = bid_sz[:, :3].sum(axis=1)
    top_ask_vol = ask_sz[:, :3].sum(axis=1)
    curr_max = np.maximum(top_bid_vol, top_ask_vol)
    mean_size = (
        pd.Series(np.where(curr_max > 0, curr_max, np.nan), index=df_mbp.index)
        .rolling(200, min_periods=1)
        .mean()
        .fillna(1.0)
        .clip(lower=1e-9)
    )
    # ── Spoofing Detection (Fixed) ─────────────────────────────────────────
    # BUG FIXED: The old threshold (mean_size * 1.5) used the *total* top-3
    # depth as reference, making it mathematically impossible for a single-
    # level cancel drop to exceed it (single drop ≈ 1/3 of top-3 total).
    # Fix: use per-level (L0) average size as the reference baseline.
    mean_l0 = (
        pd.Series(
            np.where(
                (bid_sz[:, 0] + ask_sz[:, 0]) > 0,
                (bid_sz[:, 0] + ask_sz[:, 0]) / 2.0,
                np.nan,
            ),
            index=df_mbp.index,
        )
        .rolling(200, min_periods=1)
        .mean()
        .bfill()
        .fillna(1.0)
        .clip(lower=1e-9)
    )
    bid_drop = np.maximum(np.r_[top_bid_vol[0], top_bid_vol[:-1]] - top_bid_vol, 0.0)
    ask_drop = np.maximum(np.r_[top_ask_vol[0], top_ask_vol[:-1]] - top_ask_vol, 0.0)
    # Threshold: 1.5× the typical single best-level size (realistic for a cancel)
    drop_thr = mean_l0.to_numpy(dtype=np.float64) * 1.5
    trade_mask = action.isin(TRADE_ACTIONS).to_numpy()
    # Spoof = Cancel action (not a trade) that removes more than 1.5× avg L0 size
    cancel_mask = action.isin(['C']).to_numpy()
    max_drop = np.maximum(bid_drop, ask_drop)
    is_spoof = cancel_mask & (max_drop > drop_thr)
    spoof_count = pd.Series(is_spoof.astype(np.int8), index=df_mbp.index).rolling(50, min_periods=1).sum()
    trade_count = pd.Series(trade_mask.astype(np.int8), index=df_mbp.index).rolling(50, min_periods=1).sum().clip(lower=1.0)
    out['spoofing_ratio'] = np.minimum(spoof_count / trade_count, 1.0).astype(np.float32)
    # spoofing_duration: rolling mean of relative cancel magnitude (sustained signal, not sparse)
    spoof_magnitude = np.where(is_spoof, max_drop / mean_l0.to_numpy(dtype=np.float64), 0.0)
    out['spoofing_duration'] = (
        pd.Series(spoof_magnitude, index=df_mbp.index)
        .rolling(50, min_periods=1)
        .mean()
        .astype(np.float32)
    )

    tick = max(float(tick_size), 1e-8)
    bid_diff = np.abs(np.diff(bid_px, axis=1))
    ask_diff = np.abs(np.diff(ask_px, axis=1))
    bid_gap_hits = (bid_diff > tick).astype(np.float64)
    ask_gap_hits = (ask_diff > tick).astype(np.float64)
    bid_zero_hits = (bid_sz[:, :-1] == 0).astype(np.float64) * 0.5
    ask_zero_hits = (ask_sz[:, :-1] == 0).astype(np.float64) * 0.5
    spread_penalty = (spread > tick * 2.0).astype(np.float64)
    total_checks = float(bid_gap_hits.shape[1] + ask_gap_hits.shape[1] + 1)
    raw_gap_score = (
        bid_gap_hits.sum(axis=1)
        + ask_gap_hits.sum(axis=1)
        + bid_zero_hits.sum(axis=1)
        + ask_zero_hits.sum(axis=1)
        + spread_penalty
    ) / max(total_checks, 1.0)
    out['liquidity_gaps'] = (
        pd.Series(raw_gap_score, index=df_mbp.index)
        .rolling(50, min_periods=1)
        .mean()
        .astype(np.float32)
    )

    row_mean_sz = (
        pd.DataFrame(np.concatenate([bid_sz, ask_sz], axis=1), index=df_mbp.index)
        .replace(0.0, np.nan)
        .mean(axis=1)
        .fillna(0.0)
    )
    mean_sz_hist = row_mean_sz.rolling(200, min_periods=1).mean().fillna(row_mean_sz).clip(lower=1e-9)
    wall_thr = mean_sz_hist.to_numpy(dtype=np.float64) * 3.0

    def _first_level(mask: np.ndarray, px: np.ndarray, sz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        any_hit = mask.any(axis=1)
        first_idx = np.argmax(mask, axis=1)
        first_px = np.where(any_hit, px[np.arange(len(px)), first_idx], np.nan)
        first_sz = np.where(any_hit, sz[np.arange(len(sz)), first_idx], 0.0)
        return first_px, first_sz

    bid_wall_px, bid_wall_sz = _first_level(bid_sz >= wall_thr[:, None], bid_px, bid_sz)
    ask_wall_px, ask_wall_sz = _first_level(ask_sz >= wall_thr[:, None], ask_px, ask_sz)
    bid_wall_strength = np.where(np.isfinite(bid_wall_px), bid_wall_sz / mean_sz_hist.to_numpy(dtype=np.float64), 0.0)
    ask_wall_strength = np.where(np.isfinite(ask_wall_px), ask_wall_sz / mean_sz_hist.to_numpy(dtype=np.float64), 0.0)

    dist_to_bid_wall = np.where(np.isfinite(bid_wall_px), mid - bid_wall_px, 0.0)
    dist_to_ask_wall = np.where(np.isfinite(ask_wall_px), ask_wall_px - mid, 0.0)
    dist_bid_ticks = np.where(np.isfinite(bid_wall_px), (mid - bid_wall_px) / tick, 0.0)
    dist_ask_ticks = np.where(np.isfinite(ask_wall_px), (ask_wall_px - mid) / tick, 0.0)
    valid_distance = np.where(
        (dist_bid_ticks > 0) & (dist_ask_ticks > 0),
        np.minimum(dist_bid_ticks, dist_ask_ticks),
        np.where(dist_bid_ticks > 0, dist_bid_ticks, np.where(dist_ask_ticks > 0, dist_ask_ticks, 0.0)),
    )

    def _detect_soft_gaps(px: np.ndarray, sz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_rows, n_levels = px.shape
        gap_px = np.full(n_rows, np.nan, dtype=np.float64)
        gap_strength = np.zeros(n_rows, dtype=np.float64)
        mean_sz = mean_sz_hist.to_numpy(dtype=np.float64)
        for level in range(max(n_levels - 1, 0)):
            spacing_ticks = np.abs(px[:, level] - px[:, level + 1]) / tick
            cur_sz = sz[:, level]
            nxt_sz = sz[:, level + 1]
            min_pair = np.minimum(cur_sz, nxt_sz)
            avg_pair = (cur_sz + nxt_sz) / 2.0
            scarcity = np.maximum(0.0, 1.0 - (min_pair / np.maximum(mean_sz, 1e-9)))
            void_ratio = np.maximum(0.0, 1.0 - (avg_pair / np.maximum(mean_sz, 1e-9)))
            effective_gap = spacing_ticks + 0.80 * scarcity + 0.60 * void_ratio
            is_gap = (spacing_ticks >= 2.0) | ((spacing_ticks >= 1.0) & (scarcity >= 0.65) & (effective_gap >= 1.60))
            fill_mask = np.isnan(gap_px) & is_gap
            gap_px[fill_mask] = px[fill_mask, level + 1]
            gap_strength[fill_mask] = effective_gap[fill_mask]
        return gap_px, gap_strength

    bid_gap_px, bid_gap_size = _detect_soft_gaps(bid_px, bid_sz)
    ask_gap_px, ask_gap_size = _detect_soft_gaps(ask_px, ask_sz)
    gap_size = np.maximum(bid_gap_size, ask_gap_size)

    top_bid_idx = np.minimum(4, np.maximum(bid_px.shape[1] - 1, 0))
    top_ask_idx = np.minimum(4, np.maximum(ask_px.shape[1] - 1, 0))
    price_range_ticks = np.maximum(
        ((ask_px[:, top_ask_idx] - ask_px[:, 0]) / tick) + ((bid_px[:, 0] - bid_px[:, top_bid_idx]) / tick),
        1.0,
    )
    bid_density = bid_sz[:, :5].sum(axis=1) / price_range_ticks
    ask_density = ask_sz[:, :5].sum(axis=1) / price_range_ticks
    liquidity_density = bid_density + ask_density

    out['dist_to_bid_wall'] = np.where(np.isfinite(bid_wall_px), dist_to_bid_wall, 0.0).astype(np.float32)
    out['dist_to_ask_wall'] = np.where(np.isfinite(ask_wall_px), dist_to_ask_wall, 0.0).astype(np.float32)
    out['bid_wall_px'] = np.where(np.isfinite(bid_wall_px), bid_wall_px, np.nan)
    out['ask_wall_px'] = np.where(np.isfinite(ask_wall_px), ask_wall_px, np.nan)
    out['bid_gap_px'] = np.where(np.isfinite(bid_gap_px), bid_gap_px, np.nan)
    out['ask_gap_px'] = np.where(np.isfinite(ask_gap_px), ask_gap_px, np.nan)
    out['bid_gap_size'] = np.round(bid_gap_size, 2).astype(np.float32)
    out['ask_gap_size'] = np.round(ask_gap_size, 2).astype(np.float32)
    out['bid_wall_strength'] = np.round(bid_wall_strength, 4).astype(np.float32)
    out['ask_wall_strength'] = np.round(ask_wall_strength, 4).astype(np.float32)
    out['distance_to_wall'] = np.round(valid_distance, 2).astype(np.float32)
    out['gap_size'] = np.round(gap_size, 2).astype(np.float32)
    out['liquidity_density'] = np.round(liquidity_density, 4).astype(np.float32)

    if '__emit' in df_mbp.columns:
        out['__emit'] = pd.to_numeric(df_mbp['__emit'], errors='coerce').fillna(1).astype(np.int8).values
    return out

def _merge(df_mbo, df_mbp, tolerance_ms: int = 500):
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
        mbp_ts_col = 'mbp_ts_event'
        df_mbp[mbp_ts_col] = df_mbp['ts_event']
        tolerance = pd.Timedelta(milliseconds=max(int(tolerance_ms), 1))
        df = pd.merge_asof(
            df_mbo,
            df_mbp,
            on='ts_event',
            direction='backward',
            tolerance=tolerance,
        )
        if mbp_ts_col in df.columns:
            matched = df[mbp_ts_col].notna()
            if bool(matched.any()):
                lag_ms = (
                    (df.loc[matched, 'ts_event'] - df.loc[matched, mbp_ts_col])
                    .dt.total_seconds()
                    .mul(1000.0)
                )
                lag_warn_ratio = float((lag_ms > max(tolerance_ms * 0.5, 1.0)).mean()) if len(lag_ms) else 0.0
                if lag_warn_ratio > 0.10:
                    print(
                        "  ⚠️ merge_asof lag warning: "
                        f"{lag_warn_ratio:.1%} من الصفوف المطابقة تستخدم MBP snapshot قديم نسبيًا "
                        f"(tolerance={tolerance_ms}ms)"
                    )
            df = df.drop(columns=[mbp_ts_col], errors='ignore')

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


def _write_empty_lob_artifacts(
    output_dir: str,
    *,
    status: str,
    reason: str,
    build_plan: dict | None = None,
    error: str | None = None,
) -> None:
    lob_path = os.path.join(output_dir, 'lob_tensors.npy')
    ts_path = os.path.join(output_dir, 'lob_tensor_timestamps.npy')
    empty_lob = np.zeros((0, N_TIME_STEPS, N_PRICE_LEVELS, N_CHANNELS), dtype=np.float32)
    empty_ts = np.array([], dtype=np.int64)
    np.save(lob_path, empty_lob)
    np.save(ts_path, empty_ts)
    payload = dict(build_plan or {})
    payload.update({
        'status': str(status),
        'built_tensors': 0,
        'reason': str(reason),
    })
    if error is not None:
        payload['error'] = str(error)
    with open(os.path.join(output_dir, 'lob_build_meta.json'), 'w') as f:
        json.dump(payload, f, indent=2)


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
    max_positions: int | None = None,
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
    base_emit_positions = int(len(emit_positions))
    emit_neighbor_radius = 0
    emit_neighbor_positions_capped = False

    if max_positions is not None and base_emit_positions > 0:
        limit = min(max(int(max_positions), 1), int(len(mbp_df)))
        if base_emit_positions < limit:
            selected_positions = {int(pos) for pos in emit_positions.tolist()}
            radius = 0
            while len(selected_positions) < limit:
                radius += 1
                changed = False
                for pos in emit_positions.tolist():
                    for neighbor in (int(pos) - radius, int(pos) + radius):
                        if 0 <= neighbor < len(mbp_df) and neighbor not in selected_positions:
                            selected_positions.add(int(neighbor))
                            changed = True
                            if len(selected_positions) >= limit:
                                break
                    if len(selected_positions) >= limit:
                        break
                if not changed:
                    break
            emit_neighbor_radius = int(radius)
            emit_neighbor_positions_capped = bool(len(selected_positions) >= limit)
            emit_positions = np.asarray(sorted(selected_positions)[:limit], dtype=np.int32)
            emit_neighbor_radius = max(emit_neighbor_radius, int(len(emit_positions) - base_emit_positions))

    meta = {
        'directional_events_available': int(len(candidates)),
        'event_col': event_col,
        'strong_available': int(len(strong)),
        'weak_available': int(len(weak)),
        'selected_events': int(len(selected)),
        'base_emit_positions': int(base_emit_positions),
        'emit_neighbor_radius': int(emit_neighbor_radius),
        'emit_neighbor_positions_capped': bool(emit_neighbor_positions_capped),
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


def _resolve_regime_mode(regime_mode: str) -> str:
    mode = str(regime_mode or 'rules').strip().lower()
    if mode in {'off', 'none', 'skip'}:
        return 'off'
    if mode in {'wasserstein', 'w'}:
        return 'wasserstein'
    return 'rules'


def _sample_regime_frame(
    df: pd.DataFrame,
    *,
    stride: int,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    if df is None or len(df) == 0:
        return df.iloc[:0].copy(), np.array([], dtype=np.int64), 1

    effective_stride = max(int(stride), 1)
    if effective_stride <= 1 or len(df) <= effective_stride:
        positions = np.arange(len(df), dtype=np.int64)
        return df.copy(), positions, 1

    positions = np.arange(0, len(df), effective_stride, dtype=np.int64)
    if positions[-1] != len(df) - 1:
        positions = np.append(positions, len(df) - 1)
    sampled = df.iloc[positions].copy()
    return sampled, positions, effective_stride


def _expand_sampled_regime_labels(
    n_rows: int,
    sampled_positions: np.ndarray,
    sampled_labels: np.ndarray,
) -> np.ndarray:
    out = np.zeros(max(int(n_rows), 0), dtype=np.int8)
    if n_rows <= 0 or len(sampled_positions) == 0 or len(sampled_labels) == 0:
        return out

    positions = np.asarray(sampled_positions, dtype=np.int64)
    labels = np.asarray(sampled_labels, dtype=np.int8)
    if len(positions) != len(labels):
        raise ValueError(
            f"sampled_positions ({len(positions)}) and sampled_labels ({len(labels)}) must match"
        )

    if int(positions[0]) > 0:
        out[:int(positions[0])] = labels[0]

    for i, pos in enumerate(positions):
        start = max(int(pos), 0)
        stop = int(positions[i + 1]) if i + 1 < len(positions) else int(n_rows)
        out[start:stop] = labels[i]

    return out


def _fit_regime_surface(
    df: pd.DataFrame,
    *,
    train_idx: np.ndarray,
    split_ctx: dict,
    output_dir: str,
    regime_mode: str = 'rules',
    regime_stride: int = 1,
    regime_window: int = 50,
    regime_progress_every: int = 0,
) -> tuple[np.ndarray, dict]:
    mode = _resolve_regime_mode(regime_mode)
    info = {
        'mode': mode,
        'requested_stride': max(int(regime_stride), 1),
        'effective_stride': 1,
        'train_rows': 0,
        'full_rows': int(len(df)),
        'train_sample_rows': 0,
        'full_sample_rows': 0,
        'status': 'skipped',
    }
    if len(df) == 0 or mode == 'off':
        return np.zeros(len(df), dtype=np.int8), info

    train_regime_df = df.iloc[train_idx].copy()
    if train_regime_df.empty:
        fallback_end = max(1, min(len(df), int(split_ctx.get('split_idx', len(df)))))
        train_regime_df = df.iloc[:fallback_end].copy()

    requested_stride = max(int(regime_stride), 1)
    # Small datasets do not benefit from coarse sampling; keep full fidelity.
    if len(df) <= max(5_000, requested_stride * 4):
        requested_stride = 1

    full_sample_df, full_positions, effective_stride = _sample_regime_frame(
        df,
        stride=requested_stride,
    )
    train_sample_df, _, _ = _sample_regime_frame(
        train_regime_df,
        stride=effective_stride,
    )

    info.update({
        'effective_stride': int(effective_stride),
        'train_rows': int(len(train_regime_df)),
        'train_sample_rows': int(len(train_sample_df)),
        'full_sample_rows': int(len(full_sample_df)),
    })

    print(
        "  ℹ️ Regime surface: "
        f"mode={mode} | stride={effective_stride} | "
        f"train_sample={len(train_sample_df):,}/{len(train_regime_df):,} | "
        f"full_sample={len(full_sample_df):,}/{len(df):,}"
    )

    if mode == 'wasserstein':
        clf = WassersteinRegimeClassifier(
            window=max(int(regime_window), 10),
            n_regimes=4,
            progress_every=max(int(regime_progress_every), 0),
        )
        clf.fit(train_sample_df)
        sampled_labels = clf.predict(full_sample_df).astype(np.int8)
    else:
        clf = RegimeClassifier(n_regimes=4, model_type='rules')
        clf.fit(train_sample_df, output_dir=output_dir)
        sampled_labels = clf.predict(full_sample_df).astype(np.int8)

    full_labels = _expand_sampled_regime_labels(len(df), full_positions, sampled_labels)
    info['status'] = 'ok'
    return full_labels, info


def _normalize_and_save(
    df,
    output_dir,
    rolling_window: int = 50,
    external_scaler_path: str | None = None,
    fit_aux_models: bool = True,
    regime_mode: str = 'rules',
    regime_stride: int = 50,
    regime_window: int = 50,
    regime_progress_every: int = 25_000,
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
    df_train_fs = _sample_feature_selection_frame(df_train, label_col='bias_label', max_rows=200_000)
    target    = df_train_fs.get('bias_label', pd.Series(np.zeros(len(df_train_fs))))

    try:
        if fit_aux_models:
            protected_in_data = [f for f in PROTECTED_FEATURES if f in feat_cols_available] + EMBEDDING_COLS
            filtered = spearman_redundancy_filter(df_train_fs[feat_cols_available], threshold=0.85, target=target, protected=set(protected_in_data))
            if len(feat_cols_available) - len(filtered) > 0: print(f"  Spearman: حذف {len(feat_cols_available) - len(filtered)} feature متكررة")

            selected = mrmr_selection(df_train_fs[filtered], target, n_features=min(40, len(filtered)), protected=set(protected_in_data))
            print(
                f"  mRMR: {len(feat_cols_available)} → {len(selected)} feature "
                f"(على sample={len(df_train_fs):,} من train={len(df_train):,})"
            )
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
            selected_mode = _resolve_regime_mode(regime_mode)
            regime_labels, regime_info = _fit_regime_surface(
                df,
                train_idx=train_idx,
                split_ctx=split_ctx,
                output_dir=output_dir,
                regime_mode=selected_mode,
                regime_stride=regime_stride,
                regime_window=regime_window,
                regime_progress_every=regime_progress_every,
            )
            df['regime_cluster'] = regime_labels.astype(np.int8)
            if selected_mode == 'wasserstein':
                df['regime_wasserstein'] = df['regime_cluster'].astype(np.int8)
            if 'regime_label' not in df.columns:
                df['regime_label'] = df['regime_cluster'].astype(np.int8)
            print(
                "  ✅ Regime: "
                f"mode={selected_mode} | "
                f"sampled_train={regime_info.get('train_sample_rows', 0):,} | "
                f"sampled_predict={regime_info.get('full_sample_rows', 0):,} | "
                f"expanded_rows={len(df):,}"
            )
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

    final_dir = _artifact_phase_dir(output_dir, FINAL_FEATURE_DIR)
    final_shards = write_parquet_shards(
        df[out_cols],
        final_dir,
        stem='features',
        rows_per_shard=250_000,
    )
    path = final_dir

    with open(os.path.join(output_dir, 'final_feature_shards.json'), 'w') as f:
        json.dump(final_shards, f, indent=2)

    print(
        "  ✅ Final Parquet: "
        f"{len(final_shards)} shard(s) | "
        f"{len(MODEL_FEATURE_COLS)} raw + {len(roll_cols)} rolling + meta"
    )
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
    out_artifact = os.path.join(output_dir, FINAL_FEATURE_DIR)
    L(f"🚀 python train_v19.py --data {out_artifact} --output {output_dir}")
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

def run_refinery(
    mbo_path,
    mbp_path,
    symbol='',
    output_dir='outputs',
    chunksize=300_000,
    label_mode='v19',
    n_workers=None,
    chunk_rows: int | None = None,
    mbo_workers: int | None = None,
    mbp_workers: int | None = None,
    resume: bool = False,
    target_bars=500,
    label_horizon: int = 150,          # FIX: 50 → 150 (يتوافق مع شمعة 5 دقائق)
    event_roll_window: int = 50,
    direction_threshold_ticks: float = DEFAULT_V22_DIRECTION_THRESHOLD_TICKS,
    causal_threshold_mode: str = 'expanding',
    lob_event_sample: int = LOB_EVENT_SAMPLE_DEFAULT,
    external_scaler_path: str | None = None,
    fit_aux_models: bool = True,
    tp_mult: float = DEFAULT_V22_TP_MULT,
    sl_mult: float = 1.0,
    kalman_slope_threshold: float = 0.05,   # FIX: 1e-5 → 0.05
    trend_strength_min: float = 0.05,
    regime_mode: str = 'rules',
    regime_stride: int = 50,
    regime_window: int = 50,
    regime_progress_every: int = 25_000,
    deterministic_stage1: bool = True,
    allow_unsafe_multiprocessing: bool = False,
    merge_tolerance_ms: int = 500,
    shard_warmup_rows: int = 5_000,
):
    os.makedirs(output_dir, exist_ok=True)
    t0 = datetime.datetime.now()

    print_gpu_report()

    print("="*70)
    print(f"🚀 [REFINERY V19 SHARDED] label={label_mode.upper()} workers={n_workers or 'auto'}")
    print("="*70)
    resolved_chunk_rows = int(chunk_rows or chunksize or 0)
    if resolved_chunk_rows <= 0:
        resolved_chunk_rows = 2_000_000
    requested_workers = int(n_workers if n_workers is not None else get_multiprocessing_workers())
    mbo_workers = int(mbo_workers or requested_workers or 1)
    mbp_workers = int(mbp_workers or requested_workers or 1)
    effective_workers = max(mbo_workers, mbp_workers)
    print(f"  ⚡ Sharded streaming ({resolved_chunk_rows:,}/shard)")
    print(f"  ⚙️  Workers: MBO={mbo_workers} | MBP={mbp_workers}")
    print(f"  ♻️ Resume: {'ON' if resume else 'OFF'} | Warmup rows={int(max(shard_warmup_rows, 0)):,}")
    print(
        "  🧭 Regime: "
        f"mode={_resolve_regime_mode(regime_mode)} | "
        f"stride={max(int(regime_stride), 1)} | "
        f"window={max(int(regime_window), 10)}"
    )

    _require_causal_label_runtime(label_mode)

    mbp_exists = os.path.exists(mbp_path) and os.path.getsize(mbp_path) > 0
    deeplob_enabled = bool(mbp_exists and _load_deeplob_components())
    lob_limits = _deeplob_runtime_limits(lob_event_sample=lob_event_sample)
    if mbp_exists and not deeplob_enabled:
        print("  ⚠️ DeepLOB runtime غير متاح — Step 3e سيُنتج LOB artifacts فارغة")

    print(f"\n📥 Phase A — Canonical Ingest")
    print(f"  MBO: {mbo_path}")
    mbo_records = _canonicalize_input_file(
        mbo_path,
        kind='mbo',
        output_dir=output_dir,
        chunk_rows=resolved_chunk_rows,
        resume=resume,
    )
    if not mbo_records:
        raise RuntimeError("❌ No canonical MBO shards were produced")

    mbp_records: list[dict] = []
    if mbp_exists:
        print(f"  MBP: {mbp_path}")
        mbp_records = _canonicalize_input_file(
            mbp_path,
            kind='mbp',
            output_dir=output_dir,
            chunk_rows=resolved_chunk_rows,
            resume=resume,
        )

    sample_mbo_source = read_table(mbo_records[0]['path'])
    sample_mbo = sample_mbo_source.head(min(5_000, max(100, len(sample_mbo_source))))
    cal = AutoCalibrator(n_ticks=2000).fit(sample_mbo)
    engines = cal.build_engines(include_extended=True)
    cal_params = {
        'tick_size': float(cal.tick_size),
        'min_price_move': float(cal.min_price_move),
        'typical_size': float(cal.typical_size),
        'volatility': float(cal.volatility),
        'sweep_thresh': float(getattr(engines.get('sweep', object()), 'threshold', 0.05)),
    }
    _tick = float(cal.tick_size) if float(cal.tick_size) > 0 else 0.0001

    print("\n⚙️  Phase B — MBP Vectorized Shards...")
    mbp_feature_records: list[dict] = []
    if mbp_records:
        mbp_feature_dir = _artifact_phase_dir(output_dir, 'features', 'mbp')
        mbp_tasks = []
        for idx, record in enumerate(sorted(mbp_records, key=lambda item: int(item['shard_idx']))):
            prev_record = mbp_records[idx - 1] if idx > 0 else None
            mbp_tasks.append({
                'current': record,
                'prev': prev_record,
                'warmup_rows': int(max(shard_warmup_rows, 0)),
                'tick_size': _tick,
                'out_path': os.path.join(mbp_feature_dir, f'mbp_{int(record["shard_idx"]):05d}.parquet'),
                'resume': resume,
            })
        mbp_feature_records = _run_shard_tasks(mbp_tasks, _process_mbp_shard_task, mbp_workers)
        write_checkpoint(
            output_dir,
            'mbp_features',
            {'processed_shards': len(mbp_feature_records), 'rows': int(sum(r['rows'] for r in mbp_feature_records))},
        )

    print("\n⚙️  Phase C — MBO Two-Pass...")
    mbo_pass1_dir = _artifact_phase_dir(output_dir, 'features', 'mbo_pass1')
    mbo_tasks = []
    for idx, record in enumerate(sorted(mbo_records, key=lambda item: int(item['shard_idx']))):
        prev_record = mbo_records[idx - 1] if idx > 0 else None
        mbo_tasks.append({
            'current': record,
            'prev': prev_record,
            'cal_params': cal_params,
            'warmup_rows': int(max(shard_warmup_rows, 0)),
            'out_path': os.path.join(mbo_pass1_dir, f'mbo_{int(record["shard_idx"]):05d}.parquet'),
            'resume': resume,
        })
    mbo_pass1_records = _run_shard_tasks(mbo_tasks, _process_mbo_shard_task, mbo_workers)
    if not mbo_pass1_records:
        raise RuntimeError("❌ مفيش trades في MBO — تأكد أن الملف يحتوي action=T أو F")
    write_checkpoint(
        output_dir,
        'mbo_pass1',
        {'processed_shards': len(mbo_pass1_records), 'rows': int(sum(r['rows'] for r in mbo_pass1_records))},
    )

    mbo_final_dir = _artifact_phase_dir(output_dir, 'features', 'mbo_final')
    reusable_mbo_final_records: list[dict] = []
    if resume:
        ordered_pass1 = sorted(mbo_pass1_records, key=lambda item: int(item['shard_idx']))
        reusable_mbo_final_records = []
        for record in ordered_pass1:
            shard_idx = int(record['shard_idx'])
            final_path = os.path.join(mbo_final_dir, f'mbo_final_{shard_idx:05d}.parquet')
            if not os.path.exists(final_path):
                reusable_mbo_final_records = []
                break
            reusable_mbo_final_records.append(_read_existing_shard_record(final_path, shard_idx))

    if reusable_mbo_final_records:
        print("  ♻️ Reusing existing global trade-driven stateful features (resume)...")
        mbo_final_records = reusable_mbo_final_records
    else:
        print("  🔁 Rebuilding global trade-driven stateful features...")
        mbo_pass1_df = _load_records_frame(mbo_pass1_records)
        mbo_final_df = _rebuild_trade_stateful_features(mbo_pass1_df, cal_params)
        del mbo_pass1_df
        mbo_final_records = _rewrite_records_from_frame(
            mbo_final_df,
            records=mbo_pass1_records,
            output_dir=output_dir,
            phase_name='mbo_final',
        )
        write_checkpoint(
            output_dir,
            'mbo_pass2',
            {'processed_shards': len(mbo_final_records), 'rows': int(len(mbo_final_df))},
        )
        del mbo_final_df

    print("\n⚙️  Phase D — Merge MBO & MBP Shards...")
    merged_records: list[dict] = []
    merged_dir = _artifact_phase_dir(output_dir, 'features', 'merged')
    mbp_cache: dict[str, pd.DataFrame] = {}
    ordered_mbo_final = sorted(mbo_final_records, key=lambda item: int(item['shard_idx']))
    for record in ordered_mbo_final:
        shard_idx = int(record['shard_idx'])
        merged_path = os.path.join(merged_dir, f'merged_{shard_idx:05d}.parquet')
        if resume and os.path.exists(merged_path):
            merged_records.append(_read_existing_shard_record(merged_path, shard_idx))
            continue
        mbo_shard = read_table(record['path'])
        if len(mbo_shard) == 0:
            merged = mbo_shard.copy()
        elif mbp_feature_records:
            ts_min = pd.to_datetime(mbo_shard['ts_event'].min())
            ts_max = pd.to_datetime(mbo_shard['ts_event'].max())
            relevant_records = _select_relevant_mbp_records(
                mbp_feature_records,
                ts_min=ts_min,
                ts_max=ts_max,
                tolerance_ms=merge_tolerance_ms,
            )
            mbp_frames = []
            for mbp_record in relevant_records:
                cache_key = mbp_record['path']
                if cache_key not in mbp_cache:
                    mbp_cache[cache_key] = read_table(cache_key)
                mbp_frames.append(mbp_cache[cache_key])
            mbp_slice = pd.concat(mbp_frames, ignore_index=True) if mbp_frames else pd.DataFrame()
            merged = _merge_mbo_mbp_chunk(mbo_shard, mbp_slice, tolerance_ms=merge_tolerance_ms)
        else:
            merged = mbo_shard.copy()
            for c in ['obi','spoofing_ratio','spoofing_duration','liquidity_trap','liquidity_gaps','dist_to_bid_wall','dist_to_ask_wall']:
                merged[c] = 0.0
        write_table(merged, merged_path, compression='snappy')
        merged_records.append(_shard_record(merged_path, merged, shard_idx))
    write_checkpoint(
        output_dir,
        'merge',
        {'processed_shards': len(merged_records), 'rows': int(sum(r['rows'] for r in merged_records))},
    )
    df_merged = _finalize_merged_frame(_load_records_frame(merged_records))
    gc.collect()

    print("\n⚙️  Step 3b — Rolling Context Features...")
    df_merged = _add_rolling_context(df_merged)

    print("\n⚙️  Step 3b-ii — Session Zone Features (metadata only)...")
    df_merged = add_session_features(df_merged, ts_col='ts_event')

    # ── التعديل 1: Cyclical Session Encoding ─────────────────────
    print("  ✅ Cyclical Session Encoding...")
    df_merged = add_cyclical_session_features(df_merged, ts_col='ts_event')

    # ── التعديل 2: GARCH Volatility Proxy ───────────────────────
    print("  ⏳ GARCH Volatility Proxy...")
    price_col_g = 'price' if 'price' in df_merged.columns else 'close'
    garch_df = GARCHVolatilityProxy.compute_series(df_merged[price_col_g])
    df_merged['garch_vol']    = garch_df['garch_vol'].values
    df_merged['garch_regime'] = garch_df['garch_regime'].values
    print(f"  ✅ GARCH: vol range=[{df_merged['garch_vol'].min():.6f}, {df_merged['garch_vol'].max():.6f}]")

    print("\n⚙️  Step 3c — Fractional Differentiation...")
    df_merged, frac_cols = apply_fractional_diff(df_merged, d=0.4)

    print("\n⚙️  Step 3d — Daily/Weekly Levels...")
    df_merged = compute_daily_weekly_levels(df_merged)

    if label_mode in {'v19', 'v22'} and V19_LABELS_AVAILABLE:
        label_runtime = 'V22' if V19_LABELS_SOURCE == 'modules.labels_v22' else 'V19'
        print(f"\n⚙️  Step 4 — {label_runtime} Causal Event Labels...")
        df_labeled = _call_build_causal_event_labels(
            df_merged,
            horizon=label_horizon,
            event_roll_window=event_roll_window,
            direction_threshold_ticks=direction_threshold_ticks,
            causal_threshold_mode=causal_threshold_mode,
            tp_mult=tp_mult,
            sl_mult=sl_mult,
            neutral_mult=0.45,
            tick_size=_tick,
            kalman_slope_threshold=kalman_slope_threshold,
            trend_strength_min=trend_strength_min,
            n_workers=effective_workers,
        )
    else:
        print("\n⚙️  Step 4 — Fallback Session Labeling...")
        if V19_LABELS_IMPORT_ERROR is not None:
            print(f"  ⚠️ V19 labels import failed: {V19_LABELS_IMPORT_ERROR}")
        df_labeled = _label_sessions(df_merged)

    # ── V19 Step 3e: LOB Tensor Dataset (event-rich emit positions) ─────
    lob_path = os.path.join(output_dir, 'lob_tensors.npy')
    ts_path = os.path.join(output_dir, 'lob_tensor_timestamps.npy')
    build_plan_path = os.path.join(output_dir, 'lob_build_plan.json')
    lob_meta_path = os.path.join(output_dir, 'lob_build_meta.json')
    existing_lob_meta = {}
    if os.path.exists(lob_meta_path):
        try:
            with open(lob_meta_path) as f:
                existing_lob_meta = json.load(f)
        except Exception:
            existing_lob_meta = {}
    reusable_lob_statuses = {'built', 'skipped', 'empty_event_sample'}
    if (
        resume and
        os.path.exists(lob_path) and
        os.path.exists(ts_path) and
        not lob_limits['force'] and
        str(existing_lob_meta.get('status', '')).strip().lower() in reusable_lob_statuses
    ):
        print("\n⚙️  Step 3e — Reusing existing LOB Tensor Dataset (resume)...")
    elif deeplob_enabled and mbp_records:
        print("\n⚙️  Step 3e — Building Event-Rich LOB Tensor Dataset (V19)...")
        try:
            mbo_rows = int(sum(int(record.get('rows', 0)) for record in mbo_final_records))
            mbp_rows = int(sum(int(record.get('rows', 0)) for record in mbp_records))
            total_lob_events = mbo_rows + mbp_rows
            tensor_bytes = max(1, N_TIME_STEPS * N_PRICE_LEVELS * N_CHANNELS * np.dtype(np.float32).itemsize)
            max_tensors_by_bytes = max(0, lob_limits['max_bytes'] // tensor_bytes)
            effective_max_tensors = lob_limits['max_tensors']
            if max_tensors_by_bytes > 0:
                effective_max_tensors = min(effective_max_tensors, max_tensors_by_bytes)
            sample_cap = int(min(lob_limits['lob_event_sample'], max(effective_max_tensors, 0) or lob_limits['lob_event_sample']))
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
            }
            skip_large = (
                (total_lob_events > lob_limits['max_events']) or
                (effective_max_tensors <= 0)
            ) and not lob_limits['force']
            with open(build_plan_path, 'w') as f:
                json.dump(build_plan, f, indent=2)

            if skip_large:
                _write_empty_lob_artifacts(
                    output_dir,
                    status='skipped',
                    reason='auto_skip_large_step3e',
                    build_plan=build_plan,
                )
                print(
                    "  ⚠️ Step 3e skipped تلقائيًا: "
                    f"events={total_lob_events:,}, budget_tensors={effective_max_tensors:,}. "
                    "استخدم QUANTSYSTEM_FORCE_LOB=1 لو أردت تشغيله يدويًا."
                )
            else:
                lob_mbp_index = _load_records_columns(mbp_records, ['ts_event'])
                emit_positions, emit_meta = _select_event_rich_lob_emit_positions(
                    df_labeled,
                    lob_mbp_index,
                    max_events=sample_cap,
                )
                build_plan.update(emit_meta)
                with open(build_plan_path, 'w') as f:
                    json.dump(build_plan, f, indent=2)

                if len(emit_positions) == 0:
                    _write_empty_lob_artifacts(
                        output_dir,
                        status='empty_event_sample',
                        reason='no_event_emit_positions',
                        build_plan=build_plan,
                    )
                    print("  ⚠️ لا توجد event-rich positions كافية لبناء LOB tensors")
                else:
                    lob_mbo_src = _load_lob_source_from_records(mbo_final_records, 'mbo')
                    lob_mbp_src = _load_lob_source_from_records(mbp_records, 'mbp')
                    if len(lob_mbp_src) == 0:
                        _write_empty_lob_artifacts(
                            output_dir,
                            status='empty_mbp_source',
                            reason='mbp_source_unavailable',
                            build_plan=build_plan,
                        )
                        print("  ⚠️ MBP source فارغ بعد canonical ingest — تعذر بناء LOB tensors")
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
                    del lob_mbo_src, lob_mbp_src
        except Exception as e:
            _write_empty_lob_artifacts(
                output_dir,
                status='failed',
                reason='exception',
                build_plan=build_plan if 'build_plan' in locals() else None,
                error=str(e),
            )
            print(f"  ⚠️ LOB Tensor build failed: {e}")
    else:
        reason = 'mbp_missing'
        if mbp_exists and not deeplob_enabled:
            reason = 'deeplob_runtime_unavailable'
        print(f"\n⚙️  Step 3e — Skipped ({reason})...")
        _write_empty_lob_artifacts(
            output_dir,
            status='disabled',
            reason=reason,
            build_plan={
                'force': lob_limits['force'],
                'max_events': lob_limits['max_events'],
                'max_tensors': lob_limits['max_tensors'],
                'lob_event_sample': int(lob_limits['lob_event_sample']),
            },
        )

    # ⑦ FIX: del df_merged بعد انتهاء كل المسارات
    del df_merged

    print("\n⚙️  Step 5 — Deep Autoencoder + Normalize + Save...")
    df_final, out_path, roll_cols = _normalize_and_save(
        df_labeled,
        output_dir,
        external_scaler_path=external_scaler_path,
        fit_aux_models=fit_aux_models,
        regime_mode=regime_mode,
        regime_stride=regime_stride,
        regime_window=regime_window,
        regime_progress_every=regime_progress_every,
    )
    del df_labeled

    split_meta = {}
    split_path = os.path.join(output_dir, 'refinery_split.json')
    if os.path.exists(split_path):
        try:
            with open(split_path) as f:
                split_meta = json.load(f)
        except Exception:
            split_meta = {}
    final_shards = []
    final_shards_path = os.path.join(output_dir, 'final_feature_shards.json')
    if os.path.exists(final_shards_path):
        with open(final_shards_path) as f:
            final_shards = json.load(f)

    contract = _refinery_dataset_contract(
        df_final,
        mbo_path=mbo_path,
        mbp_path=mbp_path,
        label_mode=label_mode,
        output_dir=output_dir,
        split_meta=split_meta,
        requested_workers=requested_workers,
        effective_workers=effective_workers,
        deeplob_enabled=deeplob_enabled,
        deterministic_stage1=deterministic_stage1,
        merge_tolerance_ms=merge_tolerance_ms,
    )
    contract.update({
        'artifact_layout': 'stage1_v2_sharded_parquet',
        'data_format': 'parquet',
        'chunk_rows': int(resolved_chunk_rows),
        'mbo_workers': int(mbo_workers),
        'mbp_workers': int(mbp_workers),
        'resume_enabled': bool(resume),
        'final_feature_shards': final_shards,
        'normalized_shards': {
            'mbo': mbo_records,
            'mbp': mbp_records,
        },
        'feature_shards': {
            'mbp': mbp_feature_records,
            'mbo_pass1': mbo_pass1_records,
            'mbo_final': mbo_final_records,
            'merged': merged_records,
        },
    })
    artifact_manifest_path = write_manifest(
        output_dir=output_dir,
        kind='refinery_v19',
        config={
            'label_mode': label_mode,
            'chunksize': resolved_chunk_rows,
            'requested_workers': requested_workers,
            'effective_workers': effective_workers,
            'deterministic_stage1': bool(deterministic_stage1),
            'allow_unsafe_multiprocessing': bool(allow_unsafe_multiprocessing),
            'resume': bool(resume),
            'chunk_rows': int(resolved_chunk_rows),
            'mbo_workers': int(mbo_workers),
            'mbp_workers': int(mbp_workers),
            'shard_warmup_rows': int(max(shard_warmup_rows, 0)),
            'label_horizon': int(label_horizon),
            'event_roll_window': int(event_roll_window),
            'direction_threshold_ticks': float(direction_threshold_ticks),
            'causal_threshold_mode': str(causal_threshold_mode),
            'tp_mult': float(tp_mult),
            'sl_mult': float(sl_mult),
            'regime_mode': str(_resolve_regime_mode(regime_mode)),
            'regime_stride': int(max(regime_stride, 1)),
            'regime_window': int(max(regime_window, 10)),
            'regime_progress_every': int(max(regime_progress_every, 0)),
            'merge_tolerance_ms': int(merge_tolerance_ms),
        },
        inputs={
            'mbo': os.path.abspath(mbo_path),
            'mbp': os.path.abspath(mbp_path),
        },
        metrics={
            'rows': int(len(df_final)),
            'event_rows': int(pd.to_numeric(df_final.get('event_flag', 0), errors='coerce').fillna(0).astype(np.int8).sum()) if len(df_final) else 0,
        },
        extra=contract,
        filename='artifact_manifest.json',
    )
    print(f"  ✅ Artifact manifest: {artifact_manifest_path}")

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
    except ImportError:
        DEEPLOB_AVAILABLE = False
        print("  ⚠️ DeepLOB module غير متاح")

    return DEEPLOB_AVAILABLE

if __name__=='__main__':
    p = argparse.ArgumentParser(description='QuantSystem V19 Data Refinery')
    p.add_argument('--mbo',        required=True)
    p.add_argument('--mbp',        required=True)
    p.add_argument('--symbol',     default='')
    p.add_argument('--output',     default='outputs')
    p.add_argument('--chunk_rows', '--chunksize', dest='chunk_rows', type=int, default=2_000_000)
    p.add_argument('--label_mode', choices=['v19', 'v22'], default='v19')
    p.add_argument('--n_workers',   type=int, default=None)
    p.add_argument('--mbo_workers', type=int, default=None)
    p.add_argument('--mbp_workers', type=int, default=None)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--shard_warmup_rows', type=int, default=5_000)
    p.add_argument('--target_bars', type=int, default=500,
                   help='عدد الـ Volume Bars لكل session (500=swing, 200=scalp, 1000=position)')
    p.add_argument('--label_horizon', type=int, default=150,
                   help='Base forward horizon for causal labels (default: 150)')
    p.add_argument('--event_roll_window', type=int, default=50,
                   help='Rolling window for event filter (default: 50)')
    p.add_argument('--direction_threshold_ticks', type=float, default=DEFAULT_V22_DIRECTION_THRESHOLD_TICKS,
                   help=f'Directional threshold floor in ticks (default: {DEFAULT_V22_DIRECTION_THRESHOLD_TICKS:.1f})')
    p.add_argument('--causal_threshold_mode', choices=['expanding', 'fixed'], default='expanding',
                   help='threshold mode for train_event_flag selection (default: expanding)')
    p.add_argument('--lob_event_sample', type=int, default=LOB_EVENT_SAMPLE_DEFAULT,
                   help='Max event-rich emit positions for LOB tensors')
    p.add_argument('--tp_mult', type=float, default=DEFAULT_V22_TP_MULT,
                   help=f'TP multiplier applied to dynamic threshold (default: {DEFAULT_V22_TP_MULT:.1f})')
    p.add_argument('--sl_mult', type=float, default=1.0,
                   help='SL multiplier applied to dynamic threshold (default: 1.0)')
    p.add_argument('--kalman_slope_threshold', type=float, default=0.05,
                   help='Kalman slope threshold for trend direction (default: 0.05)')
    p.add_argument('--trend_strength_min', type=float, default=0.05,
                   help='Minimum opposite-trend strength required to veto directional labels (default: 0.05)')
    p.add_argument('--regime_mode', choices=['rules', 'wasserstein', 'off'], default='rules',
                   help='regime surface mode for stage1 metadata (default: rules)')
    p.add_argument('--regime_stride', type=int, default=50,
                   help='sample every N rows before expanding regime back to full rows (default: 50)')
    p.add_argument('--regime_window', type=int, default=50,
                   help='window size for optional Wasserstein regime mode (default: 50)')
    p.add_argument('--regime_progress_every', type=int, default=25_000,
                   help='progress print cadence for Wasserstein rolling loops (default: 25000, 0 disables)')
    p.add_argument('--merge_tolerance_ms', type=int, default=500,
                   help='merge_asof tolerance in milliseconds between MBO and MBP (default: 500)')
    a  = p.parse_args()
    cs = None if a.chunk_rows == 0 else a.chunk_rows
    run_refinery(a.mbo, a.mbp, a.symbol, a.output,
                 chunksize=cs, chunk_rows=cs, label_mode=a.label_mode,
                 n_workers=a.n_workers, target_bars=a.target_bars,
                 mbo_workers=a.mbo_workers, mbp_workers=a.mbp_workers,
                 resume=a.resume, shard_warmup_rows=a.shard_warmup_rows,
                 label_horizon=a.label_horizon,
                 event_roll_window=a.event_roll_window,
                 direction_threshold_ticks=a.direction_threshold_ticks,
                 causal_threshold_mode=a.causal_threshold_mode,
                 lob_event_sample=a.lob_event_sample,
                 tp_mult=a.tp_mult,
                 sl_mult=a.sl_mult,
                 kalman_slope_threshold=a.kalman_slope_threshold,
                 trend_strength_min=a.trend_strength_min,
                 regime_mode=a.regime_mode,
                 regime_stride=a.regime_stride,
                 regime_window=a.regime_window,
                 regime_progress_every=a.regime_progress_every,
                 merge_tolerance_ms=a.merge_tolerance_ms)
