"""
train_v19.py - QuantSystem V19 leakage-safe training foundation
================================================================
V19-alpha focuses on three urgent fixes:
  1. Causal labels generated in prepare_training_data.py --label_mode v19
  2. OOF CatBoost meta-features instead of in-sample stacking
  3. Split-before-sequences for the MetaLearner to avoid train/val overlap

This first V19 slice intentionally prioritizes correctness over architecture
completeness. The DeepLOB visual branch will be reintroduced once fold-aware
OOF visual embeddings are added with timestamp-safe alignment.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prepare_training_data import (
    BINARY_FEATURES,
    CATBOOST_ADVISOR_FEATURES,
    RAW_STAT_FEATURE_COLS,
    RAW_STAT_PREFIX,
    TEMPORAL_DROP_COLS,
)
VISUAL_EMB_DIM = 8
from modules.config_v19 import load_v19_config
from modules.dynamic_labels import (
    DEFAULT_EVENT_OBI_THR,
    DEFAULT_EVENT_ROLL_WINDOW,
    DEFAULT_EVENT_SCORE_THRESHOLD,
    DEFAULT_EVENT_SHIFT_Z_THR,
    DEFAULT_EVENT_VOL_MULT,
    DEFAULT_EVENT_WALL_STR_THR,
)
from modules.feature_factory_v19 import (
    DEFAULT_PASSTHROUGH_COLS,
    apply_scaler_params_to_frame,
    prepare_feature_frame,
)
from modules.gpu_config import detect_gpu
from modules.manifest_v19 import write_manifest
from modules.oof_stacking import (
    align_probability_columns,
    fill_uncovered_one_hot,
    fill_uncovered_probabilities,
    run_sequential_oof,
)
from modules.purging_embargo import embargo_observations, purge_overlapping, walk_forward_expanding
from modules.regime_classifier import RegimeClassifier

try:
    from catboost import CatBoostClassifier, Pool
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False

BIAS_LABELS = {0: 'LONG', 1: 'SHORT', 2: 'NEUTRAL'}
N_CLUSTERS = 4
N_CB_PROBS = 2
SEQ_LEN = 50
SCHEMA_VERSION = 'v19-event-binary'
TRAIN_MODE_EVENT_BINARY = 'event_binary'
PHASE_FULL = 'full'
PHASE_CATBOOST = 'catboost'
PHASE_VISUAL = 'visual'
PHASE_TRAIN = 'train'
STAGE_TO_PHASE = {
    0: PHASE_FULL,
    1: PHASE_CATBOOST,
    2: PHASE_VISUAL,
    3: PHASE_TRAIN,
}
META_FEATURE_NAMES = [
    'cb_prob_long', 'cb_prob_short',
    'cluster_0', 'cluster_1', 'cluster_2', 'cluster_3',
]
VISUAL_FEATURE_NAMES = [f'vis_emb_{i}' for i in range(VISUAL_EMB_DIM)]
SEQUENCE_AUX_LAST_STEP_ONLY = 'last_step_only'
FORBIDDEN_MODEL_INPUT_COLS = {
    'forward_return',
    'label_end_ts',
    'ts_event',
    'bias_label',
    'conf_label',
    'signal_quality',
    'regime_label',
    'regime_cluster',
    'event_flag',
    'train_event_flag',
    'event_score',
    'event_trigger_count',
    'is_expansion',
    'label_horizon_steps',
}
TRAINING_PASSTHROUGH_COLS = [
    col for col in (list(DEFAULT_PASSTHROUGH_COLS) + RAW_STAT_FEATURE_COLS)
    if col in {
        'ts_event',
        'label_end_ts',
        'price',
        'size',
        'bias_label',
        'conf_label',
        'signal_quality',
        'regime_label',
        'regime_cluster',
        'event_flag',
        'train_event_flag',
        'event_score',
        'event_trigger_count',
        'is_expansion',
        'liq_score',
        'forward_return',
        'label_horizon_steps',
        *RAW_STAT_FEATURE_COLS,
    }
]


def _assert_no_forbidden_model_inputs(cols: list[str]) -> None:
    requested = {str(col) for col in cols}
    requested_raw = {col[len(RAW_STAT_PREFIX):] for col in requested if col.startswith(RAW_STAT_PREFIX)}
    leaked = sorted((requested | requested_raw) & FORBIDDEN_MODEL_INPUT_COLS)
    if leaked:
        raise ValueError(
            "❌ Forbidden leakage-prone columns requested for model inputs: "
            f"{leaked}"
        )


def _resolve_phase(stage: int = 0, phase: str | None = None) -> str:
    if phase is not None:
        phase = str(phase).strip().lower()
        aliases = {
            'all': PHASE_FULL,
            'full': PHASE_FULL,
            'catboost': PHASE_CATBOOST,
            'cb': PHASE_CATBOOST,
            'visual': PHASE_VISUAL,
            'deeplob': PHASE_VISUAL,
            'train': PHASE_TRAIN,
            'meta': PHASE_TRAIN,
            'training': PHASE_TRAIN,
        }
        if phase not in aliases:
            raise ValueError(f"Unknown phase: {phase}")
        return aliases[phase]
    return STAGE_TO_PHASE.get(int(stage), PHASE_FULL)


def _phase_banner(phase: str) -> str:
    return {
        PHASE_FULL: 'full pipeline',
        PHASE_CATBOOST: 'catboost-only',
        PHASE_VISUAL: 'visual-only',
        PHASE_TRAIN: 'train-only',
    }.get(phase, phase)


def _load_meta_learner_class():
    # التعديل 3: استخدام MetaLearnerTCNLSTM (TCN + LSTM)
    try:
        from modules.meta_learner import MetaLearnerTCNLSTM
        return MetaLearnerTCNLSTM
    except ImportError:
        from modules.meta_learner import MetaLearnerLSTM
        return MetaLearnerLSTM


def _load_deeplob_runtime():
    try:
        from modules.deeplob_cnn import DeepLOBCNN

        return DeepLOBCNN, True
    except ImportError:
        return None, False


def _resolve_catboost_device(catboost_device: str = 'auto') -> tuple[str, str | None]:
    mode = str(catboost_device or 'auto').strip().lower()
    env_mode = os.environ.get('QUANTSYSTEM_CATBOOST_DEVICE', '').strip().lower()
    if mode == 'auto' and env_mode in {'cpu', 'gpu'}:
        mode = env_mode

    if mode == 'gpu':
        return 'GPU', '0'
    if mode == 'cpu':
        return 'CPU', None

    gpu_info = detect_gpu()
    return ('GPU', '0') if gpu_info.get('available') else ('CPU', None)


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    protected = {
        'bias_label', 'setup_label', 'conf_label', 'signal_quality',
        'regime_label', 'regime_cluster', 'event_flag', 'is_expansion',
        'ts_event', 'label_end_ts', 'forward_return', 'label_horizon_steps', 'liq_score',
    }
    dropped = [c for c in TEMPORAL_DROP_COLS if c in df.columns and c not in protected]
    if dropped:
        print(f"  🛡️ Anti-Leakage Drop: {dropped}")
    return df.drop(columns=dropped, errors='ignore')

def handle_rare_classes(df, label_col="bias_label", min_samples=10, strategy="auto"):
    """
    min_samples: الحد الأدنى المقبول لكل class
    strategy:
        - "drop": حذف الفئات النادرة
        - "merge": دمجها مع أقرب class
        - "auto": يقرر تلقائيًا
    """
    counts = df[label_col].value_counts()
    rare_classes = counts[counts < min_samples].index.tolist()

    if not rare_classes:
        return df

    print(f"⚠️ Classes قليلة: {rare_classes} → counts={counts.to_dict()}")

    if strategy == "drop" or (strategy == "auto" and len(counts) > 2):
        print("🗑️ حذف الفئات النادرة...")
        df = df[~df[label_col].isin(rare_classes)].copy()

    elif strategy == "merge" or (strategy == "auto" and len(counts) <= 2):
        print("🔀 دمج الفئات النادرة مع الأكثر شيوعًا...")
        majority_class = counts.idxmax()
        df[label_col] = df[label_col].apply(
            lambda x: majority_class if x in rare_classes else x
        )

    return df

def load_training_csv(csv_path: str) -> pd.DataFrame:
    print(f"\n📥 قراءة: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    for col in ('ts_event', 'label_end_ts'):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce').dt.tz_localize(None)

    if 'bias_label' not in df.columns:
        raise ValueError("❌ 'bias_label' غير موجود — شغّل prepare_training_data.py --label_mode v19 أولاً")
    if 'ts_event' not in df.columns:
        raise ValueError("❌ 'ts_event' غير موجود — V19 يحتاج timestamps محفوظة في CSV")

    df = _sanitize_df(df)
    raw_cols = [f'{RAW_STAT_PREFIX}{col}' for col in CATBOOST_ADVISOR_FEATURES if f'{RAW_STAT_PREFIX}{col}' in df.columns]
    if len(raw_cols) < len(CATBOOST_ADVISOR_FEATURES):
        missing = [col for col in CATBOOST_ADVISOR_FEATURES if f'{RAW_STAT_PREFIX}{col}' not in df.columns]
        print(
            "  ⚠️ Missing raw stat columns for fold-clean scaling: "
            f"{missing}. Re-run prepare_training_data.py to unlock the strict anti-leakage path."
        )
    df = prepare_feature_frame(
        df,
        stat_features=CATBOOST_ADVISOR_FEATURES + raw_cols,
        scaler_params=None,
        already_scaled=True,
        passthrough_cols=TRAINING_PASSTHROUGH_COLS,
        timestamp_cols=('ts_event', 'label_end_ts'),
    )
    print(f"  Shape: {df.shape}")
    print(f"  Labels: {df['bias_label'].value_counts().to_dict()}")
    return df


def build_event_training_view(
    df: pd.DataFrame,
    mode: str = TRAIN_MODE_EVENT_BINARY,
    quality_weight_strong: float = 2.0,
    quality_weight_weak: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    if mode != TRAIN_MODE_EVENT_BINARY:
        raise ValueError(f'Unsupported training mode: {mode}')

    out = df.copy()
    if 'ts_event' in out.columns:
        out = out.sort_values('ts_event').reset_index(drop=True)
    out['event_flag'] = pd.to_numeric(out.get('event_flag', 0), errors='coerce').fillna(0).astype(np.int8)
    out['train_event_flag'] = pd.to_numeric(out.get('train_event_flag', out['event_flag']), errors='coerce').fillna(0).astype(np.int8)
    out['bias_label'] = pd.to_numeric(out.get('bias_label', 2), errors='coerce').fillna(2).astype(np.int8)
    out['signal_quality'] = pd.to_numeric(out.get('signal_quality', 0), errors='coerce').fillna(0).astype(np.int8)

    directional_mask = out['bias_label'].isin([0, 1])
    event_col = 'train_event_flag' if 'train_event_flag' in out.columns else 'event_flag'
    event_mask = (out[event_col] == 1) & directional_mask

    fallback_reason = None
    if not event_mask.any() and event_col != 'event_flag':
        fallback_mask = (out['event_flag'] == 1) & directional_mask
        if fallback_mask.any():
            event_col = 'event_flag'
            event_mask = fallback_mask
            fallback_reason = 'fallback_to_event_flag'

    if not event_mask.any() and directional_mask.any():
        event_col = 'bias_label'
        event_mask = directional_mask
        fallback_reason = 'fallback_to_directional_rows'
        print(
            "  ⚠️ Event Training View fallback: no directional rows via event flags; "
            "using directional bias rows directly."
        )

    event_df = out.loc[event_mask].copy().reset_index(drop=True)
    if event_df.empty:
        raise RuntimeError('❌ لا توجد directional event rows صالحة للتدريب بعد تطبيق event view')

    event_df['quality_sample_weight'] = np.where(
        event_df['signal_quality'].values.astype(np.int8) == 2,
        float(quality_weight_strong),
        float(quality_weight_weak),
    ).astype(np.float32)
    event_df['conf_target'] = (event_df['signal_quality'].values.astype(np.int8) == 2).astype(np.float32)
    event_df['event_seq_idx'] = np.arange(len(event_df), dtype=np.int32)

    info = {
        'mode': mode,
        'event_col': event_col,
        'rows_full': int(len(out)),
        'rows_event_directional': int(len(event_df)),
        'event_rate_full': float(event_mask.mean()),
        'raw_event_rate_full': float(out['event_flag'].mean()),
        'fallback_reason': fallback_reason,
        'quality_weight_strong': float(quality_weight_strong),
        'quality_weight_weak': float(quality_weight_weak),
        'bias_counts': {str(k): int(v) for k, v in event_df['bias_label'].value_counts().to_dict().items()},
        'quality_counts': {str(k): int(v) for k, v in event_df['signal_quality'].value_counts().to_dict().items()},
    }
    print(
        "  ✅ Event Training View: "
        f"{info['rows_event_directional']:,}/{info['rows_full']:,} rows "
        f"({info['event_rate_full']:.1%}) via {event_col} "
        f"| raw_event={info['raw_event_rate_full']:.1%} "
        f"| bias={info['bias_counts']} | quality={info['quality_counts']}"
    )
    return event_df, info


def _raw_feature_name(col: str) -> str:
    return f'{RAW_STAT_PREFIX}{col}'


def _raw_stat_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # Hard guard: passthrough/meta columns such as forward_return may exist in
    # the CSV for analysis/backtesting, but they must never enter model inputs.
    _assert_no_forbidden_model_inputs(list(cols))
    data = {}
    for col in cols:
        raw_col = _raw_feature_name(col)
        src = raw_col if raw_col in df.columns else col
        if src in df.columns:
            series = pd.to_numeric(df[src], errors='coerce').fillna(0.0).astype(np.float32)
        else:
            series = pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)
        data[col] = series
    return pd.DataFrame(data, index=df.index)


def _fit_scaler_params_from_frame(frame: pd.DataFrame) -> dict:
    params = {}
    for col in frame.columns:
        s = pd.to_numeric(frame[col], errors='coerce').fillna(0.0).astype(np.float32)
        if col in BINARY_FEATURES:
            params[col] = {'type': 'binary'}
            continue

        median_ = float(s.median())
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        if iqr > 1e-8:
            params[col] = {'type': 'robust', 'median': median_, 'iqr': iqr}
            continue

        smin = float(s.min())
        smax = float(s.max())
        rng = smax - smin
        if rng > 1e-8:
            params[col] = {'type': 'minmax', 'min': smin, 'max': smax}
        else:
            params[col] = {'type': 'zero'}
    return params


def _apply_scaler_to_stat_frame(frame: pd.DataFrame, scaler_params: dict) -> pd.DataFrame:
    scaled = apply_scaler_params_to_frame(frame, scaler_params or {})
    return scaled[frame.columns].astype(np.float32)


def _build_scaled_stat_matrix(df: pd.DataFrame, cols: list[str], scaler_params: dict) -> np.ndarray:
    raw_frame = _raw_stat_frame(df, cols)
    scaled = _apply_scaler_to_stat_frame(raw_frame, scaler_params)
    return scaled[cols].values.astype(np.float32)


def _project_sequence_aux_context(
    window: np.ndarray,
    n_stat_feat: int,
    sequence_aux_mode: str = SEQUENCE_AUX_LAST_STEP_ONLY,
) -> np.ndarray:
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f'Expected 2D sequence window, got {arr.shape}')
    if sequence_aux_mode != SEQUENCE_AUX_LAST_STEP_ONLY or arr.shape[1] <= int(n_stat_feat):
        return arr.astype(np.float32, copy=True)

    out = arr.astype(np.float32, copy=True)
    out[:-1, int(n_stat_feat):] = 0.0
    return out


def _quality_sample_weights(df: pd.DataFrame, strong_weight: float = 2.0, weak_weight: float = 1.0) -> np.ndarray:
    quality = pd.to_numeric(df.get('signal_quality', 1), errors='coerce').fillna(1).astype(np.int32).values
    return np.where(quality == 2, float(strong_weight), float(weak_weight)).astype(np.float32)


def _time_series(df: pd.DataFrame, col: str, fallback: str | None = None) -> pd.Series:
    if col in df.columns:
        s = pd.to_datetime(df[col], utc=True, errors='coerce').dt.tz_localize(None)
    elif fallback and fallback in df.columns:
        s = pd.to_datetime(df[fallback], utc=True, errors='coerce').dt.tz_localize(None)
    else:
        s = pd.Series(pd.date_range('2026-01-01', periods=len(df), freq='s'))

    s = s.ffill()
    if s.isna().any():
        first_valid = s.first_valid_index()
        if first_valid is not None:
            first_pos = s.index.get_loc(first_valid)
            first_ts = s.iloc[first_pos]
            for pos in range(first_pos - 1, -1, -1):
                s.iloc[pos] = first_ts - pd.Timedelta(seconds=(first_pos - pos))
    if s.isna().any():
        base = pd.Timestamp('2026-01-01')
        s = pd.Series([base + pd.Timedelta(seconds=i) for i in range(len(df))])
    return s


def _sequence_split_context(
    df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    train_frac: float = 0.80,
) -> dict:
    n = len(df)
    split_idx = max(seq_len * 2, int(n * train_frac))
    split_idx = min(max(split_idx, seq_len), n)
    split_time = _time_series(df, 'ts_event').iloc[min(split_idx, n - 1)]
    label_end = _time_series(df, 'label_end_ts', fallback='ts_event')
    train_row_ok = (np.arange(n) < split_idx) & (label_end < split_time)
    val_row_ok = np.arange(n) >= split_idx
    return {
        'split_idx': int(split_idx),
        'split_time': split_time,
        'label_end': label_end,
        'train_row_ok': train_row_ok,
        'val_row_ok': val_row_ok,
    }


def build_inference_scaler_params(
    df: pd.DataFrame,
    cols: list[str],
    train_frac: float = 0.80,
) -> tuple[dict, dict]:
    split_ctx = _sequence_split_context(df, seq_len=SEQ_LEN, train_frac=train_frac)
    raw_frame = _raw_stat_frame(df, cols)
    train_mask = split_ctx['train_row_ok']
    if not np.any(train_mask):
        train_mask = np.arange(len(df)) < split_ctx['split_idx']
    train_frame = raw_frame.loc[train_mask]
    if train_frame.empty:
        train_frame = raw_frame.iloc[:max(1, split_ctx['split_idx'])]
    scaler_params = _fit_scaler_params_from_frame(train_frame)
    info = {
        'split_idx': int(split_ctx['split_idx']),
        'split_time': str(split_ctx['split_time']),
        'scaler_train_rows': int(len(train_frame)),
    }
    return scaler_params, info


def _save_scaler_params(output_dir: str, scaler_params: dict) -> str:
    path = os.path.join(output_dir, 'scaler_params.json')
    with open(path, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    return path


def copy_inference_artifacts(csv_path: str, output_dir: str) -> dict:
    copied = {}
    src_dir = os.path.dirname(os.path.abspath(csv_path))
    for name in ('selected_features.txt', 'refinery_report.txt', 'lob_tensor_timestamps.npy'):
        src = os.path.join(src_dir, name)
        dst = os.path.join(output_dir, name)
        if os.path.exists(src):
            if os.path.abspath(src) == os.path.abspath(dst):
                copied[name] = dst
                continue
            shutil.copy2(src, dst)
            copied[name] = dst
    return copied


def build_time_splits(
    df: pd.DataFrame,
    n_folds: int = 6,
    test_size: float = 0.10,
    embargo_pct: float = 0.02,
    min_train_pct: float = 0.20,
):
    n = len(df)
    t0 = _time_series(df, 'ts_event')
    t1 = _time_series(df, 'label_end_ts', fallback='ts_event')
    splits = list(
        walk_forward_expanding(
            n,
            n_folds=n_folds,
            test_size=test_size,
            embargo_pct=embargo_pct,
            t0=t0,
            t1=t1,
            min_train_pct=min_train_pct,
        )
    )
    if not splits:
        raise RuntimeError('❌ تعذر بناء time splits صالحة لـ V19')
    return splits, t0, t1


def _build_inner_time_split(
    train_idx: np.ndarray,
    t0: pd.Series,
    t1: pd.Series,
    embargo_pct: float,
    val_frac: float = 0.15,
    min_val_rows: int = 50,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if len(train_idx) < max(100, min_val_rows * 2):
        return None, None

    val_n = max(min_val_rows, int(len(train_idx) * val_frac))
    val_n = min(val_n, len(train_idx) - 50)
    if val_n < min_val_rows:
        return None, None

    inner_val = train_idx[-val_n:]
    inner_train_raw = train_idx[:-val_n]
    if len(inner_train_raw) < 50:
        return None, None

    inner_train = purge_overlapping(inner_train_raw, inner_val, t1, t0)
    inner_train = embargo_observations(inner_train, inner_val, embargo_pct)
    if len(inner_train) < 50 or len(inner_val) < 20:
        return None, None
    return inner_train, inner_val


def stage1_oof_meta(
    df: pd.DataFrame,
    output_dir: str,
    splits=None,
    n_folds: int = 6,
    test_size: float = 0.10,
    embargo_pct: float = 0.02,
    min_train_pct: float = 0.20,
    t0: pd.Series | None = None,
    t1: pd.Series | None = None,
    inference_scaler_params: dict | None = None,
    catboost_device: str = 'auto',
    quality_weight_strong: float = 2.0,
    quality_weight_weak: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    print("\n" + "═" * 65)
    print("🐱 STAGE 1 — V19 OOF CatBoost + Regime Meta-Features")
    print("═" * 65)
    if not inference_scaler_params:
        raise RuntimeError(
            '❌ inference_scaler_params is required for stage1_oof_meta. '
            'Refusing to fall back to a full-data scaler.'
        )

    n = len(df)
    raw_stat = _raw_stat_frame(df, CATBOOST_ADVISOR_FEATURES)
    y = df['bias_label'].fillna(1).astype(np.int32).values
    if splits is None:
        splits, t0, t1 = build_time_splits(
            df,
            n_folds=n_folds,
            test_size=test_size,
            embargo_pct=embargo_pct,
            min_train_pct=min_train_pct,
        )
    elif t0 is None or t1 is None:
        _, t0, t1 = build_time_splits(
            df,
            n_folds=n_folds,
            test_size=test_size,
            embargo_pct=embargo_pct,
            min_train_pct=min_train_pct,
        )

    print(f"  Splits: {len(splits)} | Rows: {n:,}")
    cb_task_type, cb_devices = _resolve_catboost_device(catboost_device)
    print(f"  CatBoost device: {cb_task_type}")

    priors = np.bincount(y, minlength=N_CB_PROBS).astype(np.float32)
    priors = priors / max(priors.sum(), 1.0)

    if not CB_AVAILABLE:
        raise RuntimeError(
            "❌ CatBoost غير مثبّت. هذه المرحلة لم تتدرب فعليًا.\n"
            "ثبّت الحزمة داخل البيئة الحالية ثم أعد التشغيل:\n"
            "pip install catboost"
        )

    regime_tmp_dir = tempfile.mkdtemp(prefix='_oof_regime_tmp_', dir=output_dir)
    os.makedirs(regime_tmp_dir, exist_ok=True)

    def _cb_predict(train_idx, test_idx, fold_no):
        inner_train, inner_val = _build_inner_time_split(train_idx, t0, t1, embargo_pct)
        fit_idx = inner_train if inner_train is not None else train_idx
        fold_scaler = _fit_scaler_params_from_frame(raw_stat.iloc[fit_idx])
        X_fit = _apply_scaler_to_stat_frame(raw_stat.iloc[fit_idx], fold_scaler).values.astype(np.float32)
        X_test = _apply_scaler_to_stat_frame(raw_stat.iloc[test_idx], fold_scaler).values.astype(np.float32)

        sw = _quality_sample_weights(
            df.iloc[fit_idx],
            strong_weight=quality_weight_strong,
            weak_weight=quality_weight_weak,
        )
        model = CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            loss_function='Logloss',
            eval_metric='Logloss',
            early_stopping_rounds=50 if inner_val is not None else None,
            use_best_model=inner_val is not None,
            verbose=0,
            random_seed=42 + fold_no,
            task_type=cb_task_type,
            devices=cb_devices,
        )
        tr_pool = Pool(X_fit, y[fit_idx], weight=sw, feature_names=CATBOOST_ADVISOR_FEATURES)
        eval_set = None
        if inner_val is not None:
            X_val = _apply_scaler_to_stat_frame(raw_stat.iloc[inner_val], fold_scaler).values.astype(np.float32)
            eval_set = Pool(X_val, y[inner_val], feature_names=CATBOOST_ADVISOR_FEATURES)
        model.fit(tr_pool, eval_set=eval_set, plot=False)
        present_classes = getattr(model, 'classes_', np.unique(y[fit_idx]))
        preds = align_probability_columns(
            model.predict_proba(X_test),
            N_CB_PROBS,
            classes=present_classes,
        )
        pred_labels = np.argmax(preds, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y[test_idx],
            pred_labels,
            labels=[0, 1],
            average='macro',
            zero_division=0,
        )
        return preds, {
            'directional_precision': float(precision),
            'directional_recall': float(recall),
            'directional_f1': float(f1),
        }

    oof_probs_raw, prob_covered, prob_reports = run_sequential_oof(
        n, N_CB_PROBS, splits, _cb_predict
    )
    oof_probs = fill_uncovered_probabilities(oof_probs_raw, prob_covered, priors=priors)

    def _regime_predict(train_idx, test_idx, fold_no):
        clf = RegimeClassifier(n_regimes=N_CLUSTERS)
        clf.fit(df.iloc[train_idx].copy(), output_dir=regime_tmp_dir)
        labels = clf.predict(df.iloc[test_idx].copy())
        out = np.zeros((len(test_idx), N_CLUSTERS), dtype=np.float32)
        for i, label in enumerate(labels):
            if 0 <= int(label) < N_CLUSTERS:
                out[i, int(label)] = 1.0
            else:
                out[i, 0] = 1.0
        return out, {'cluster_counts': np.bincount(labels.astype(np.int32), minlength=N_CLUSTERS).tolist()}

    regime_raw, regime_covered, regime_reports = run_sequential_oof(
        n, N_CLUSTERS, splits, _regime_predict
    )
    regime_oh = fill_uncovered_one_hot(regime_raw, regime_covered, default_class=0)
    coverage = prob_covered & regime_covered

    fold_metrics = {
        'catboost_folds': prob_reports,
        'regime_folds': regime_reports,
        'coverage_ratio': float(coverage.mean()),
    }
    with open(os.path.join(output_dir, 'stage1_v19_metrics.json'), 'w') as f:
        json.dump(fold_metrics, f, indent=2)

    if not inference_scaler_params:
        raise RuntimeError(
            '❌ inference_scaler_params is required for the final CatBoost fit. '
            'Refusing to fall back to a full-data scaler.'
        )
    final_scaler = inference_scaler_params
    X_final = _apply_scaler_to_stat_frame(raw_stat, final_scaler).values.astype(np.float32)
    final_sw = _quality_sample_weights(
        df,
        strong_weight=quality_weight_strong,
        weak_weight=quality_weight_weak,
    )
    final_model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        loss_function='Logloss',
        eval_metric='Logloss',
        early_stopping_rounds=50,
        use_best_model=False,
        verbose=50,
        random_seed=42,
        task_type=cb_task_type,
        devices=cb_devices,
    )
    final_model.fit(Pool(X_final, y, weight=final_sw, feature_names=CATBOOST_ADVISOR_FEATURES), plot=False)
    final_model.save_model(os.path.join(output_dir, 'catboost_advisor_v19.cbm'))
    final_cb_classes = np.asarray(getattr(final_model, 'classes_', np.unique(y)), dtype=np.int32).tolist()
    with open(os.path.join(output_dir, 'catboost_classes_v19.json'), 'w') as f:
        json.dump({'classes': final_cb_classes}, f, indent=2)

    final_regime = RegimeClassifier(n_regimes=N_CLUSTERS)
    final_regime.fit(df.copy(), output_dir=output_dir)
    live_probs = align_probability_columns(
        final_model.predict_proba(X_final),
        N_CB_PROBS,
        classes=final_cb_classes,
    )
    live_regime_labels = final_regime.predict(df.copy())
    live_regime_oh = np.zeros((n, N_CLUSTERS), dtype=np.float32)
    for i, label in enumerate(live_regime_labels):
        if 0 <= int(label) < N_CLUSTERS:
            live_regime_oh[i, int(label)] = 1.0
        else:
            live_regime_oh[i, 0] = 1.0
    np.save(
        os.path.join(output_dir, 'meta_features_live_v19.npy'),
        np.concatenate([live_probs, live_regime_oh], axis=1).astype(np.float32),
    )

    meta = np.concatenate([oof_probs, regime_oh], axis=1).astype(np.float32)
    np.save(os.path.join(output_dir, 'meta_features_oof_v19.npy'), meta)
    np.save(os.path.join(output_dir, 'meta_coverage_v19.npy'), coverage.astype(np.uint8))
    print(f"  ✅ OOF Meta Features: {meta.shape}")
    print(f"  ✅ Coverage: {coverage.sum():,}/{len(coverage):,} ({coverage.mean():.1%})")
    return meta, coverage


def _load_lob_inputs(lob_path: str | None, lob_ts_path: str | None):
    if not lob_path or not os.path.exists(lob_path):
        return None, None
    tensors = np.load(lob_path, mmap_mode='r')
    if lob_ts_path and os.path.exists(lob_ts_path):
        ts_raw = np.load(lob_ts_path)
        ts = pd.to_datetime(ts_raw.astype(np.int64), unit='ns', utc=True, errors='coerce').tz_localize(None)
    else:
        ts = pd.to_datetime(pd.Series(range(len(tensors))), unit='s', utc=True, errors='coerce').dt.tz_localize(None)
    if len(ts) != len(tensors):
        print(
            "  ⚠️ LOB tensors/timestamps length mismatch: "
            f"tensors={len(tensors):,}, timestamps={len(ts):,}. "
            f"Visual stage will use the first {min(len(tensors), len(ts)):,} aligned items only."
        )
    return tensors, pd.Series(ts)


def _align_lob_to_rows(
    df: pd.DataFrame,
    lob_timestamps: pd.Series,
    max_age: str = '5s',
    max_tensors: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_lob = len(lob_timestamps) if max_tensors is None else min(len(lob_timestamps), int(max_tensors))
    row_ts = _time_series(df, 'ts_event')
    row_df = pd.DataFrame({
        'ts_event': row_ts,
        'row_idx': np.arange(len(df), dtype=np.int32),
    })
    lob_df = pd.DataFrame({
        'ts_event': pd.to_datetime(pd.Series(lob_timestamps).iloc[:n_lob], utc=True, errors='coerce').dt.tz_localize(None),
        'tensor_idx': np.arange(n_lob, dtype=np.int32),
    }).dropna(subset=['ts_event']).sort_values('ts_event').reset_index(drop=True)

    row_merged = pd.merge_asof(
        row_df.sort_values('ts_event'),
        lob_df,
        on='ts_event',
        direction='backward',
        tolerance=pd.Timedelta(max_age),
    ).sort_values('row_idx')

    row_to_tensor = row_merged['tensor_idx'].fillna(-1).astype(np.int32).values

    obi_series = df.get('obi', pd.Series(np.zeros(len(df)))).fillna(0).astype(np.float32)
    tensor_targets = np.zeros(n_lob, dtype=np.float32)
    tensor_target_seen = np.zeros(n_lob, dtype=bool)
    tensor_rows = pd.merge_asof(
        lob_df,
        row_df.sort_values('ts_event'),
        on='ts_event',
        direction='backward',
        tolerance=pd.Timedelta(max_age),
    )
    for _, match in tensor_rows.dropna(subset=['row_idx']).iterrows():
        tensor_idx = int(match['tensor_idx'])
        row_idx = int(match['row_idx'])
        tensor_targets[tensor_idx] = float(obi_series.iloc[row_idx])
        tensor_target_seen[tensor_idx] = True

    return row_to_tensor, tensor_targets, tensor_target_seen


def stage2_oof_visual_embeddings(
    df: pd.DataFrame,
    output_dir: str,
    splits,
    lob_tensors,
    lob_timestamps: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    print("\n" + "═" * 65)
    print("👁️ STAGE 2 — V19 OOF DeepLOB Visual Embeddings")
    print("═" * 65)

    n_rows = len(df)
    zero_emb = np.zeros((n_rows, VISUAL_EMB_DIM), dtype=np.float32)
    zero_cov = np.zeros(n_rows, dtype=bool)

    if lob_tensors is None or lob_timestamps is None or len(lob_tensors) == 0:
        print("  ⚠️ LOB inputs غير متاحة — visual embeddings = 0")
        np.save(os.path.join(output_dir, 'visual_embeddings_v19.npy'), zero_emb)
        np.save(os.path.join(output_dir, 'visual_embeddings_live_v19.npy'), zero_emb)
        np.save(os.path.join(output_dir, 'visual_coverage_v19.npy'), zero_cov.astype(np.uint8))
        return zero_emb, zero_cov

    DeepLOBCNN, DEEPLOB_IMPORT_OK = _load_deeplob_runtime()
    if not DEEPLOB_IMPORT_OK:
        print("  ⚠️ DeepLOB import غير متاح — visual embeddings = 0")
        np.save(os.path.join(output_dir, 'visual_embeddings_v19.npy'), zero_emb)
        np.save(os.path.join(output_dir, 'visual_embeddings_live_v19.npy'), zero_emb)
        np.save(os.path.join(output_dir, 'visual_coverage_v19.npy'), zero_cov.astype(np.uint8))
        return zero_emb, zero_cov

    row_to_tensor, tensor_targets, tensor_target_seen = _align_lob_to_rows(
        df,
        lob_timestamps,
        max_tensors=len(lob_tensors),
    )
    row_embs = np.zeros((n_rows, VISUAL_EMB_DIM), dtype=np.float32)
    row_cov = np.zeros(n_rows, dtype=bool)

    metrics = {
        'n_rows': int(n_rows),
        'n_tensors': int(min(len(lob_tensors), len(lob_timestamps))),
        'n_tensors_raw': int(len(lob_tensors)),
        'n_timestamps_raw': int(len(lob_timestamps)),
        'rows_with_tensor': int(np.sum(row_to_tensor >= 0)),
        'folds': [],
    }

    fold_tmp_dir = tempfile.mkdtemp(prefix='_oof_cnn_tmp_', dir=output_dir)

    for fold_no, (train_idx, test_idx) in enumerate(splits, start=1):
        test_tensor_ids = np.unique(row_to_tensor[test_idx][row_to_tensor[test_idx] >= 0]).astype(np.int32)
        train_tensor_ids = np.unique(row_to_tensor[train_idx][row_to_tensor[train_idx] >= 0]).astype(np.int32)

        if len(test_tensor_ids) == 0:
            metrics['folds'].append({'fold': fold_no, 'train_tensors': int(len(train_tensor_ids)), 'test_tensors': 0})
            continue

        overlap = np.intersect1d(train_tensor_ids, test_tensor_ids, assume_unique=False)
        if len(overlap) > 0:
            train_tensor_ids = train_tensor_ids[~np.isin(train_tensor_ids, overlap)]

        train_tensor_ids = train_tensor_ids[tensor_target_seen[train_tensor_ids]]
        if len(train_tensor_ids) < 32:
            metrics['folds'].append({
                'fold': fold_no,
                'train_tensors': int(len(train_tensor_ids)),
                'test_tensors': int(len(test_tensor_ids)),
                'status': 'insufficient_train_tensors',
            })
            continue

        brain_path = os.path.join(fold_tmp_dir, f'deeplob_fold_{fold_no}.keras')
        cnn = DeepLOBCNN(brain_file=brain_path)
        if cnn.model is None:
            metrics['folds'].append({
                'fold': fold_no,
                'train_tensors': int(len(train_tensor_ids)),
                'test_tensors': int(len(test_tensor_ids)),
                'status': 'cnn_unavailable',
            })
            continue

        X_tr = np.asarray(lob_tensors[train_tensor_ids], dtype=np.float32)
        y_tr = tensor_targets[train_tensor_ids].reshape(-1, 1).astype(np.float32)
        cnn.fit_auxiliary(X_tr, y_tr, epochs=20, batch=256, output_dir=fold_tmp_dir)

        X_te = np.asarray(lob_tensors[test_tensor_ids], dtype=np.float32)
        emb_te = cnn.get_embeddings(X_te).astype(np.float32)
        emb_map = {int(tid): emb_te[i] for i, tid in enumerate(test_tensor_ids)}

        fold_rows = 0
        for row_idx in test_idx:
            tensor_idx = int(row_to_tensor[row_idx])
            if tensor_idx < 0:
                continue
            if tensor_idx in emb_map:
                row_embs[row_idx] = emb_map[tensor_idx]
                row_cov[row_idx] = True
                fold_rows += 1

        metrics['folds'].append({
            'fold': fold_no,
            'train_tensors': int(len(train_tensor_ids)),
            'test_tensors': int(len(test_tensor_ids)),
            'covered_rows': int(fold_rows),
            'status': 'ok',
        })

    live_row_embs = np.zeros((n_rows, VISUAL_EMB_DIM), dtype=np.float32)
    final_tensor_ids = np.unique(row_to_tensor[row_to_tensor >= 0]).astype(np.int32)
    final_tensor_ids = final_tensor_ids[tensor_target_seen[final_tensor_ids]]
    if len(final_tensor_ids) >= 32:
        final_brain_tmp = os.path.join(fold_tmp_dir, 'deeplob_final_tmp.keras')
        final_cnn = DeepLOBCNN(brain_file=final_brain_tmp)
        if final_cnn.model is not None:
            X_final = np.asarray(lob_tensors[final_tensor_ids], dtype=np.float32)
            y_final = tensor_targets[final_tensor_ids].reshape(-1, 1).astype(np.float32)
            final_cnn.fit_auxiliary(X_final, y_final, epochs=25, batch=256, output_dir=output_dir)
            final_cnn.model.save(os.path.join(output_dir, 'deeplob_cnn_v19.keras'))
            final_emb = np.asarray(final_cnn.get_embeddings(X_final), dtype=np.float32)
            final_emb_map = {int(tid): final_emb[i] for i, tid in enumerate(final_tensor_ids)}
            for row_idx, tensor_idx in enumerate(row_to_tensor):
                if int(tensor_idx) in final_emb_map:
                    live_row_embs[row_idx] = final_emb_map[int(tensor_idx)]

    np.save(os.path.join(output_dir, 'visual_embeddings_v19.npy'), row_embs.astype(np.float32))
    np.save(os.path.join(output_dir, 'visual_embeddings_live_v19.npy'), live_row_embs.astype(np.float32))
    np.save(os.path.join(output_dir, 'visual_coverage_v19.npy'), row_cov.astype(np.uint8))
    with open(os.path.join(output_dir, 'visual_metrics_v19.json'), 'w') as f:
        json.dump({
            **metrics,
            'coverage_ratio': float(row_cov.mean()),
        }, f, indent=2)

    print(f"  ✅ Visual Embeddings: {row_embs.shape}")
    print(f"  ✅ Visual Coverage: {row_cov.sum():,}/{n_rows:,} ({row_cov.mean():.1%})")
    return row_embs, row_cov


def build_safe_sequences(
    df: pd.DataFrame,
    X_rows: np.ndarray,
    coverage_mask: np.ndarray,
    seq_len: int = SEQ_LEN,
    train_frac: float = 0.80,
    min_seq_coverage: float = 0.80,
    n_stat_feat: int = len(CATBOOST_ADVISOR_FEATURES),
    sequence_aux_mode: str = SEQUENCE_AUX_LAST_STEP_ONLY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    n = len(df)
    split_ctx = _sequence_split_context(df, seq_len=seq_len, train_frac=train_frac)
    split_idx = split_ctx['split_idx']
    split_time = split_ctx['split_time']

    y_bias = df['bias_label'].fillna(1).astype(np.int32).values
    if 'conf_target' in df.columns:
        y_conf = pd.to_numeric(df['conf_target'], errors='coerce').fillna(0).astype(np.float32).values
    elif 'signal_quality' in df.columns:
        y_conf = (pd.to_numeric(df['signal_quality'], errors='coerce').fillna(0).astype(np.int32).values == 2).astype(np.float32)
    else:
        y_conf = df.get('conf_label', pd.Series(np.zeros(n))).fillna(0).astype(np.float32).values

    train_row_ok = split_ctx['train_row_ok']
    val_row_ok = split_ctx['val_row_ok']

    X_tr, yb_tr, yc_tr = [], [], []
    X_val, yb_val, yc_val = [], [], []

    def _coverage_ok(window_cov: np.ndarray) -> bool:
        cov = np.asarray(window_cov, dtype=bool)
        if cov.size == 0:
            return False
        if not bool(cov[-1]):
            return False
        if sequence_aux_mode == SEQUENCE_AUX_LAST_STEP_ONLY:
            return True
        return float(np.mean(cov)) >= float(min_seq_coverage)

    for end_idx in range(seq_len - 1, split_idx):
        start_idx = end_idx - seq_len + 1
        if not train_row_ok[end_idx]:
            continue
        if not _coverage_ok(coverage_mask[start_idx:end_idx + 1]):
            continue
        X_tr.append(
            _project_sequence_aux_context(
                X_rows[start_idx:end_idx + 1],
                n_stat_feat=n_stat_feat,
                sequence_aux_mode=sequence_aux_mode,
            )
        )
        yb_tr.append(y_bias[end_idx])
        yc_tr.append(y_conf[end_idx])

    for end_idx in range(split_idx + seq_len - 1, n):
        start_idx = end_idx - seq_len + 1
        if start_idx < split_idx:
            continue
        if not val_row_ok[end_idx]:
            continue
        if not _coverage_ok(coverage_mask[start_idx:end_idx + 1]):
            continue
        X_val.append(
            _project_sequence_aux_context(
                X_rows[start_idx:end_idx + 1],
                n_stat_feat=n_stat_feat,
                sequence_aux_mode=sequence_aux_mode,
            )
        )
        yb_val.append(y_bias[end_idx])
        yc_val.append(y_conf[end_idx])

    X_tr = np.array(X_tr, dtype=np.float32)
    yb_tr = np.array(yb_tr, dtype=np.int32)
    yc_tr = np.array(yc_tr, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    yb_val = np.array(yb_val, dtype=np.int32)
    yc_val = np.array(yc_val, dtype=np.float32)

    stats = {
        'split_idx': int(split_idx),
        'split_time': str(split_time),
        'train_sequences': int(len(X_tr)),
        'val_sequences': int(len(X_val)),
        'min_seq_coverage': float(min_seq_coverage),
        'sequence_aux_mode': str(sequence_aux_mode),
    }
    return X_tr, yb_tr, yc_tr, X_val, yb_val, yc_val, stats


def _compute_bias_class_weights(y_bias: np.ndarray, max_weight: float = 2.5) -> dict[int, float]:
    y_bias = np.asarray(y_bias, dtype=np.int32)
    valid = y_bias[(y_bias >= 0) & (y_bias < 2)]
    if valid.size == 0:
        return {}
    counts = np.bincount(valid, minlength=2)[:2]
    nonzero = counts[counts > 0]
    if nonzero.size < 2:
        return {int(cls): 1.0 for cls, count in enumerate(counts) if count > 0}
    majority = float(nonzero.max())
    weights: dict[int, float] = {}
    for cls, count in enumerate(counts):
        if count <= 0:
            continue
        weights[int(cls)] = float(np.clip(majority / float(count), 1.0, max_weight))
    return weights


def _infer_event_gate_schema(df: pd.DataFrame) -> dict:
    event_cfg = {
        'roll_window': DEFAULT_EVENT_ROLL_WINDOW,
        'vol_mult': DEFAULT_EVENT_VOL_MULT,
        'obi_thr': DEFAULT_EVENT_OBI_THR,
        'wall_str_thr': DEFAULT_EVENT_WALL_STR_THR,
        'shift_z_thr': DEFAULT_EVENT_SHIFT_Z_THR,
        'score_threshold': DEFAULT_EVENT_SCORE_THRESHOLD,
    }
    if len(df) == 0 or 'event_score' not in df.columns:
        return event_cfg

    event_col = 'train_event_flag' if 'train_event_flag' in df.columns else 'event_flag'
    event_mask = (
        pd.to_numeric(df.get(event_col, 0), errors='coerce')
        .fillna(0)
        .astype(np.int8)
        == 1
    )
    if not event_mask.any():
        return event_cfg

    event_scores = pd.to_numeric(df.get('event_score', 0.0), errors='coerce')
    selected_scores = event_scores[event_mask].dropna()
    if len(selected_scores):
        event_cfg['score_threshold'] = float(max(selected_scores.min(), 0.0))
    return event_cfg


def stage3_meta_learner_v19(
    df: pd.DataFrame,
    meta_features: np.ndarray,
    visual_embeddings: np.ndarray,
    coverage_mask: np.ndarray,
    inference_scaler_params: dict,
    output_dir: str,
    event_gate_cfg: dict | None = None,
    epochs: int = 100,
    batch: int = 64,
    train_frac: float = 0.80,
    min_seq_coverage: float = 0.80,
) -> None:
    print("\n" + "═" * 65)
    print("🧠 STAGE 3 — V19 MetaLearner (Safe Sequence Split)")
    print("═" * 65)
    MetaLearnerLSTM = _load_meta_learner_class()

    # OOF CatBoost probabilities / visual embeddings are intentionally exposed
    # only on the last step of each sequence. Historical timesteps stay pure
    # market-structure features to avoid fold-boundary artifacts in the LSTM.
    sequence_aux_mode = SEQUENCE_AUX_LAST_STEP_ONLY
    X_stat = _build_scaled_stat_matrix(df, CATBOOST_ADVISOR_FEATURES, inference_scaler_params)
    X_rows = np.concatenate([X_stat, meta_features, visual_embeddings], axis=1).astype(np.float32)

    X_tr, yb_tr, yc_tr, X_val, yb_val, yc_val, split_stats = build_safe_sequences(
        df,
        X_rows,
        coverage_mask=coverage_mask,
        seq_len=SEQ_LEN,
        train_frac=train_frac,
        min_seq_coverage=min_seq_coverage,
        n_stat_feat=len(CATBOOST_ADVISOR_FEATURES),
        sequence_aux_mode=sequence_aux_mode,
    )
    if len(X_tr) == 0 or len(X_val) == 0:
        raise RuntimeError(
            '❌ لا توجد sequences كافية بعد تطبيق coverage + safe split. '
            'جرّب زيادة الداتا أو تقليل test_size/embargo.'
        )

    print(f"  Train Sequences: {len(X_tr):,}")
    print(f"  Val Sequences:   {len(X_val):,}")
    bias_counts = np.bincount(yb_tr, minlength=2)[:2]
    bias_class_weights = _compute_bias_class_weights(yb_tr)
    print(
        "  Bias Seq Counts: "
        f"LONG={int(bias_counts[0]):,} SHORT={int(bias_counts[1]):,}"
    )
    if bias_class_weights:
        print(
            "  Bias Class Weights: "
            + " ".join(
                f"{BIAS_LABELS.get(cls, cls)}={weight:.2f}"
                for cls, weight in sorted(bias_class_weights.items())
            )
        )

    meta = MetaLearnerLSTM(
        seq_len=SEQ_LEN,
        n_stat_feat=len(CATBOOST_ADVISOR_FEATURES),
        n_meta_feat=int(meta_features.shape[1]),
        n_visual_emb=VISUAL_EMB_DIM,
        brain_file=os.path.join(output_dir, 'meta_learner_v19.keras'),
        lstm_units_1=128,
        lstm_units_2=64,
        dropout=0.25,
        confidence_threshold=0.65,
    )

    history = meta.fit_train_val(
        X_tr, yb_tr, yc_tr,
        X_val, yb_val, yc_val,
        epochs=epochs,
        batch=batch,
        output_dir=output_dir,
        class_weights=bias_class_weights,
    )

    if history is not None:
        hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        event_gate_cfg = event_gate_cfg or _infer_event_gate_schema(df)
        with open(os.path.join(output_dir, 'meta_learner_v19_history.json'), 'w') as f:
            json.dump(
                {
                    'history': hist_dict,
                    'split': split_stats,
                    'bias_class_weights': {str(k): float(v) for k, v in bias_class_weights.items()},
                    'event_gate': event_gate_cfg,
                },
                f,
                indent=2,
            )
        schema = {
            'version': SCHEMA_VERSION,
            'seq_len': SEQ_LEN,
            'stat_features': CATBOOST_ADVISOR_FEATURES,
            'meta_features': META_FEATURE_NAMES,
            'visual_features': VISUAL_FEATURE_NAMES,
            'sequence_aux_mode': sequence_aux_mode,
            'passthrough_cols': TRAINING_PASSTHROUGH_COLS,
            'timestamp_cols': ['ts_event', 'label_end_ts'],
            'input_dim': int(X_rows.shape[1]),
            'artifacts': {
                'catboost_model': 'catboost_advisor_v19.cbm',
                'catboost_classes': 'catboost_classes_v19.json',
                'meta_model': 'meta_learner_v19.keras',
                'regime_model': 'regime_classifier.pkl',
                'scaler_params': 'scaler_params.json',
                'deeplob_model': 'deeplob_cnn_v19.keras',
                'visual_embeddings': 'visual_embeddings_live_v19.npy',
                'visual_embeddings_oof': 'visual_embeddings_v19.npy',
                'visual_embeddings_live': 'visual_embeddings_live_v19.npy',
                'meta_features_oof': 'meta_features_oof_v19.npy',
                'meta_features_live': 'meta_features_live_v19.npy',
            },
            'event_gate': event_gate_cfg,
        }
        with open(os.path.join(output_dir, 'feature_schema_v19.json'), 'w') as f:
            json.dump(schema, f, indent=2)
        print("  ✅ MetaLearner V19 history + schema محفوظان")


def _load_required_stage1_artifacts(output_dir: str, n_rows: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    meta_path = os.path.join(output_dir, 'meta_features_oof_v19.npy')
    coverage_path = os.path.join(output_dir, 'meta_coverage_v19.npy')
    required_files = [
        meta_path,
        coverage_path,
        os.path.join(output_dir, 'catboost_advisor_v19.cbm'),
        os.path.join(output_dir, 'catboost_classes_v19.json'),
        os.path.join(output_dir, 'regime_classifier.pkl'),
    ]
    missing = [path for path in required_files if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            '❌ CatBoost stage artifacts missing. '
            'شغّل المرحلة الثانية أولاً:\n'
            'python train_v19.py --csv <training_features_ready.csv> --output <dir> --phase catboost\n'
            f'Missing: {missing}'
        )

    meta_features = np.load(meta_path)
    coverage = np.load(coverage_path).astype(bool)
    if n_rows is not None and (len(meta_features) != int(n_rows) or len(coverage) != int(n_rows)):
        raise ValueError(
            f'❌ Stage1 cached artifacts shape mismatch: meta={len(meta_features)}, coverage={len(coverage)}, expected={int(n_rows)}'
        )
    return meta_features, coverage


def _load_or_init_visual_artifacts(output_dir: str, n_rows: int) -> tuple[np.ndarray, np.ndarray, str]:
    visual_path = os.path.join(output_dir, 'visual_embeddings_v19.npy')
    visual_live_path = os.path.join(output_dir, 'visual_embeddings_live_v19.npy')
    visual_cov_path = os.path.join(output_dir, 'visual_coverage_v19.npy')

    if os.path.exists(visual_path) and os.path.exists(visual_cov_path):
        visual = np.load(visual_path).astype(np.float32)
        visual_cov = np.load(visual_cov_path).astype(bool)
        if len(visual) == n_rows and len(visual_cov) == n_rows:
            return visual, visual_cov, 'cache'

    visual = np.zeros((n_rows, VISUAL_EMB_DIM), dtype=np.float32)
    visual_cov = np.zeros(n_rows, dtype=bool)
    np.save(visual_path, visual)
    np.save(visual_live_path, visual)
    np.save(visual_cov_path, visual_cov.astype(np.uint8))
    return visual, visual_cov, 'zeros'


def run_training_pipeline(
    csv_path: str,
    output_dir: str = 'outputs_v19',
    lob_path: str | None = None,
    lob_ts_path: str | None = None,
    epochs: int = 100,
    batch: int = 64,
    n_folds: int = 6,
    test_size: float = 0.10,
    embargo_pct: float = 0.02,
    min_train_pct: float = 0.20,
    train_frac: float = 0.80,
    min_seq_coverage: float = 0.80,
    stage: int = 0,
    phase: str | None = None,
    catboost_device: str = 'auto',
    training_mode: str | None = None,
    quality_weight_strong: float | None = None,
    quality_weight_weak: float | None = None,
    config_snapshot: dict | None = None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    started_at = datetime.datetime.now()
    resolved_phase = _resolve_phase(stage=stage, phase=phase)

    print('=' * 65)
    print('🚀 QuantSystem V19 — Leakage-Safe Training Foundation')
    print(f'   Output: {output_dir}')
    print(f'   Phase: {_phase_banner(resolved_phase)}')
    print('=' * 65)

    train_cfg = (config_snapshot or {}).get('training', {})
    training_mode = training_mode or str(train_cfg.get('mode', TRAIN_MODE_EVENT_BINARY))
    quality_weight_strong = float(
        quality_weight_strong if quality_weight_strong is not None else train_cfg.get('quality_weight_strong', 2.0)
    )
    quality_weight_weak = float(
        quality_weight_weak if quality_weight_weak is not None else train_cfg.get('quality_weight_weak', 1.0)
    )

    df_full = load_training_csv(csv_path)
    event_df, event_view_info = build_event_training_view(
        df_full,
        mode=training_mode,
        quality_weight_strong=quality_weight_strong,
        quality_weight_weak=quality_weight_weak,
    )
    with open(os.path.join(output_dir, 'event_training_view.json'), 'w') as f:
        json.dump(event_view_info, f, indent=2)
    copied_artifacts = copy_inference_artifacts(csv_path, output_dir)
    if copied_artifacts:
        print(f"  ✅ Inference artifacts copied: {list(copied_artifacts)}")

    inference_scaler_params, scaler_info = build_inference_scaler_params(
        event_df,
        CATBOOST_ADVISOR_FEATURES,
        train_frac=train_frac,
    )
    scaler_path = _save_scaler_params(output_dir, inference_scaler_params)
    print(
        f"  ✅ Model scaler saved: {scaler_path} | "
        f"rows={scaler_info['scaler_train_rows']:,} | split={scaler_info['split_time']}"
    )

    if not lob_path:
        guess = os.path.join(os.path.dirname(os.path.abspath(csv_path)), 'lob_tensors.npy')
        if os.path.exists(guess):
            lob_path = guess
    if not lob_ts_path:
        guess_ts = os.path.join(os.path.dirname(os.path.abspath(csv_path)), 'lob_tensor_timestamps.npy')
        if os.path.exists(guess_ts):
            lob_ts_path = guess_ts

    splits, split_t0, split_t1 = build_time_splits(
        event_df,
        n_folds=n_folds,
        test_size=test_size,
        embargo_pct=embargo_pct,
        min_train_pct=min_train_pct,
    )

    meta_path = os.path.join(output_dir, 'meta_features_oof_v19.npy')
    coverage_path = os.path.join(output_dir, 'meta_coverage_v19.npy')
    visual_path = os.path.join(output_dir, 'visual_embeddings_v19.npy')
    visual_cov_path = os.path.join(output_dir, 'visual_coverage_v19.npy')

    if resolved_phase in (PHASE_FULL, PHASE_CATBOOST) or (
        resolved_phase == PHASE_VISUAL and not (os.path.exists(meta_path) and os.path.exists(coverage_path))
    ):
        meta_features, coverage = stage1_oof_meta(
            event_df,
            output_dir,
            splits=splits,
            n_folds=n_folds,
            test_size=test_size,
            embargo_pct=embargo_pct,
            min_train_pct=min_train_pct,
            t0=split_t0,
            t1=split_t1,
            inference_scaler_params=inference_scaler_params,
            catboost_device=catboost_device,
            quality_weight_strong=quality_weight_strong,
            quality_weight_weak=quality_weight_weak,
        )
    else:
        meta_features, coverage = _load_required_stage1_artifacts(output_dir, n_rows=len(event_df))
        print(f'✅ CatBoost artifacts loaded from cache: {meta_features.shape}')

    if resolved_phase == PHASE_CATBOOST:
        elapsed = (datetime.datetime.now() - started_at).total_seconds()
        summary = {
            'rows_full': int(len(df_full)),
            'rows_event': int(len(event_df)),
            'training_mode': training_mode,
            'meta_shape': list(meta_features.shape),
            'meta_coverage_ratio': float(np.mean(coverage)),
            'visual_shape': None,
            'visual_coverage_ratio': None,
            'scaler_train_rows': int(scaler_info['scaler_train_rows']),
            'elapsed_seconds': float(elapsed),
            'stage': int(stage),
            'phase': resolved_phase,
        }
        manifest_path = write_manifest(
            output_dir=output_dir,
            kind='train_v19_catboost',
            config=config_snapshot or {
                'epochs': epochs,
                'batch': batch,
                'n_folds': n_folds,
                'test_size': test_size,
                'embargo_pct': embargo_pct,
                'train_frac': train_frac,
                'stage': stage,
                'phase': resolved_phase,
                'catboost_device': catboost_device,
                'mode': training_mode,
                'quality_weight_strong': quality_weight_strong,
                'quality_weight_weak': quality_weight_weak,
            },
            inputs={
                'csv': csv_path,
                'lob': lob_path,
                'lob_ts': lob_ts_path,
            },
            metrics=summary,
        )
        print('\n' + '=' * 65)
        print('✅ CatBoost stage completed independently')
        print(f'📄 Manifest: {manifest_path}')
        print('=' * 65)
        return {
            **summary,
            'output_dir': output_dir,
            'manifest': manifest_path,
            'meta_features': meta_path,
        }

    lob_tensors, lob_timestamps = _load_lob_inputs(lob_path, lob_ts_path)
    if resolved_phase in (PHASE_FULL, PHASE_VISUAL):
        visual_embeddings, visual_coverage = stage2_oof_visual_embeddings(
            event_df,
            output_dir,
            splits=splits,
            lob_tensors=lob_tensors,
            lob_timestamps=lob_timestamps,
        )
    else:
        visual_embeddings, visual_coverage, visual_source = _load_or_init_visual_artifacts(output_dir, len(event_df))
        print(f'✅ Visual embeddings ready: {visual_embeddings.shape} | source={visual_source}')

    if resolved_phase == PHASE_VISUAL:
        elapsed = (datetime.datetime.now() - started_at).total_seconds()
        summary = {
            'rows_full': int(len(df_full)),
            'rows_event': int(len(event_df)),
            'training_mode': training_mode,
            'meta_shape': list(meta_features.shape),
            'meta_coverage_ratio': float(np.mean(coverage)),
            'visual_shape': list(visual_embeddings.shape),
            'visual_coverage_ratio': float(np.mean(visual_coverage)),
            'scaler_train_rows': int(scaler_info['scaler_train_rows']),
            'elapsed_seconds': float(elapsed),
            'stage': int(stage),
            'phase': resolved_phase,
        }
        manifest_path = write_manifest(
            output_dir=output_dir,
            kind='train_v19_visual',
            config=config_snapshot or {
                'epochs': epochs,
                'batch': batch,
                'n_folds': n_folds,
                'test_size': test_size,
                'embargo_pct': embargo_pct,
                'train_frac': train_frac,
                'stage': stage,
                'phase': resolved_phase,
                'catboost_device': catboost_device,
                'mode': training_mode,
                'quality_weight_strong': quality_weight_strong,
                'quality_weight_weak': quality_weight_weak,
            },
            inputs={
                'csv': csv_path,
                'lob': lob_path,
                'lob_ts': lob_ts_path,
            },
            metrics=summary,
        )
        print('\n' + '=' * 65)
        print('✅ Visual stage completed independently')
        print(f'📄 Manifest: {manifest_path}')
        print('=' * 65)
        return {
            **summary,
            'output_dir': output_dir,
            'manifest': manifest_path,
            'visual_embeddings': visual_path,
        }

    if resolved_phase in (PHASE_FULL, PHASE_TRAIN):
        try:
            _load_meta_learner_class()
        except Exception as e:
            raise RuntimeError(
                "❌ TensorFlow/MetaLearner غير متاح. المرحلة الثالثة لا يمكن تشغيلها الآن.\n"
                "ثبّت TensorFlow أولًا أو شغّل المرحلة الثالثة على Linux/WSL2."
            ) from e
        stage3_meta_learner_v19(
            event_df,
            meta_features,
            visual_embeddings,
            coverage_mask=coverage,
            inference_scaler_params=inference_scaler_params,
            output_dir=output_dir,
            event_gate_cfg=_infer_event_gate_schema(df_full),
            epochs=epochs,
            batch=batch,
            train_frac=train_frac,
            min_seq_coverage=min_seq_coverage,
        )

    elapsed = (datetime.datetime.now() - started_at).total_seconds()
    summary = {
        'rows_full': int(len(df_full)),
        'rows_event': int(len(event_df)),
        'training_mode': training_mode,
        'meta_shape': list(meta_features.shape),
        'meta_coverage_ratio': float(np.mean(coverage)),
        'visual_shape': list(visual_embeddings.shape),
        'visual_coverage_ratio': float(np.mean(visual_coverage)),
        'scaler_train_rows': int(scaler_info['scaler_train_rows']),
        'elapsed_seconds': float(elapsed),
        'stage': int(stage),
        'phase': resolved_phase,
    }
    manifest_path = write_manifest(
        output_dir=output_dir,
        kind='train_v19',
        config=config_snapshot or {
            'epochs': epochs,
            'batch': batch,
            'n_folds': n_folds,
            'test_size': test_size,
            'embargo_pct': embargo_pct,
            'min_train_pct': min_train_pct,
            'train_frac': train_frac,
            'min_seq_coverage': min_seq_coverage,
            'stage': stage,
            'phase': resolved_phase,
            'catboost_device': catboost_device,
            'mode': training_mode,
            'quality_weight_strong': quality_weight_strong,
            'quality_weight_weak': quality_weight_weak,
        },
        inputs={
            'csv': csv_path,
            'lob': lob_path,
            'lob_ts': lob_ts_path,
        },
        metrics=summary,
    )
    print('\n' + '=' * 65)
    print(f'✅ V19 training slice اكتمل في {elapsed:.0f}s ({elapsed/60:.1f} دقيقة)')
    print(f'📄 Manifest: {manifest_path}')
    print('=' * 65)
    return {
        **summary,
        'output_dir': output_dir,
        'manifest': manifest_path,
        'feature_schema': os.path.join(output_dir, 'feature_schema_v19.json'),
        'visual_embeddings': visual_path,
        'meta_features': meta_path,
    }


def main():
    defaults = load_v19_config().get('training', {})
    p = argparse.ArgumentParser(description='QuantSystem V19 leakage-safe training')
    p.add_argument('--csv', required=True, help='training_features_ready.csv from label_mode=v19')
    p.add_argument('--lob', default=None, help='optional lob_tensors.npy')
    p.add_argument('--lob_ts', default=None, help='optional lob_tensor_timestamps.npy')
    p.add_argument('--output', default=defaults.get('output_dir', 'outputs_v19'), help='output directory')
    p.add_argument('--epochs', type=int, default=int(defaults.get('epochs', 100)))
    p.add_argument('--batch', type=int, default=int(defaults.get('batch', 64)))
    p.add_argument('--n_folds', type=int, default=int(defaults.get('n_folds', 6)))
    p.add_argument('--test_size', type=float, default=float(defaults.get('test_size', 0.10)))
    p.add_argument('--embargo_pct', type=float, default=float(defaults.get('embargo_pct', 0.02)))
    p.add_argument('--min_train_pct', type=float, default=float(defaults.get('min_train_pct', 0.20)))
    p.add_argument('--train_frac', type=float, default=float(defaults.get('train_frac', 0.80)))
    p.add_argument('--min_seq_coverage', type=float, default=float(defaults.get('min_seq_coverage', 0.80)))
    p.add_argument('--stage', type=int, default=int(defaults.get('stage', 0)), help='0=all, 1=stage1 only, 2=stage2 only, 3=stage3 only')
    p.add_argument('--phase', default=None, choices=['full', 'all', 'catboost', 'cb', 'visual', 'deeplob', 'train', 'training', 'meta'], help='preferred named phase: catboost-only, visual-only, or train-only')
    p.add_argument('--catboost_device', default='auto', choices=['auto', 'cpu', 'gpu'], help='device selection for CatBoost stage')
    p.add_argument('--training_mode', default=str(defaults.get('mode', TRAIN_MODE_EVENT_BINARY)))
    p.add_argument('--quality_weight_strong', type=float, default=float(defaults.get('quality_weight_strong', 2.0)))
    p.add_argument('--quality_weight_weak', type=float, default=float(defaults.get('quality_weight_weak', 1.0)))
    p.add_argument('--config', default=None, help='optional config file to override defaults')
    args = p.parse_args()

    cfg = load_v19_config(args.config)
    run_training_pipeline(
        csv_path=args.csv,
        output_dir=args.output,
        lob_path=args.lob,
        lob_ts_path=args.lob_ts,
        epochs=args.epochs,
        batch=args.batch,
        n_folds=args.n_folds,
        test_size=args.test_size,
        embargo_pct=args.embargo_pct,
        min_train_pct=args.min_train_pct,
        train_frac=args.train_frac,
        min_seq_coverage=args.min_seq_coverage,
        stage=args.stage,
        phase=args.phase,
        catboost_device=args.catboost_device,
        training_mode=args.training_mode,
        quality_weight_strong=args.quality_weight_strong,
        quality_weight_weak=args.quality_weight_weak,
        config_snapshot=cfg,
    )


if __name__ == '__main__':
    main()
