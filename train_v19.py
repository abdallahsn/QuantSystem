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
import pickle
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss, precision_recall_fscore_support

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
from modules.decision_policy_v19 import (
    DEFAULT_DECISION_POLICY_ARTIFACT,
    build_decision_policy,
)
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
    ROBUST_IQR_MIN,
    apply_scaler_params_to_frame,
    fit_numeric_scaler_param,
    infer_meta_feature_layout,
    prepare_feature_frame,
    resolve_meta_feature_names,
)
from modules.feature_artifact_v19 import load_artifact_manifest, load_feature_artifact, resolve_artifact_root
from modules.gpu_config import detect_gpu
from modules.manifest_v19 import write_manifest
from modules.oof_stacking import (
    align_probability_columns,
    fill_uncovered_one_hot,
    fill_uncovered_probabilities,
    run_sequential_oof,
)
from modules.purging_embargo import embargo_observations, purge_overlapping, walk_forward_expanding
from modules.regime_classifier import (
    HMM_AVAILABLE,
    REGIME_META_SCORE_COLS,
    REGIME_ONE_HOT_COLS,
    RegimeClassifier,
)

try:
    from catboost import CatBoostClassifier, Pool
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
    XGB_IMPORT_ERROR = None
except Exception as exc:
    XGB_AVAILABLE = False
    XGB_IMPORT_ERROR = exc

BIAS_LABELS = {0: 'LONG', 1: 'SHORT', 2: 'NEUTRAL'}
N_CLUSTERS = 4
N_CB_PROBS = 2
N_XGB_PROBS = 2
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
LEGACY_META_FEATURE_NAMES = resolve_meta_feature_names(include_xgboost=False)
META_FEATURE_NAMES = resolve_meta_feature_names(include_xgboost=True)
VISUAL_FEATURE_NAMES = [f'vis_emb_{i}' for i in range(VISUAL_EMB_DIM)]
SEQUENCE_AUX_LAST_STEP_ONLY = 'last_step_only'
SEQUENCE_AUX_ALL_STEPS = 'all_steps'
VISUAL_COVERAGE_FAIL_FAST = True
TREE_MODEL_SCALER_CLIP_RANGE: tuple[float, float] | None = None
DEFAULT_LOB_MAX_AGE = '500ms'
DEFAULT_STAT_FEATURE_LIMIT = 20
EXTRA_STAT_FEATURE_CANDIDATES = [
    'rel_vol',
    'hour_sin',
    'hour_cos',
    'london_active',
    'ny_active',
    'overlap_active',
]
TREE_CLASS_WEIGHT_MAX = 2.5
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
    'path_outcome',
    'adverse_path_flag',
    'kalman_trend_label',
    'kalman_trend_strength',
    'kalman_price',
    'bias_label_raw',
    'effective_threshold_ticks',
    'effective_tp_long_ticks',
    'effective_tp_short_ticks',
    'effective_sl_ticks',
    'meta_trade_side',
    'meta_label',
    'meta_label_active',
    'meta_outcome_ticks',
    'soft_label',
    'soft_label_confidence',
    'soft_label_entropy',
    'soft_label_scenarios',
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
        'bias_label_raw',
        'effective_threshold_ticks',
        'effective_tp_long_ticks',
        'effective_tp_short_ticks',
        'effective_sl_ticks',
        'meta_trade_side',
        'meta_label',
        'meta_label_active',
        'meta_outcome_ticks',
        'soft_label',
        'soft_label_confidence',
        'soft_label_entropy',
        'soft_label_scenarios',
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


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        key = str(value).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _stat_feature_candidates() -> list[str]:
    return _ordered_unique(list(CATBOOST_ADVISOR_FEATURES) + list(EXTRA_STAT_FEATURE_CANDIDATES))


def _feature_available(df: pd.DataFrame, feature: str) -> bool:
    feature = str(feature)
    return feature in df.columns or f'{RAW_STAT_PREFIX}{feature}' in df.columns


def _load_ranked_selected_features(path: str) -> list[str]:
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        lines = [str(line).strip() for line in f.read().splitlines()]
    return [line for line in lines if line]


def _resolve_active_stat_features(
    df: pd.DataFrame,
    *,
    artifacts_dir: str,
    feature_limit: int = DEFAULT_STAT_FEATURE_LIMIT,
) -> tuple[list[str], dict]:
    limit = max(int(feature_limit), 1)
    candidate_pool = _stat_feature_candidates()
    available_candidates = [feat for feat in candidate_pool if _feature_available(df, feat)]
    selected_path = os.path.join(artifacts_dir, 'selected_features.txt')
    ranked_selected = _load_ranked_selected_features(selected_path)
    ranked_candidates = [
        feat for feat in ranked_selected
        if feat in candidate_pool and feat in available_candidates
    ]
    active = ranked_candidates[:limit] if ranked_candidates else available_candidates[:limit]
    if len(active) < limit:
        for feat in available_candidates:
            if feat not in active:
                active.append(feat)
            if len(active) >= limit:
                break
    active = _ordered_unique(active)[:limit]
    if not active:
        raise RuntimeError(
            '❌ No active statistical features available after resolving selected_features.txt '
            f'under {artifacts_dir}.'
        )
    info = {
        'feature_limit': int(limit),
        'selected_features_path': selected_path if os.path.exists(selected_path) else None,
        'selected_features_count': int(len(ranked_selected)),
        'selected_candidates_count': int(len(ranked_candidates)),
        'available_candidate_count': int(len(available_candidates)),
        'active_feature_count': int(len(active)),
        'selected_features_used': bool(len(ranked_candidates) > 0),
        'active_features': list(active),
    }
    return active, info


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


def _frame_symbol_counts(df: pd.DataFrame) -> tuple[str | None, dict[str, int]]:
    if 'symbol' in df.columns:
        col = 'symbol'
    elif 'instrument_id' in df.columns:
        col = 'instrument_id'
    else:
        return None, {}
    counts = df[col].fillna('UNKNOWN').astype(str).value_counts()
    return col, {str(key): int(value) for key, value in counts.items()}


def _assert_single_contract_df(df: pd.DataFrame, *, context: str) -> None:
    col, counts = _frame_symbol_counts(df)
    if col is None or len(counts) <= 1:
        return
    raise RuntimeError(
        "❌ Mixed-contract/symbol data is not allowed in V19 hardened training "
        f"| context={context} | column={col} | counts={counts}"
    )


def _safe_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def _binary_ece(y_true: np.ndarray, p_long: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    p_long = _safe_prob(p_long)
    if y_true.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, int(max(n_bins, 2)) + 1)
    bucket = np.digitize(p_long, bins[1:-1], right=False)
    ece = 0.0
    for b in range(len(bins) - 1):
        mask = bucket == b
        if not np.any(mask):
            continue
        conf = float(np.mean(p_long[mask]))
        acc = float(np.mean(y_true[mask] == 0))
        ece += abs(conf - acc) * (float(np.sum(mask)) / float(y_true.size))
    return float(ece)


def _fit_long_isotonic_calibrator(
    y_bias: np.ndarray,
    p_long: np.ndarray,
) -> tuple[IsotonicRegression | None, dict]:
    y_bias = np.asarray(y_bias, dtype=np.int32)
    p_long = _safe_prob(p_long)
    mask = np.isin(y_bias, [0, 1])
    if int(mask.sum()) < 32:
        return None, {
            'enabled': False,
            'reason': 'insufficient_directional_oof_rows',
            'rows': int(mask.sum()),
        }
    y_long = (y_bias[mask] == 0).astype(np.int32)
    if np.unique(y_long).size < 2:
        return None, {
            'enabled': False,
            'reason': 'single_class_directional_oof_rows',
            'rows': int(mask.sum()),
        }
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    calibrated = calibrator.fit_transform(p_long[mask], y_long)
    raw_probs = np.c_[1.0 - p_long[mask], p_long[mask]]
    cal_probs = np.c_[1.0 - calibrated, calibrated]
    report = {
        'enabled': True,
        'rows': int(mask.sum()),
        'raw_brier': float(np.mean((p_long[mask] - y_long) ** 2)),
        'calibrated_brier': float(np.mean((calibrated - y_long) ** 2)),
        'raw_ece': _binary_ece(y_bias[mask], p_long[mask]),
        'calibrated_ece': _binary_ece(y_bias[mask], calibrated),
        'raw_nll': float(log_loss(y_long, raw_probs, labels=[0, 1])),
        'calibrated_nll': float(log_loss(y_long, cal_probs, labels=[0, 1])),
    }
    return calibrator, report


def _apply_long_calibrator(calibrator: IsotonicRegression | None, probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float32)
    if calibrator is None or arr.ndim != 2 or arr.shape[1] < 2:
        return arr.astype(np.float32, copy=True)
    p_long = _safe_prob(arr[:, 0])
    cal_long = np.asarray(calibrator.transform(p_long), dtype=np.float64)
    cal_long = np.clip(cal_long, 1e-6, 1.0 - 1e-6)
    out = np.zeros_like(arr, dtype=np.float32)
    out[:, 0] = cal_long.astype(np.float32)
    out[:, 1] = (1.0 - cal_long).astype(np.float32)
    return out


def _calibration_metrics(
    y_bias: np.ndarray,
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
) -> dict:
    y_bias = np.asarray(y_bias, dtype=np.int32)
    raw_probs = np.asarray(raw_probs, dtype=np.float64)
    calibrated_probs = np.asarray(calibrated_probs, dtype=np.float64)
    mask = np.isin(y_bias, [0, 1])
    if int(mask.sum()) == 0:
        return {
            'rows': 0,
            'raw_brier': None,
            'calibrated_brier': None,
            'raw_ece': None,
            'calibrated_ece': None,
            'raw_nll': None,
            'calibrated_nll': None,
        }
    y_dir = y_bias[mask]
    raw = np.clip(raw_probs[mask], 1e-6, 1.0 - 1e-6)
    raw = raw / np.clip(raw.sum(axis=1, keepdims=True), 1e-9, None)
    cal = np.clip(calibrated_probs[mask], 1e-6, 1.0 - 1e-6)
    cal = cal / np.clip(cal.sum(axis=1, keepdims=True), 1e-9, None)
    y_long = (y_dir == 0).astype(np.float64)
    return {
        'rows': int(mask.sum()),
        'raw_brier': float(np.mean((raw[:, 0] - y_long) ** 2)),
        'calibrated_brier': float(np.mean((cal[:, 0] - y_long) ** 2)),
        'raw_ece': _binary_ece(y_dir, raw[:, 0]),
        'calibrated_ece': _binary_ece(y_dir, cal[:, 0]),
        'raw_nll': float(log_loss(y_dir, raw, labels=[0, 1])),
        'calibrated_nll': float(log_loss(y_dir, cal, labels=[0, 1])),
    }


def _fit_temperature_from_probs(y_bias: np.ndarray, probs: np.ndarray) -> tuple[float | None, dict]:
    y_bias = np.asarray(y_bias, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float64)
    mask = np.isin(y_bias, [0, 1])
    if int(mask.sum()) < 32 or probs.ndim != 2 or probs.shape[1] < 2:
        return None, {'enabled': False, 'reason': 'insufficient_validation_probs', 'rows': int(mask.sum())}
    y_dir = y_bias[mask]
    probs = np.clip(probs[mask], 1e-6, 1.0 - 1e-6)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-9, None)
    logits = np.log(probs)
    temperatures = np.linspace(0.5, 3.0, 51)
    best_t = None
    best_nll = None
    raw_nll = float(log_loss(y_dir, probs, labels=[0, 1]))
    for temp in temperatures:
        scaled = logits / float(temp)
        scaled -= scaled.max(axis=1, keepdims=True)
        exp_scaled = np.exp(scaled)
        cal_probs = exp_scaled / np.clip(exp_scaled.sum(axis=1, keepdims=True), 1e-9, None)
        nll = float(log_loss(y_dir, cal_probs, labels=[0, 1]))
        if best_nll is None or nll < best_nll:
            best_nll = nll
            best_t = float(temp)
    if best_t is None:
        return None, {'enabled': False, 'reason': 'temperature_search_failed', 'rows': int(mask.sum())}
    scaled = logits / best_t
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    cal_probs = exp_scaled / np.clip(exp_scaled.sum(axis=1, keepdims=True), 1e-9, None)
    return best_t, {
        'enabled': True,
        'rows': int(mask.sum()),
        'temperature': float(best_t),
        'raw_nll': raw_nll,
        'calibrated_nll': float(log_loss(y_dir, cal_probs, labels=[0, 1])),
        'raw_brier': float(np.mean((probs[:, 0] - (y_dir == 0).astype(np.float64)) ** 2)),
        'calibrated_brier': float(np.mean((cal_probs[:, 0] - (y_dir == 0).astype(np.float64)) ** 2)),
        'raw_ece': _binary_ece(y_dir, probs[:, 0]),
        'calibrated_ece': _binary_ece(y_dir, cal_probs[:, 0]),
    }


def _load_stage1_meta_feature_names(output_dir: str, meta_dim: int) -> list[str]:
    names_path = os.path.join(output_dir, 'meta_feature_names_v19.json')
    if os.path.exists(names_path):
        with open(names_path) as f:
            payload = json.load(f)
        names = payload.get('meta_features')
        if not isinstance(names, list) or len(names) != int(meta_dim):
            raise ValueError(
                f'❌ Stage1 meta feature names mismatch: expected {meta_dim} columns, got {names}'
            )
        infer_meta_feature_layout(names)
        return [str(col) for col in names]
    try:
        return resolve_meta_feature_names(meta_dim=int(meta_dim))
    except Exception as exc:
        raise ValueError(
            '❌ Unsupported stage1 meta feature surface: '
            f'actual_dim={int(meta_dim)} | supported_dims='
            f'[{len(META_FEATURE_NAMES)} => catboost+xgboost+regime, '
            f'{len(LEGACY_META_FEATURE_NAMES)} => legacy catboost-only]'
        ) from exc


def _meta_layout_label(meta_feature_names: list[str]) -> str:
    layout = infer_meta_feature_layout(list(meta_feature_names))
    base_models = [str(spec.get('name', 'unknown')) for spec in layout.get('base_models', [])]
    regime_dim = len(layout.get('regime_meta_cols', []))
    if base_models == ['catboost', 'xgboost'] and regime_dim == 7:
        return 'catboost+xgboost+regime'
    if base_models == ['catboost'] and regime_dim == 7:
        return 'legacy catboost-only'
    return '+'.join(base_models) + f'+regime({regime_dim})'


def _write_feature_coverage_drift_report(
    df: pd.DataFrame,
    *,
    stat_features: list[str] | None = None,
    split_time: pd.Timestamp | str | None,
    output_dir: str,
    protected_features: set[str] | None = None,
) -> str:
    protected_features = protected_features or set()
    stat_features = list(stat_features or CATBOOST_ADVISOR_FEATURES)
    ts = _time_series(df, 'ts_event')
    months = ts.dt.to_period('M').astype(str)
    raw_stat = _raw_stat_frame(df, stat_features)
    split_ts = _parse_optional_timestamp(split_time)
    train_mask = np.ones(len(df), dtype=bool) if split_ts is None else (ts < split_ts).to_numpy(dtype=bool)
    train_ref = raw_stat.loc[train_mask] if np.any(train_mask) else raw_stat

    report: dict[str, object] = {
        'generated_at': datetime.datetime.utcnow().replace(microsecond=0).isoformat() + 'Z',
        'split_time': None if split_ts is None else str(split_ts),
        'months': [],
        'dead_features_3m': [],
        'stat_features': stat_features,
    }
    dead_streak = {feat: 0 for feat in stat_features}
    for month in sorted(months.unique()):
        month_mask = (months == month).to_numpy(dtype=bool)
        month_frame = raw_stat.loc[month_mask]
        feature_stats = {}
        for feat in stat_features:
            vals = pd.to_numeric(month_frame[feat], errors='coerce')
            ref_vals = pd.to_numeric(train_ref[feat], errors='coerce')
            non_zero_rate = float((vals.fillna(0.0) != 0.0).mean()) if len(vals) else 0.0
            missing_rate = float(vals.isna().mean()) if len(vals) else 0.0
            ref_mean = float(ref_vals.fillna(0.0).mean()) if len(ref_vals) else 0.0
            ref_std = float(ref_vals.fillna(0.0).std(ddof=0)) if len(ref_vals) else 1.0
            ref_std = ref_std if abs(ref_std) > 1e-8 else 1.0
            obs_mean = float(vals.fillna(0.0).mean()) if len(vals) else 0.0
            obs_std = float(vals.fillna(0.0).std(ddof=0)) if len(vals) else 0.0
            psi = abs(obs_mean - ref_mean) / ref_std + abs(obs_std - ref_std) / ref_std
            feature_stats[feat] = {
                'non_zero_rate': round(non_zero_rate, 4),
                'missing_rate': round(missing_rate, 4),
                'psi': round(float(psi), 4),
                'protected': bool(feat in protected_features),
            }
            if feat not in protected_features and non_zero_rate < 0.01:
                dead_streak[feat] += 1
            else:
                dead_streak[feat] = 0
        report['months'].append({
            'month': str(month),
            'rows': int(month_mask.sum()),
            'features': feature_stats,
        })
    report['dead_features_3m'] = sorted([feat for feat, streak in dead_streak.items() if streak >= 3])
    path = os.path.join(output_dir, 'feature_coverage_drift_report.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    return path

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
    print(f"\n📥 قراءة artifact: {csv_path}")
    df = load_feature_artifact(csv_path)
    _assert_single_contract_df(df, context='load_training_csv')

    if 'bias_label' not in df.columns:
        raise ValueError("❌ 'bias_label' غير موجود — شغّل prepare_training_data.py --label_mode v19 أولاً")
    if 'ts_event' not in df.columns:
        raise ValueError("❌ 'ts_event' غير موجود — V19 يحتاج timestamps محفوظة في CSV")
    if 'label_end_ts' not in df.columns:
        raise ValueError(
            "❌ 'label_end_ts' غير موجود — V19 hardened training requires label end timestamps "
            "for chronological purging and leakage-safe validation."
        )

    df = _sanitize_df(df)
    stat_candidates = _stat_feature_candidates()
    raw_cols = [f'{RAW_STAT_PREFIX}{col}' for col in stat_candidates if f'{RAW_STAT_PREFIX}{col}' in df.columns]
    safe_nonraw_features = {'hour_sin', 'hour_cos', 'london_active', 'ny_active', 'overlap_active'}
    if len(raw_cols) < len(stat_candidates):
        missing = [
            col for col in stat_candidates
            if f'{RAW_STAT_PREFIX}{col}' not in df.columns and col not in safe_nonraw_features
        ]
        if missing:
            print(
                "  ⚠️ Missing raw stat columns for fold-clean scaling: "
                f"{missing}. Re-run prepare_training_data.py to unlock the strict anti-leakage path."
            )
    df = prepare_feature_frame(
        df,
        stat_features=stat_candidates + raw_cols,
        scaler_params=None,
        already_scaled=True,
        passthrough_cols=TRAINING_PASSTHROUGH_COLS,
        timestamp_cols=('ts_event', 'label_end_ts'),
    )
    ts_event = _time_series(df, 'ts_event')
    label_end = _time_series(df, 'label_end_ts', fallback='ts_event')
    if not ts_event.is_monotonic_increasing:
        print("  ⚠️ Input rows were not monotonic by ts_event — sorting chronologically before training")
        df = df.assign(ts_event=ts_event, label_end_ts=label_end).sort_values('ts_event').reset_index(drop=True)
        ts_event = _time_series(df, 'ts_event')
        label_end = _time_series(df, 'label_end_ts', fallback='ts_event')
    invalid_horizon = (label_end < ts_event).to_numpy(dtype=bool)
    if np.any(invalid_horizon):
        bad_rows = np.flatnonzero(invalid_horizon)[:5].tolist()
        raise ValueError(
            "❌ Found label_end_ts earlier than ts_event. "
            f"rows={int(np.sum(invalid_horizon))} sample_indices={bad_rows}"
        )
    print(
        "  🕒 Timestamp Range: "
        f"ts_event=[{ts_event.iloc[0]} → {ts_event.iloc[-1]}] | "
        f"label_end_ts=[{label_end.iloc[0]} → {label_end.iloc[-1]}] | "
        f"rows={len(df):,}"
    )
    print(f"  Shape: {df.shape}")
    print(f"  Labels: {df['bias_label'].value_counts().to_dict()}")
    return df


def build_event_training_view(
    df: pd.DataFrame,
    mode: str = TRAIN_MODE_EVENT_BINARY,
    quality_weight_strong: float = 2.0,
    quality_weight_weak: float = 1.0,
    train_frac: float = 0.80,
    split_time: pd.Timestamp | str | None = None,
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
    quality_term = np.where(
        event_df['signal_quality'].values.astype(np.int8) == 2,
        1.0,
        np.where(event_df['signal_quality'].values.astype(np.int8) == 1, 0.5, 0.0),
    ).astype(np.float32)
    score_split_ctx = _sequence_split_context(
        event_df,
        seq_len=SEQ_LEN,
        train_frac=train_frac,
        split_time=_parse_optional_timestamp(split_time),
    )
    score_train_mask = np.asarray(score_split_ctx['train_row_ok'], dtype=bool)
    if not np.any(score_train_mask):
        fallback_rows = int(min(max(score_split_ctx['split_idx'], 1), len(event_df)))
        if fallback_rows >= len(event_df) and len(event_df) > 1:
            fallback_rows = len(event_df) - 1
        score_train_mask = np.zeros(len(event_df), dtype=bool)
        score_train_mask[:max(fallback_rows, 1)] = True
        print(
            "  ⚠️ Event score normalization fallback: "
            f"no strict train rows via split guard; using prefix rows={int(np.sum(score_train_mask)):,}"
        )
    event_score_values = event_df.get('event_score')
    if event_score_values is None:
        event_score_values = pd.Series(0.0, index=event_df.index, dtype=np.float32)
    event_scores = pd.to_numeric(event_score_values, errors='coerce').fillna(0.0).astype(np.float32)
    score_fit_values = event_scores.loc[score_train_mask]
    score_min = float(score_fit_values.min()) if len(score_fit_values) else 0.0
    score_max = float(score_fit_values.max()) if len(score_fit_values) else score_min
    score_rng = score_max - score_min
    if score_rng > 1e-8:
        normalized_event_score = ((event_scores - score_min) / score_rng).clip(0.0, 1.0).astype(np.float32)
    else:
        normalized_event_score = pd.Series(np.zeros(len(event_df), dtype=np.float32), index=event_df.index)
    event_df['normalized_event_score'] = normalized_event_score.astype(np.float32)
    event_df['conf_target'] = np.clip(
        0.5 * quality_term + 0.5 * normalized_event_score.values.astype(np.float32),
        0.0,
        1.0,
    ).astype(np.float32)
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
        'score_fit_rows': int(np.sum(score_train_mask)),
        'score_fit_min': float(score_min),
        'score_fit_max': float(score_max),
        'score_fit_split_time': str(score_split_ctx['split_time']),
        'conf_target_mean': float(event_df['conf_target'].mean()) if len(event_df) else 0.0,
        'conf_target_std': float(event_df['conf_target'].std(ddof=0)) if len(event_df) else 0.0,
    }
    print(
        "  ✅ Event Training View: "
        f"{info['rows_event_directional']:,}/{info['rows_full']:,} rows "
        f"({info['event_rate_full']:.1%}) via {event_col} "
        f"| raw_event={info['raw_event_rate_full']:.1%} "
        f"| bias={info['bias_counts']} | quality={info['quality_counts']} "
        f"| score_fit_rows={info['score_fit_rows']:,} "
        f"| score_fit_range=[{info['score_fit_min']:.4f}, {info['score_fit_max']:.4f}] "
        f"| split={info['score_fit_split_time']}"
    )
    return event_df, info


def _raw_feature_name(col: str) -> str:
    return f'{RAW_STAT_PREFIX}{col}'


def _raw_stat_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # Hard guard: passthrough/meta columns such as forward_return may exist in
    # the CSV for analysis/backtesting, but they must never enter model inputs.
    _assert_no_forbidden_model_inputs(list(cols))
    ts_event = _time_series(df, 'ts_event') if 'ts_event' in df.columns else None
    hour_values = None if ts_event is None else ts_event.dt.hour.to_numpy(dtype=np.float32)
    data = {}
    for col in cols:
        raw_col = _raw_feature_name(col)
        src = raw_col if raw_col in df.columns else col
        if src in df.columns:
            series = pd.to_numeric(df[src], errors='coerce').fillna(0.0).astype(np.float32)
        elif hour_values is not None and col == 'hour_sin':
            series = pd.Series(np.sin(2.0 * np.pi * hour_values / 24.0).astype(np.float32), index=df.index)
        elif hour_values is not None and col == 'hour_cos':
            series = pd.Series(np.cos(2.0 * np.pi * hour_values / 24.0).astype(np.float32), index=df.index)
        elif hour_values is not None and col == 'london_active':
            series = pd.Series(((hour_values >= 7.0) & (hour_values < 16.0)).astype(np.float32), index=df.index)
        elif hour_values is not None and col == 'ny_active':
            series = pd.Series(((hour_values >= 13.0) & (hour_values < 22.0)).astype(np.float32), index=df.index)
        elif hour_values is not None and col == 'overlap_active':
            series = pd.Series(((hour_values >= 13.0) & (hour_values < 16.0)).astype(np.float32), index=df.index)
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
        params[col] = fit_numeric_scaler_param(s, robust_iqr_min=ROBUST_IQR_MIN)
    return params


def _apply_scaler_to_stat_frame(
    frame: pd.DataFrame,
    scaler_params: dict,
    clip_range: tuple[float, float] | None = (-10.0, 10.0),
) -> pd.DataFrame:
    scaled = apply_scaler_params_to_frame(frame, scaler_params or {}, clip_range=clip_range)
    return scaled[frame.columns].astype(np.float32)


def _build_scaled_stat_matrix(
    df: pd.DataFrame,
    cols: list[str],
    scaler_params: dict,
    clip_range: tuple[float, float] | None = (-10.0, 10.0),
) -> np.ndarray:
    raw_frame = _raw_stat_frame(df, cols)
    scaled = _apply_scaler_to_stat_frame(raw_frame, scaler_params, clip_range=clip_range)
    return scaled[cols].values.astype(np.float32)


def _adaptive_tree_depth(n_features: int) -> int:
    n_features = max(int(n_features), 0)
    if n_features < 50:
        return 4
    if n_features < 100:
        return 6
    return 7


def _compute_binary_class_weights(y_bias: np.ndarray, max_weight: float = TREE_CLASS_WEIGHT_MAX) -> list[float] | None:
    y_bias = np.asarray(y_bias, dtype=np.int32)
    counts = np.bincount(y_bias[(y_bias >= 0) & (y_bias < 2)], minlength=2)[:2]
    if counts.size < 2 or np.any(counts <= 0):
        return None
    total = float(np.sum(counts))
    raw_weights = [total / (2.0 * float(count)) for count in counts]
    weights = [float(np.clip(w, 1.0, max_weight)) for w in raw_weights]
    return weights


def _fold_scaler_diagnostics(scaler_params: dict | None) -> dict:
    scaler_params = scaler_params or {}
    type_counts: dict[str, int] = {}
    for params in scaler_params.values():
        scaler_type = str((params or {}).get('type', 'missing'))
        type_counts[scaler_type] = int(type_counts.get(scaler_type, 0) + 1)
    total = int(sum(type_counts.values()))
    zero_count = int(type_counts.get('zero', 0))
    return {
        'feature_count': total,
        'type_counts': type_counts,
        'zero_count': zero_count,
        'zero_pct': float(zero_count / max(total, 1)),
    }


def _stabilize_fold_scaler(
    fold_scaler: dict | None,
    inference_scaler_params: dict | None,
) -> tuple[dict, dict]:
    stabilized = {str(col): dict(params or {}) for col, params in (fold_scaler or {}).items()}
    inference_scaler_params = inference_scaler_params or {}
    fallback_used: list[str] = []
    for col, params in list(stabilized.items()):
        if str((params or {}).get('type', '')) != 'zero':
            continue
        fallback = inference_scaler_params.get(col)
        if isinstance(fallback, dict) and str(fallback.get('type', '')) != 'zero':
            stabilized[col] = dict(fallback)
            fallback_used.append(str(col))
    diagnostics = _fold_scaler_diagnostics(stabilized)
    diagnostics['zero_fallback_features'] = fallback_used
    diagnostics['zero_fallback_count'] = int(len(fallback_used))
    return stabilized, diagnostics


def _adaptive_tree_early_stopping_rounds(train_rows: int, has_eval_set: bool) -> int | None:
    if not has_eval_set:
        return None
    train_rows = max(int(train_rows), 0)
    if train_rows < 500:
        return 30
    if train_rows < 2_000:
        return 50
    if train_rows < 10_000:
        return 75
    return 100


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
    primary = None
    if col in df.columns:
        primary = pd.to_datetime(df[col], utc=True, errors='coerce').dt.tz_localize(None)
    fallback_series = None
    if fallback and fallback in df.columns:
        fallback_series = pd.to_datetime(df[fallback], utc=True, errors='coerce').dt.tz_localize(None)

    if primary is None and fallback_series is None:
        raise ValueError(
            f"❌ Missing required timestamp column '{col}'"
            + (f" (fallback '{fallback}' also missing)." if fallback else ".")
        )

    if primary is None:
        s = fallback_series.copy()
    elif fallback_series is None:
        s = primary.copy()
    else:
        s = primary.where(primary.notna(), fallback_series)

    invalid = s.isna()
    if invalid.any():
        sample = np.flatnonzero(invalid.to_numpy(dtype=bool))[:5].tolist()
        raise ValueError(
            f"❌ Invalid timestamps in '{col}'"
            + (f" after fallback='{fallback}'" if fallback else "")
            + f": rows={int(invalid.sum())} sample_indices={sample}"
        )
    return s.reset_index(drop=True)


def _parse_optional_timestamp(value) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors='coerce')
    if pd.isna(ts):
        return None
    return ts.tz_localize(None)


def _chronological_holdout_indices(n_rows: int, holdout_frac: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    n_rows = max(int(n_rows), 0)
    if n_rows <= 1:
        idx = np.arange(n_rows, dtype=np.int32)
        return idx, np.array([], dtype=np.int32)
    holdout_rows = int(max(1, round(n_rows * float(holdout_frac))))
    holdout_rows = min(holdout_rows, n_rows - 1)
    split_idx = n_rows - holdout_rows
    train_idx = np.arange(split_idx, dtype=np.int32)
    holdout_idx = np.arange(split_idx, n_rows, dtype=np.int32)
    return train_idx, holdout_idx


def _regime_priors_from_meta(regime_meta: np.ndarray, covered_mask: np.ndarray) -> np.ndarray:
    regime_meta = np.asarray(regime_meta, dtype=np.float32)
    covered_mask = np.asarray(covered_mask, dtype=bool).reshape(-1)
    out = np.zeros((regime_meta.shape[0], regime_meta.shape[1]), dtype=np.float32)
    if regime_meta.ndim != 2 or regime_meta.shape[1] == 0:
        return out
    if np.any(covered_mask):
        priors = regime_meta[covered_mask].mean(axis=0)
    else:
        priors = np.zeros(regime_meta.shape[1], dtype=np.float32)
    if regime_meta.shape[1] >= N_CLUSTERS:
        one_hot_block = np.clip(priors[:N_CLUSTERS], 0.0, None)
        if float(one_hot_block.sum()) <= 0.0:
            one_hot_block = np.ones(N_CLUSTERS, dtype=np.float32) / float(N_CLUSTERS)
        else:
            one_hot_block = one_hot_block / float(one_hot_block.sum())
        priors[:N_CLUSTERS] = one_hot_block.astype(np.float32)
    return np.repeat(priors.reshape(1, -1), regime_meta.shape[0], axis=0).astype(np.float32)


def _build_row_time_mask(
    df: pd.DataFrame,
    *,
    start_ts: pd.Timestamp | str | None = None,
    end_ts: pd.Timestamp | str | None = None,
) -> np.ndarray:
    if len(df) == 0:
        return np.array([], dtype=bool)
    ts = _time_series(df, 'ts_event')
    mask = np.ones(len(df), dtype=bool)
    start = _parse_optional_timestamp(start_ts)
    end = _parse_optional_timestamp(end_ts)
    if start is not None:
        mask &= ts.values >= start.to_datetime64()
    if end is not None:
        mask &= ts.values < end.to_datetime64()
    return mask


def _resolve_training_window(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.80,
    split_time: pd.Timestamp | str | None = None,
    train_days: float | None = None,
    backtest_days: float | None = None,
    window_end: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, dict]:
    if len(df) == 0:
        return df.copy(), {
            'mode': 'empty',
            'split_time': None,
            'split_source': 'empty',
            'train_start_time': None,
            'holdout_start_time': None,
            'holdout_end_time_exclusive': None,
            'ts_min': None,
            'ts_max': None,
            'rows_total': 0,
            'train_rows': 0,
            'holdout_rows': 0,
            'requested_train_days': None,
            'requested_backtest_days': None,
        }

    ts_all = _time_series(df, 'ts_event')
    source_ts_min = ts_all.min()
    source_ts_max = ts_all.max()
    end_exclusive = _parse_optional_timestamp(window_end)
    if end_exclusive is None:
        end_exclusive = source_ts_max + pd.Timedelta(microseconds=1)

    requested_train_days = float(train_days) if train_days is not None else None
    requested_backtest_days = float(backtest_days) if backtest_days is not None else None
    explicit_split = _parse_optional_timestamp(split_time)

    filtered = df.copy().reset_index(drop=True)
    mode = 'full_dataset'
    split_source = 'train_frac'
    train_start_time = None

    if (
        requested_train_days is not None
        or requested_backtest_days is not None
        or window_end is not None
    ):
        if explicit_split is None:
            if requested_backtest_days is None:
                raise ValueError(
                    '❌ backtest_days مطلوب عند استخدام train_days/window_end بدون split_time صريح.'
                )
            explicit_split = end_exclusive - pd.Timedelta(days=requested_backtest_days)
            split_source = 'backtest_days'
        else:
            split_source = 'explicit_time'

        if requested_train_days is not None:
            train_start_time = explicit_split - pd.Timedelta(days=requested_train_days)

        row_mask = _build_row_time_mask(df, start_ts=train_start_time, end_ts=end_exclusive)
        filtered = df.loc[row_mask].reset_index(drop=True)
        ts_filtered = ts_all.loc[row_mask].reset_index(drop=True)
        mode = 'fixed_day_window'

        if filtered.empty:
            raise RuntimeError('❌ النافذة الزمنية المطلوبة للتدريب/الباك تست فارغة.')
        if not (ts_filtered < explicit_split).any():
            raise RuntimeError('❌ لا توجد صفوف تدريب قبل split_time داخل النافذة المطلوبة.')
        if not (ts_filtered >= explicit_split).any():
            raise RuntimeError('❌ لا توجد صفوف holdout بعد split_time داخل النافذة المطلوبة.')
    elif explicit_split is not None:
        split_source = 'artifact_split_time'

    split_ctx = _sequence_split_context(
        filtered,
        seq_len=SEQ_LEN,
        train_frac=train_frac,
        split_time=explicit_split,
    )
    ts_filtered = _time_series(filtered, 'ts_event')
    train_start_effective = train_start_time if train_start_time is not None else ts_filtered.min()

    info = {
        'mode': mode,
        'split_time': str(split_ctx['split_time']),
        'split_source': split_source if explicit_split is not None else str(split_ctx.get('split_source', 'train_frac')),
        'train_start_time': None if pd.isna(train_start_effective) else str(train_start_effective),
        'holdout_start_time': str(split_ctx['split_time']),
        'holdout_end_time_exclusive': None if end_exclusive is None else str(end_exclusive),
        'ts_min': None if pd.isna(ts_filtered.min()) else str(ts_filtered.min()),
        'ts_max': None if pd.isna(ts_filtered.max()) else str(ts_filtered.max()),
        'rows_total': int(len(filtered)),
        'train_rows': int(np.sum(split_ctx['train_row_ok'])),
        'holdout_rows': int(np.sum(split_ctx['val_row_ok'])),
        'requested_train_days': requested_train_days,
        'requested_backtest_days': requested_backtest_days,
        'source_ts_min': None if pd.isna(source_ts_min) else str(source_ts_min),
        'source_ts_max': None if pd.isna(source_ts_max) else str(source_ts_max),
    }
    return filtered, info


def _sequence_split_context(
    df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    train_frac: float = 0.80,
    split_time: pd.Timestamp | None = None,
) -> dict:
    n = len(df)
    split_idx = max(seq_len * 2, int(n * train_frac))
    split_idx = min(max(split_idx, seq_len), n)
    ts_event = _time_series(df, 'ts_event')
    split_source = 'train_frac'
    if split_time is None:
        split_time = ts_event.iloc[min(split_idx, n - 1)]
    else:
        split_time = pd.to_datetime(split_time, utc=True, errors='coerce')
        if pd.isna(split_time):
            split_time = ts_event.iloc[min(split_idx, n - 1)]
        else:
            split_time = split_time.tz_localize(None)
            split_source = 'explicit_time'
            split_idx = int(np.searchsorted(ts_event.values.astype('datetime64[ns]'), split_time.to_datetime64(), side='left'))
            split_idx = min(max(split_idx, seq_len), n)
    label_end = _time_series(df, 'label_end_ts', fallback='ts_event')
    if split_source == 'explicit_time':
        time_ok = label_end <= split_time
    else:
        time_ok = label_end < split_time
    train_row_ok = (np.arange(n) < split_idx) & time_ok
    val_row_ok = np.arange(n) >= split_idx
    return {
        'split_idx': int(split_idx),
        'split_time': split_time,
        'split_source': split_source,
        'label_end': label_end,
        'train_row_ok': train_row_ok,
        'val_row_ok': val_row_ok,
    }


def build_inference_scaler_params(
    df: pd.DataFrame,
    cols: list[str],
    train_frac: float = 0.80,
    split_time: pd.Timestamp | str | None = None,
) -> tuple[dict, dict]:
    split_ctx = _sequence_split_context(df, seq_len=SEQ_LEN, train_frac=train_frac, split_time=split_time)
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
        'split_source': str(split_ctx.get('split_source', 'train_frac')),
    }
    return scaler_params, info


def _save_scaler_params(output_dir: str, scaler_params: dict) -> str:
    path = os.path.join(output_dir, 'scaler_params.json')
    with open(path, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    return path


def copy_inference_artifacts(csv_path: str, output_dir: str) -> dict:
    copied = {}
    src_dir = resolve_artifact_root(csv_path)
    for name in (
        'selected_features.txt',
        'refinery_report.txt',
        'lob_tensor_timestamps.npy',
        'artifact_manifest.json',
        'refinery_split.json',
        'lob_build_meta.json',
        'final_feature_shards.json',
    ):
        src = os.path.join(src_dir, name)
        dst = os.path.join(output_dir, name)
        if os.path.exists(src):
            if os.path.abspath(src) == os.path.abspath(dst):
                copied[name] = dst
                continue
            shutil.copy2(src, dst)
            copied[name] = dst
    return copied


def _resolve_default_lob_paths(csv_path: str) -> tuple[str | None, str | None]:
    try:
        src_dir = resolve_artifact_root(csv_path)
    except Exception:
        src_dir = os.path.dirname(os.path.abspath(csv_path))
    lob_path = os.path.join(src_dir, 'lob_tensors.npy')
    lob_ts_path = os.path.join(src_dir, 'lob_tensor_timestamps.npy')
    return (
        lob_path if os.path.exists(lob_path) else None,
        lob_ts_path if os.path.exists(lob_ts_path) else None,
    )


def _load_source_refinery_contract(csv_path: str) -> dict:
    src_dir = resolve_artifact_root(csv_path)
    manifest_path = os.path.join(src_dir, 'artifact_manifest.json')
    split_path = os.path.join(src_dir, 'refinery_split.json')
    out = {
        'source_csv': os.path.abspath(src_dir),
        'dataset_id': None,
        'schema_version': None,
        'label_mode': None,
        'split_time': None,
        'manifest_path': manifest_path if os.path.exists(manifest_path) else None,
    }
    manifest = load_artifact_manifest(csv_path) if os.path.exists(manifest_path) else {}
    if manifest:
        try:
            extra = manifest.get('extra', {}) or {}
            out['dataset_id'] = extra.get('dataset_id')
            out['schema_version'] = extra.get('schema_version')
            out['label_mode'] = extra.get('label_mode')
            split_meta = extra.get('split_meta', {}) or {}
            out['split_time'] = split_meta.get('split_time')
        except Exception:
            pass
    if out['split_time'] is None and os.path.exists(split_path):
        try:
            with open(split_path) as f:
                split_meta = json.load(f)
            out['split_time'] = split_meta.get('split_time')
        except Exception:
            pass
    return out


def _resolve_meta_learner_profile(
    train_sequences: int,
    visual_seq_coverage: float,
) -> dict:
    train_sequences = int(max(train_sequences, 0))
    visual_seq_coverage = float(np.clip(visual_seq_coverage, 0.0, 1.0))
    if train_sequences < 2500 or visual_seq_coverage < 0.85:
        return {
            'name': 'compact',
            'lstm_units_1': 48,
            'lstm_units_2': 24,
            'visual_dropout_rate': 0.35,
            'confidence_threshold': 0.62,
            'force_rebuild': True,
        }
    return {
        'name': 'standard',
        'lstm_units_1': 128,
        'lstm_units_2': 64,
        'visual_dropout_rate': 0.25,
        'confidence_threshold': 0.65,
        'force_rebuild': False,
    }


def build_time_splits(
    df: pd.DataFrame,
    n_folds: int = 6,
    test_size: float = 0.10,
    embargo_pct: float = 0.02,
    min_train_pct: float = 0.20,
    embargo_min_pct: float | None = None,
    embargo_horizon_quantile: float = 0.95,
):
    n = len(df)
    requested_n_folds = int(max(n_folds, 1))
    t0 = _time_series(df, 'ts_event')
    t1 = _time_series(df, 'label_end_ts', fallback='ts_event')
    directional_h = pd.to_numeric(
        df.loc[df['bias_label'].isin([0, 1]), 'label_horizon_steps']
        if 'bias_label' in df.columns and 'label_horizon_steps' in df.columns
        else pd.Series(dtype=np.float64),
        errors='coerce',
    ).dropna()
    dynamic_embargo_rows = max(
        int(np.ceil(n * float(embargo_min_pct if embargo_min_pct is not None else embargo_pct))),
        int(np.ceil(np.percentile(directional_h, float(embargo_horizon_quantile) * 100.0))) if len(directional_h) else 0,
    )
    effective_embargo_pct = float(dynamic_embargo_rows / max(n, 1))
    # Small directional-event datasets become unstable when split into too many tiny
    # walk-forward test blocks. Cap the requested fold count so each test block
    # stays meaningfully sized before walk_forward_expanding applies its own logic.
    test_n = max(1, int(n * test_size))
    min_train = max(int(n * min_train_pct), 200)
    min_train = min(min_train, max(test_n + 50, n - test_n))
    tail_n = max(0, n - min_train)
    min_desired_test_rows = 500 if n >= 3000 else 350
    adaptive_fold_cap = max(1, int(tail_n // max(min_desired_test_rows, 1))) if tail_n > 0 else 1
    effective_requested_folds = max(3, min(requested_n_folds, adaptive_fold_cap)) if tail_n > 0 else 1
    if effective_requested_folds < requested_n_folds:
        print(
            "  ⚠️ Adaptive fold reduction: "
            f"requested={requested_n_folds} → effective={effective_requested_folds} "
            f"for rows={n:,} to avoid tiny test folds."
        )
    splits = list(
        walk_forward_expanding(
            n,
            n_folds=effective_requested_folds,
            test_size=test_size,
            embargo_pct=effective_embargo_pct,
            t0=t0,
            t1=t1,
            min_train_pct=min_train_pct,
        )
    )
    if not splits:
        raise RuntimeError('❌ تعذر بناء time splits صالحة لـ V19')
    return splits, t0, t1, {
        'dynamic_embargo_rows': int(dynamic_embargo_rows),
        'effective_embargo_pct': float(effective_embargo_pct),
        'embargo_horizon_quantile': float(embargo_horizon_quantile),
        'requested_n_folds': int(requested_n_folds),
        'effective_requested_n_folds': int(effective_requested_folds),
    }


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
    stat_features: list[str] | None = None,
    splits=None,
    n_folds: int = 6,
    test_size: float = 0.10,
    embargo_pct: float = 0.02,
    min_train_pct: float = 0.20,
    t0: pd.Series | None = None,
    t1: pd.Series | None = None,
    inference_scaler_params: dict | None = None,
    catboost_device: str = 'auto',
    include_xgboost: bool = False,
    quality_weight_strong: float = 2.0,
    quality_weight_weak: float = 1.0,
    cost_config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    stage1_label = 'CatBoost + Regime Meta-Features'
    if include_xgboost:
        stage1_label = 'CatBoost + XGBoost + Regime Meta-Features'
    print("\n" + "═" * 65)
    print(f"🐱 STAGE 1 — V19 OOF {stage1_label}")
    print("═" * 65)
    if not inference_scaler_params:
        raise RuntimeError(
            '❌ inference_scaler_params is required for stage1_oof_meta. '
            'Refusing to fall back to a full-data scaler.'
        )

    n = len(df)
    stat_features = list(stat_features or CATBOOST_ADVISOR_FEATURES)
    raw_stat = _raw_stat_frame(df, stat_features)
    y = df['bias_label'].fillna(1).astype(np.int32).values
    if splits is None:
        splits, t0, t1, _ = build_time_splits(
            df,
            n_folds=n_folds,
            test_size=test_size,
            embargo_pct=embargo_pct,
            min_train_pct=min_train_pct,
        )
    elif t0 is None or t1 is None:
        _, t0, t1, _ = build_time_splits(
            df,
            n_folds=n_folds,
            test_size=test_size,
            embargo_pct=embargo_pct,
            min_train_pct=min_train_pct,
        )

    print(f"  Splits: {len(splits)} | Rows: {n:,}")
    cb_task_type, cb_devices = _resolve_catboost_device(catboost_device)
    print(f"  CatBoost device: {cb_task_type}")
    print(f"  Class distribution: {pd.Series(y).value_counts().sort_index().to_dict()}")

    priors = np.bincount(y, minlength=N_CB_PROBS).astype(np.float32)
    priors = priors / max(priors.sum(), 1.0)
    meta_feature_names = resolve_meta_feature_names(include_xgboost=include_xgboost)
    meta_layout = infer_meta_feature_layout(meta_feature_names)
    print(
        "  Meta Layout Guard: "
        f"base_models={[spec['name'] for spec in meta_layout['base_models']]} "
        f"| base_prob_dim={meta_layout['base_prob_dim']} "
        f"| regime_dim={len(meta_layout['regime_meta_cols'])}"
    )

    if not CB_AVAILABLE:
        raise RuntimeError(
            "❌ CatBoost غير مثبّت. هذه المرحلة لم تتدرب فعليًا.\n"
            "ثبّت الحزمة داخل البيئة الحالية ثم أعد التشغيل:\n"
            "pip install -r requirements.txt"
        )
    if include_xgboost and not XGB_AVAILABLE:
        raise RuntimeError(
            "❌ XGBoost غير متاح بينما include_xgboost=True.\n"
            f"سبب التحميل: {XGB_IMPORT_ERROR}\n"
            "أصلح البيئة أو عطّل XGBoost ثم أعد التشغيل:\n"
            "pip install -r requirements.txt"
        )

    regime_model_type = 'hmm' if HMM_AVAILABLE else 'rules'
    regime_tmp_dir = tempfile.mkdtemp(prefix='_oof_regime_tmp_', dir=output_dir)
    os.makedirs(regime_tmp_dir, exist_ok=True)
    tree_depth = _adaptive_tree_depth(len(stat_features))

    def _weighted_quality_weights(indices: np.ndarray, class_weights: list[float] | None) -> np.ndarray:
        weights = _quality_sample_weights(
            df.iloc[indices],
            strong_weight=quality_weight_strong,
            weak_weight=quality_weight_weak,
        ).astype(np.float32)
        if class_weights is not None:
            class_mult = np.asarray(class_weights, dtype=np.float32)
            weights = weights * class_mult[np.clip(y[indices], 0, len(class_mult) - 1)]
        return weights.astype(np.float32)

    def _cb_predict(train_idx, test_idx, fold_no):
        inner_train, inner_val = _build_inner_time_split(train_idx, t0, t1, embargo_pct)
        fit_idx = inner_train if inner_train is not None else train_idx
        early_stopping_rounds = _adaptive_tree_early_stopping_rounds(len(fit_idx), inner_val is not None)
        fit_ts = _time_series(df.iloc[fit_idx].reset_index(drop=True), 'ts_event')
        test_ts = _time_series(df.iloc[test_idx].reset_index(drop=True), 'ts_event')
        fit_counts = pd.Series(y[fit_idx]).value_counts().sort_index().to_dict()
        test_counts = pd.Series(y[test_idx]).value_counts().sort_index().to_dict()
        print(
            f"    Fold {fold_no}: fit={len(fit_idx):,} test={len(test_idx):,} "
            f"| fit_ts=[{fit_ts.iloc[0]} → {fit_ts.iloc[-1]}] "
            f"| test_ts=[{test_ts.iloc[0]} → {test_ts.iloc[-1]}] "
            f"| y_fit={fit_counts} | y_test={test_counts}"
        )
        if len(np.unique(y[fit_idx])) < 2:
            print(
                f"      CatBoost Fold {fold_no}: single-class train labels {sorted(np.unique(y[fit_idx]).tolist())} "
                "| using class priors"
            )
            return np.repeat(priors.reshape(1, -1), len(test_idx), axis=0).astype(np.float32), {
                'directional_precision': None,
                'directional_recall': None,
                'directional_f1': None,
                'mode': 'priors_only_single_class_train',
            }
        fold_scaler_raw = _fit_scaler_params_from_frame(raw_stat.iloc[fit_idx])
        fold_scaler, scaler_diag = _stabilize_fold_scaler(fold_scaler_raw, inference_scaler_params)
        if float(scaler_diag.get('zero_pct', 0.0)) > 0.30:
            print(
                f"      CatBoost Fold {fold_no}: degraded scaler (zero_pct={float(scaler_diag.get('zero_pct', 0.0)):.1%}) "
                "| using priors"
            )
            return np.repeat(priors.reshape(1, -1), len(test_idx), axis=0).astype(np.float32), {
                'directional_precision': None,
                'directional_recall': None,
                'directional_f1': None,
                'mode': 'priors_only_degraded_scaler',
                'fold_scaler_diagnostics': scaler_diag,
            }

        X_fit = _apply_scaler_to_stat_frame(
            raw_stat.iloc[fit_idx],
            fold_scaler,
            clip_range=TREE_MODEL_SCALER_CLIP_RANGE,
        ).values.astype(np.float32)
        X_test = _apply_scaler_to_stat_frame(
            raw_stat.iloc[test_idx],
            fold_scaler,
            clip_range=TREE_MODEL_SCALER_CLIP_RANGE,
        ).values.astype(np.float32)
        print(
            f"      CatBoost Shapes[{fold_no}]: fit={X_fit.shape} test={X_test.shape} "
            f"| scaler_fit_rows={len(fit_idx):,}"
        )

        class_weights = _compute_binary_class_weights(y[fit_idx])
        sw = _weighted_quality_weights(fit_idx, class_weights)
        model = CatBoostClassifier(
            iterations=1000,
            depth=tree_depth,
            learning_rate=0.01,
            l2_leaf_reg=3.0,
            bootstrap_type='Bernoulli',
            subsample=0.80,
            rsm=0.70,
            loss_function='Logloss',
            eval_metric='Logloss',
            early_stopping_rounds=early_stopping_rounds,
            use_best_model=inner_val is not None,
            verbose=50,
            random_seed=42 + fold_no,
            task_type=cb_task_type,
            devices=cb_devices,
        )
        tr_pool = Pool(X_fit, y[fit_idx], weight=sw, feature_names=stat_features)
        eval_set = None
        X_val = None
        if inner_val is not None:
            X_val = _apply_scaler_to_stat_frame(
                raw_stat.iloc[inner_val],
                fold_scaler,
                clip_range=TREE_MODEL_SCALER_CLIP_RANGE,
            ).values.astype(np.float32)
            eval_set = Pool(X_val, y[inner_val], feature_names=stat_features)
        model.fit(tr_pool, eval_set=eval_set, plot=False)
        present_classes = getattr(model, 'classes_', np.unique(y[fit_idx]))
        raw_preds = align_probability_columns(
            model.predict_proba(X_test),
            N_CB_PROBS,
            classes=present_classes,
        )
        calibrator = None
        calibrator_report = {'enabled': False, 'reason': 'no_inner_validation'}
        if X_val is not None:
            val_raw = align_probability_columns(
                model.predict_proba(X_val),
                N_CB_PROBS,
                classes=present_classes,
            )
            calibrator, calibrator_report = _fit_long_isotonic_calibrator(y[inner_val], val_raw[:, 0])
        preds = _apply_long_calibrator(calibrator, raw_preds)
        test_metrics = _calibration_metrics(y[test_idx], raw_preds, preds)
        pred_labels = np.argmax(preds, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y[test_idx],
            pred_labels,
            labels=[0, 1],
            average='macro',
            zero_division=0,
        )
        cb_best_iteration = None
        if inner_val is not None:
            best_iter = getattr(model, 'get_best_iteration', lambda: None)()
            cb_best_iteration = None if best_iter is None else int(best_iter)
        return preds, {
            'directional_precision': float(precision),
            'directional_recall': float(recall),
            'directional_f1': float(f1),
            'classes': [int(cls) for cls in np.asarray(present_classes).reshape(-1).tolist()],
            'best_iteration': cb_best_iteration,
            'early_stopping_rounds': early_stopping_rounds,
            'class_weights': class_weights,
            'fold_scaler_diagnostics': scaler_diag,
            'calibration': {
                'inner_validation': calibrator_report,
                'outer_test': test_metrics,
            },
        }

    oof_probs_raw, prob_covered, prob_reports = run_sequential_oof(n, N_CB_PROBS, splits, _cb_predict)
    oof_probs = fill_uncovered_probabilities(oof_probs_raw, prob_covered, priors=priors)

    xgb_probs = np.zeros((n, 0), dtype=np.float32)
    xgb_covered = np.ones(n, dtype=bool)
    xgb_reports: list[dict] = []
    if include_xgboost:
        def _xgb_predict(train_idx, test_idx, fold_no):
            inner_train, inner_val = _build_inner_time_split(train_idx, t0, t1, embargo_pct)
            fit_idx = inner_train if inner_train is not None else train_idx
            early_stopping_rounds = _adaptive_tree_early_stopping_rounds(len(fit_idx), inner_val is not None)
            fit_ts = _time_series(df.iloc[fit_idx].reset_index(drop=True), 'ts_event')
            test_ts = _time_series(df.iloc[test_idx].reset_index(drop=True), 'ts_event')
            fit_counts = pd.Series(y[fit_idx]).value_counts().sort_index().to_dict()
            test_counts = pd.Series(y[test_idx]).value_counts().sort_index().to_dict()
            print(
                f"    XGB Fold {fold_no}: fit={len(fit_idx):,} test={len(test_idx):,} "
                f"| fit_ts=[{fit_ts.iloc[0]} → {fit_ts.iloc[-1]}] "
                f"| test_ts=[{test_ts.iloc[0]} → {test_ts.iloc[-1]}] "
                f"| y_fit={fit_counts} | y_test={test_counts}"
            )
            if len(np.unique(y[fit_idx])) < 2:
                print(
                    f"      XGBoost Fold {fold_no}: single-class train labels {sorted(np.unique(y[fit_idx]).tolist())} "
                    "| using class priors"
                )
                return np.repeat(priors.reshape(1, -1), len(test_idx), axis=0).astype(np.float32), {
                    'directional_precision': None,
                    'directional_recall': None,
                    'directional_f1': None,
                    'mode': 'priors_only_single_class_train',
                }
            fold_scaler_raw = _fit_scaler_params_from_frame(raw_stat.iloc[fit_idx])
            fold_scaler, scaler_diag = _stabilize_fold_scaler(fold_scaler_raw, inference_scaler_params)
            if float(scaler_diag.get('zero_pct', 0.0)) > 0.30:
                print(
                    f"      XGBoost Fold {fold_no}: degraded scaler (zero_pct={float(scaler_diag.get('zero_pct', 0.0)):.1%}) "
                    "| using priors"
                )
                return np.repeat(priors.reshape(1, -1), len(test_idx), axis=0).astype(np.float32), {
                    'directional_precision': None,
                    'directional_recall': None,
                    'directional_f1': None,
                    'mode': 'priors_only_degraded_scaler',
                    'fold_scaler_diagnostics': scaler_diag,
                }
            X_fit = _apply_scaler_to_stat_frame(
                raw_stat.iloc[fit_idx],
                fold_scaler,
                clip_range=TREE_MODEL_SCALER_CLIP_RANGE,
            ).values.astype(np.float32)
            X_test = _apply_scaler_to_stat_frame(
                raw_stat.iloc[test_idx],
                fold_scaler,
                clip_range=TREE_MODEL_SCALER_CLIP_RANGE,
            ).values.astype(np.float32)
            print(
                f"      XGBoost Shapes[{fold_no}]: fit={X_fit.shape} test={X_test.shape} "
                f"| scaler_fit_rows={len(fit_idx):,}"
            )
            class_weights = _compute_binary_class_weights(y[fit_idx])
            sw = _weighted_quality_weights(fit_idx, class_weights)
            model_kwargs = {
                'n_estimators': 800,
                'max_depth': tree_depth,
                'learning_rate': 0.03,
                'subsample': 0.80,
                'colsample_bytree': 0.70,
                'reg_lambda': 3.0,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'random_state': 84 + fold_no,
                'tree_method': 'hist',
            }
            if early_stopping_rounds is not None:
                model_kwargs['early_stopping_rounds'] = early_stopping_rounds
            model = XGBClassifier(**model_kwargs)
            fit_kwargs = {
                'sample_weight': sw,
                'verbose': False,
            }
            X_val = None
            if inner_val is not None:
                X_val = _apply_scaler_to_stat_frame(
                    raw_stat.iloc[inner_val],
                    fold_scaler,
                    clip_range=TREE_MODEL_SCALER_CLIP_RANGE,
                ).values.astype(np.float32)
                fit_kwargs['eval_set'] = [(X_val, y[inner_val])]
            model.fit(X_fit, y[fit_idx], **fit_kwargs)
            present_classes = getattr(model, 'classes_', np.unique(y[fit_idx]))
            raw_preds = align_probability_columns(
                model.predict_proba(X_test),
                N_XGB_PROBS,
                classes=present_classes,
            )
            calibrator = None
            calibrator_report = {'enabled': False, 'reason': 'no_inner_validation'}
            if X_val is not None:
                val_raw = align_probability_columns(
                    model.predict_proba(X_val),
                    N_XGB_PROBS,
                    classes=present_classes,
                )
                calibrator, calibrator_report = _fit_long_isotonic_calibrator(y[inner_val], val_raw[:, 0])
            preds = _apply_long_calibrator(calibrator, raw_preds)
            test_metrics = _calibration_metrics(y[test_idx], raw_preds, preds)
            pred_labels = np.argmax(preds, axis=1)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y[test_idx],
                pred_labels,
                labels=[0, 1],
                average='macro',
                zero_division=0,
            )
            xgb_best_iteration = getattr(model, 'best_iteration', None) if early_stopping_rounds is not None else None
            return preds, {
                'directional_precision': float(precision),
                'directional_recall': float(recall),
                'directional_f1': float(f1),
                'classes': [int(cls) for cls in np.asarray(present_classes).reshape(-1).tolist()],
                'best_iteration': None if xgb_best_iteration is None else int(xgb_best_iteration),
                'early_stopping_rounds': early_stopping_rounds,
                'class_weights': class_weights,
                'fold_scaler_diagnostics': scaler_diag,
                'calibration': {
                    'inner_validation': calibrator_report,
                    'outer_test': test_metrics,
                },
            }

        xgb_probs_raw, xgb_covered, xgb_reports = run_sequential_oof(n, N_XGB_PROBS, splits, _xgb_predict)
        xgb_probs = fill_uncovered_probabilities(xgb_probs_raw, xgb_covered, priors=priors)
    else:
        print("  ℹ️ XGBoost disabled — Stage 1 meta surface will use CatBoost + regime only.")

    regime_meta_dim = len(REGIME_ONE_HOT_COLS) + len(REGIME_META_SCORE_COLS)

    def _regime_predict(train_idx, test_idx, fold_no):
        clf = RegimeClassifier(n_regimes=N_CLUSTERS, model_type=regime_model_type)
        clf.fit(df.iloc[train_idx].copy(), output_dir=regime_tmp_dir)
        meta = clf.predict_regime_meta(df.iloc[test_idx].copy())
        labels = np.argmax(meta.loc[:, list(REGIME_ONE_HOT_COLS)].values, axis=1).astype(np.int32)
        return meta.values.astype(np.float32), {
            'cluster_counts': np.bincount(labels, minlength=N_CLUSTERS).tolist(),
            'model_type': str(getattr(clf, 'model_type', regime_model_type)),
        }

    regime_raw, regime_covered, regime_reports = run_sequential_oof(n, regime_meta_dim, splits, _regime_predict)
    regime_meta = _regime_priors_from_meta(regime_raw, regime_covered)
    regime_meta[regime_covered] = regime_raw[regime_covered]
    coverage = prob_covered & xgb_covered & regime_covered

    coverage_warnings = {
        'catboost': float(prob_covered.mean()) < 0.50,
        'xgboost': bool(include_xgboost and float(xgb_covered.mean()) < 0.50),
        'regime': float(regime_covered.mean()) < 0.50,
        'combined': float(coverage.mean()) < 0.50,
    }
    if coverage_warnings['catboost'] or coverage_warnings['xgboost']:
        xgb_segment = f" | xgboost={float(xgb_covered.mean()):.1%}" if include_xgboost else ""
        print(
            "  ⚠️ Low OOF coverage detected: "
            f"catboost={float(prob_covered.mean()):.1%}{xgb_segment}"
        )

    cb_calibrator_path = os.path.join(output_dir, 'catboost_calibrator_v19.pkl')
    xgb_calibrator_path = os.path.join(output_dir, 'xgboost_calibrator_v19.pkl')
    decision_policy_path = os.path.join(output_dir, DEFAULT_DECISION_POLICY_ARTIFACT)
    stale_stage1_paths = [cb_calibrator_path, xgb_calibrator_path, decision_policy_path]
    if not include_xgboost:
        stale_stage1_paths.extend(
            [
                os.path.join(output_dir, 'xgboost_advisor_v19.json'),
                os.path.join(output_dir, 'xgboost_classes_v19.json'),
            ]
        )
    for path in stale_stage1_paths:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    final_scaler = inference_scaler_params
    X_final = _apply_scaler_to_stat_frame(
        raw_stat,
        final_scaler,
        clip_range=TREE_MODEL_SCALER_CLIP_RANGE,
    ).values.astype(np.float32)
    final_train_idx, final_holdout_idx = _chronological_holdout_indices(len(df), holdout_frac=0.10)
    final_train_sw = _weighted_quality_weights(final_train_idx, _compute_binary_class_weights(y[final_train_idx]))
    X_train_final = X_final[final_train_idx]
    y_train_final = y[final_train_idx]
    X_holdout_final = X_final[final_holdout_idx] if len(final_holdout_idx) else np.zeros((0, X_final.shape[1]), dtype=np.float32)
    y_holdout_final = y[final_holdout_idx] if len(final_holdout_idx) else np.zeros(0, dtype=np.int32)

    final_model = CatBoostClassifier(
        iterations=1000,
        depth=tree_depth,
        learning_rate=0.01,
        l2_leaf_reg=3.0,
        bootstrap_type='Bernoulli',
        subsample=0.80,
        rsm=0.70,
        loss_function='Logloss',
        eval_metric='Logloss',
        use_best_model=len(final_holdout_idx) > 0,
        early_stopping_rounds=50 if len(final_holdout_idx) > 0 else None,
        verbose=50,
        random_seed=42,
        task_type=cb_task_type,
        devices=cb_devices,
    )
    final_pool = Pool(X_train_final, y_train_final, weight=final_train_sw, feature_names=stat_features)
    final_eval_pool = Pool(X_holdout_final, y_holdout_final, feature_names=stat_features) if len(final_holdout_idx) else None
    final_model.fit(final_pool, eval_set=final_eval_pool, plot=False)
    final_model.save_model(os.path.join(output_dir, 'catboost_advisor_v19.cbm'))
    final_cb_classes = np.asarray(getattr(final_model, 'classes_', np.unique(y_train_final)), dtype=np.int32).tolist()
    with open(os.path.join(output_dir, 'catboost_classes_v19.json'), 'w') as f:
        json.dump({'classes': final_cb_classes}, f, indent=2)
    print(f"  ✅ CatBoost final classes: {final_cb_classes}")

    final_cb_calibrator = None
    final_cb_cal_report = {'enabled': False, 'reason': 'no_holdout'}
    if len(final_holdout_idx):
        holdout_raw = align_probability_columns(
            final_model.predict_proba(X_holdout_final),
            N_CB_PROBS,
            classes=final_cb_classes,
        )
        final_cb_calibrator, final_cb_cal_report = _fit_long_isotonic_calibrator(y_holdout_final, holdout_raw[:, 0])
        if final_cb_calibrator is not None:
            with open(cb_calibrator_path, 'wb') as f:
                pickle.dump(final_cb_calibrator, f)
    live_probs_raw = align_probability_columns(
        final_model.predict_proba(X_final),
        N_CB_PROBS,
        classes=final_cb_classes,
    )
    live_probs = _apply_long_calibrator(final_cb_calibrator, live_probs_raw)

    final_xgb_cal_report = {'enabled': False, 'reason': 'disabled'}
    live_xgb_probs = np.zeros((len(X_final), 0), dtype=np.float32)
    if include_xgboost:
        final_xgb = XGBClassifier(
            n_estimators=800,
            max_depth=tree_depth,
            learning_rate=0.03,
            subsample=0.80,
            colsample_bytree=0.70,
            reg_lambda=3.0,
            objective='binary:logistic',
            eval_metric='logloss',
            early_stopping_rounds=50 if len(final_holdout_idx) else None,
            random_state=84,
            tree_method='hist',
        )
        xgb_fit_kwargs = {
            'sample_weight': final_train_sw,
            'verbose': False,
        }
        if len(final_holdout_idx):
            xgb_fit_kwargs['eval_set'] = [(X_holdout_final, y_holdout_final)]
        final_xgb.fit(X_train_final, y_train_final, **xgb_fit_kwargs)
        final_xgb.save_model(os.path.join(output_dir, 'xgboost_advisor_v19.json'))
        final_xgb_classes = np.asarray(getattr(final_xgb, 'classes_', np.unique(y_train_final)), dtype=np.int32).tolist()
        with open(os.path.join(output_dir, 'xgboost_classes_v19.json'), 'w') as f:
            json.dump({'classes': final_xgb_classes}, f, indent=2)
        print(f"  ✅ XGBoost final classes: {final_xgb_classes}")

        final_xgb_calibrator = None
        final_xgb_cal_report = {'enabled': False, 'reason': 'no_holdout'}
        if len(final_holdout_idx):
            holdout_xgb_raw = align_probability_columns(
                final_xgb.predict_proba(X_holdout_final),
                N_XGB_PROBS,
                classes=final_xgb_classes,
            )
            final_xgb_calibrator, final_xgb_cal_report = _fit_long_isotonic_calibrator(y_holdout_final, holdout_xgb_raw[:, 0])
            if final_xgb_calibrator is not None:
                with open(xgb_calibrator_path, 'wb') as f:
                    pickle.dump(final_xgb_calibrator, f)
        live_xgb_raw = align_probability_columns(
            final_xgb.predict_proba(X_final),
            N_XGB_PROBS,
            classes=final_xgb_classes,
        )
        live_xgb_probs = _apply_long_calibrator(final_xgb_calibrator, live_xgb_raw)

    final_regime = RegimeClassifier(n_regimes=N_CLUSTERS, model_type=regime_model_type)
    final_regime.fit(df.copy(), output_dir=output_dir)
    live_regime_meta = final_regime.predict_regime_meta(df.copy()).values.astype(np.float32)

    with open(os.path.join(output_dir, 'meta_feature_names_v19.json'), 'w') as f:
        json.dump({'meta_features': meta_feature_names}, f, indent=2)

    live_meta = np.concatenate([live_probs, live_xgb_probs, live_regime_meta], axis=1).astype(np.float32)
    np.save(os.path.join(output_dir, 'meta_features_live_v19.npy'), live_meta)

    meta = np.concatenate([oof_probs, xgb_probs, regime_meta], axis=1).astype(np.float32)
    np.save(os.path.join(output_dir, 'meta_features_oof_v19.npy'), meta)
    np.save(os.path.join(output_dir, 'meta_coverage_v19.npy'), coverage.astype(np.uint8))

    ensemble_oof_probs = oof_probs.astype(np.float32)
    if include_xgboost:
        ensemble_oof_probs = ((oof_probs.astype(np.float32) + xgb_probs.astype(np.float32)) / 2.0).astype(np.float32)
    decision_policy = build_decision_policy(
        df,
        ensemble_oof_probs,
        regime_meta,
        coverage,
        cost_config=cost_config,
        source='stage1_oof_base_models',
    )
    with open(decision_policy_path, 'w') as f:
        json.dump(decision_policy, f, indent=2)

    fold_metrics = {
        'base_models': [str(spec.get('name', 'unknown')) for spec in meta_layout.get('base_models', [])],
        'stat_features': stat_features,
        'catboost_folds': prob_reports,
        'xgboost_folds': xgb_reports,
        'regime_folds': regime_reports,
        'catboost_coverage_ratio': float(prob_covered.mean()),
        'xgboost_enabled': bool(include_xgboost),
        'xgboost_coverage_ratio': float(xgb_covered.mean()) if include_xgboost else None,
        'regime_coverage_ratio': float(regime_covered.mean()),
        'coverage_ratio': float(coverage.mean()),
        'coverage_warning': coverage_warnings,
        'catboost_low_coverage_warning': bool(coverage_warnings['catboost']),
        'xgboost_low_coverage_warning': bool(coverage_warnings['xgboost']),
        'regime_low_coverage_warning': bool(coverage_warnings['regime']),
        'fold_scaler_diagnostics': {
            'catboost': [report.get('fold_scaler_diagnostics', {}) for report in prob_reports],
            'xgboost': [report.get('fold_scaler_diagnostics', {}) for report in xgb_reports],
        },
        'fold_zero_pct_summary': {
            'catboost': [
                float((report.get('fold_scaler_diagnostics', {}) or {}).get('zero_pct', 0.0))
                for report in prob_reports
            ],
            'xgboost': [
                float((report.get('fold_scaler_diagnostics', {}) or {}).get('zero_pct', 0.0))
                for report in xgb_reports
            ],
        },
        'regime_source': str(regime_model_type),
        'decision_policy_artifact': os.path.basename(decision_policy_path),
        'decision_policy_coverage_ratio': float(decision_policy.get('coverage_ratio', 0.0)),
        'scaler_contract': 'OOF uses fold-local scalers; live uses inference scaler',
    }
    with open(os.path.join(output_dir, 'stage1_v19_metrics.json'), 'w') as f:
        json.dump(fold_metrics, f, indent=2)

    calibration_report = {
        'stage1_catboost_isotonic': {
            'enabled': any(bool((report.get('calibration', {}).get('inner_validation', {}) or {}).get('enabled', False)) for report in prob_reports),
            'covered_rows': int(np.sum(prob_covered)),
            'total_rows': int(len(prob_covered)),
            'folds': prob_reports,
        },
        'stage1_xgboost_isotonic': {
            'enabled': bool(include_xgboost and any(bool((report.get('calibration', {}).get('inner_validation', {}) or {}).get('enabled', False)) for report in xgb_reports)),
            'disabled': bool(not include_xgboost),
            'covered_rows': int(np.sum(xgb_covered)) if include_xgboost else 0,
            'total_rows': int(len(xgb_covered)) if include_xgboost else int(len(prob_covered)),
            'folds': xgb_reports,
        },
        'final_catboost_holdout_calibration': final_cb_cal_report,
        'final_xgboost_holdout_calibration': final_xgb_cal_report,
        'catboost_calibrator_artifact': os.path.basename(cb_calibrator_path) if os.path.exists(cb_calibrator_path) else None,
        'xgboost_calibrator_artifact': os.path.basename(xgb_calibrator_path) if include_xgboost and os.path.exists(xgb_calibrator_path) else None,
        'decision_policy_artifact': os.path.basename(decision_policy_path),
        'coverage_warning': coverage_warnings,
        'catboost_low_coverage_warning': bool(coverage_warnings['catboost']),
        'xgboost_low_coverage_warning': bool(coverage_warnings['xgboost']),
        'regime_low_coverage_warning': bool(coverage_warnings['regime']),
        'fold_zero_pct_summary': fold_metrics['fold_zero_pct_summary'],
        'regime_source': str(regime_model_type),
        'scaler_contract': 'OOF uses fold-local scalers; live uses inference scaler',
    }
    with open(os.path.join(output_dir, 'calibration_report.json'), 'w') as f:
        json.dump(calibration_report, f, indent=2)

    if meta.shape[1] != len(meta_feature_names):
        raise ValueError(
            f"❌ Stage1 meta feature width mismatch: meta={meta.shape} vs names={len(meta_feature_names)}"
        )
    print(f"  ✅ OOF Meta Features: {meta.shape} | names={meta_feature_names}")
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


def _lob_event_timestamp_overlap_stats(
    df: pd.DataFrame,
    lob_timestamps: pd.Series,
    *,
    tolerance: str = '1s',
    max_tensors: int | None = None,
) -> dict:
    n_lob = len(lob_timestamps) if max_tensors is None else min(len(lob_timestamps), int(max_tensors))
    row_ts = _time_series(df, 'ts_event')
    row_df = pd.DataFrame({
        'ts_event': row_ts,
        'row_idx': np.arange(len(df), dtype=np.int32),
    }).sort_values('ts_event')
    lob_df = pd.DataFrame({
        'ts_event': pd.to_datetime(pd.Series(lob_timestamps).iloc[:n_lob], utc=True, errors='coerce').dt.tz_localize(None),
        'tensor_idx': np.arange(n_lob, dtype=np.int32),
    }).dropna(subset=['ts_event']).sort_values('ts_event')
    if len(row_df) == 0 or len(lob_df) == 0:
        return {
            'tolerance': str(tolerance),
            'rows_total': int(len(row_df)),
            'lob_tensors_total': int(len(lob_df)),
            'matched_rows': 0,
            'matched_ratio': 0.0,
        }
    merged = pd.merge_asof(
        row_df,
        lob_df,
        on='ts_event',
        direction='backward',
        tolerance=pd.Timedelta(tolerance),
    )
    matched_rows = int(merged['tensor_idx'].notna().sum())
    return {
        'tolerance': str(tolerance),
        'rows_total': int(len(row_df)),
        'lob_tensors_total': int(len(lob_df)),
        'matched_rows': matched_rows,
        'matched_ratio': float(matched_rows / max(len(row_df), 1)),
    }


def _assert_lob_event_alignment(
    df: pd.DataFrame,
    lob_timestamps: pd.Series,
    *,
    tolerance: str = '1s',
    min_overlap_ratio: float = 0.90,
    max_tensors: int | None = None,
) -> dict:
    stats = _lob_event_timestamp_overlap_stats(
        df,
        lob_timestamps,
        tolerance=tolerance,
        max_tensors=max_tensors,
    )
    if float(stats['matched_ratio']) < float(min_overlap_ratio):
        raise RuntimeError(
            '❌ LOB/event timestamp alignment too low before Stage 2. '
            f"matched_rows={stats['matched_rows']:,}/{stats['rows_total']:,} "
            f"({float(stats['matched_ratio']):.1%}) with tolerance={stats['tolerance']}. "
            'Rebuild lob_tensors.npy/lob_tensor_timestamps.npy from the same refinery run as event_df.'
        )
    return stats


def _align_lob_to_rows(
    df: pd.DataFrame,
    lob_timestamps: pd.Series,
    max_age: str = DEFAULT_LOB_MAX_AGE,
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

    bias_series = pd.to_numeric(df.get('bias_label', 2), errors='coerce').fillna(2).astype(np.int32)
    quality_series = pd.to_numeric(df.get('signal_quality', 0), errors='coerce').fillna(0).astype(np.int32)
    train_event_series = pd.to_numeric(
        df.get('train_event_flag', df.get('event_flag', 0)),
        errors='coerce',
    ).fillna(0).astype(np.int32)
    signed_bias_target = np.zeros(len(df), dtype=np.float32)
    signed_bias_target[bias_series.values == 0] = 1.0
    signed_bias_target[bias_series.values == 1] = -1.0
    quality_scale = np.where(
        quality_series.values >= 2,
        1.0,
        np.where(quality_series.values == 1, 0.6, 0.25),
    ).astype(np.float32)
    event_scale = np.where(train_event_series.values == 1, 1.0, 0.5).astype(np.float32)
    directional_target = (signed_bias_target * quality_scale * event_scale).astype(np.float32)
    obi_series = pd.to_numeric(df.get('obi', 0.0), errors='coerce').fillna(0.0).astype(np.float32)
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
        target_value = float(directional_target[row_idx])
        if abs(target_value) < 1e-6:
            target_value = float(obi_series.iloc[row_idx])
        tensor_targets[tensor_idx] = target_value
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
        raise RuntimeError(
            '❌ DeepLOB stage requested but LOB tensors/timestamps are unavailable. '
            'Re-run stage1 with valid MBP inputs and Step 3e enabled.'
        )

    DeepLOBCNN, DEEPLOB_IMPORT_OK = _load_deeplob_runtime()
    if not DEEPLOB_IMPORT_OK:
        raise RuntimeError(
            '❌ DeepLOB runtime is required for the visual stage but TensorFlow/DeepLOB is unavailable.'
        )

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

    folds_with_test_tensors = int(
        sum(1 for fold in metrics['folds'] if int(fold.get('test_tensors', 0)) > 0)
    )
    print(f"  ✅ Visual Embeddings: {row_embs.shape}")
    print(f"  ✅ Visual Coverage: {row_cov.sum():,}/{n_rows:,} ({row_cov.mean():.1%})")
    if VISUAL_COVERAGE_FAIL_FAST and (folds_with_test_tensors == 0 or not bool(row_cov.any())):
        raise RuntimeError(
            "❌ DeepLOB visual stage completed without usable coverage. "
            f"folds_with_test_tensors={folds_with_test_tensors} | "
            f"rows_with_visual={int(row_cov.sum()):,}/{n_rows:,}. "
            "Rebuild Stage1 LOB artifacts from the same run before training the MetaLearner."
        )
    return row_embs, row_cov


def build_safe_sequences(
    df: pd.DataFrame,
    X_rows: np.ndarray,
    coverage_mask: np.ndarray,
    seq_len: int = SEQ_LEN,
    train_frac: float = 0.80,
    split_time: pd.Timestamp | str | None = None,
    min_seq_coverage: float = 0.80,
    n_stat_feat: int = len(CATBOOST_ADVISOR_FEATURES),
    sequence_aux_mode: str = SEQUENCE_AUX_LAST_STEP_ONLY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    n = len(df)
    split_ctx = _sequence_split_context(
        df,
        seq_len=seq_len,
        train_frac=train_frac,
        split_time=split_time,
    )
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


def _event_gate_train_prefix(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.80,
    split_time: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    split_ctx = _sequence_split_context(
        df,
        seq_len=SEQ_LEN,
        train_frac=train_frac,
        split_time=_parse_optional_timestamp(split_time),
    )
    train_mask = np.asarray(split_ctx.get('train_row_ok', []), dtype=bool)
    if train_mask.size != len(df) or not np.any(train_mask):
        fallback_rows = int(min(max(split_ctx.get('split_idx', 1), 1), len(df)))
        if fallback_rows >= len(df) and len(df) > 1:
            fallback_rows = len(df) - 1
        train_mask = np.zeros(len(df), dtype=bool)
        train_mask[:max(fallback_rows, 1)] = True
    return df.loc[train_mask].copy().reset_index(drop=True)


def _infer_event_gate_schema(df: pd.DataFrame) -> dict:
    event_cfg = {
        'roll_window': DEFAULT_EVENT_ROLL_WINDOW,
        'vol_mult': DEFAULT_EVENT_VOL_MULT,
        'obi_thr': DEFAULT_EVENT_OBI_THR,
        'wall_str_thr': DEFAULT_EVENT_WALL_STR_THR,
        'shift_z_thr': DEFAULT_EVENT_SHIFT_Z_THR,
        'score_threshold': DEFAULT_EVENT_SCORE_THRESHOLD,
        'rows_used': int(len(df)),
        'event_col': 'train_event_flag' if 'train_event_flag' in df.columns else 'event_flag',
        'score_threshold_source': 'train_only',
    }
    if len(df) == 0 or 'event_score' not in df.columns:
        return event_cfg

    event_col = str(event_cfg['event_col'])
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


def _build_live_visual_embeddings_for_rows(
    df: pd.DataFrame,
    output_dir: str,
    lob_tensors,
    lob_timestamps: pd.Series | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    n_rows = int(len(df))
    zero_emb = np.zeros((n_rows, VISUAL_EMB_DIM), dtype=np.float32)
    zero_cov = np.zeros(n_rows, dtype=bool)
    info = {
        'enabled': False,
        'reason': 'uninitialized',
        'rows': n_rows,
        'rows_with_visual': 0,
        'coverage_ratio': 0.0,
    }
    if n_rows == 0:
        info['reason'] = 'empty_frame'
        return zero_emb, zero_cov, info
    if lob_tensors is None or lob_timestamps is None or len(lob_tensors) == 0:
        info['reason'] = 'lob_unavailable'
        return zero_emb, zero_cov, info

    deeplob_model_path = os.path.join(output_dir, 'deeplob_cnn_v19.keras')
    if not os.path.exists(deeplob_model_path):
        info['reason'] = 'deeplob_artifact_missing'
        return zero_emb, zero_cov, info

    DeepLOBCNN, ok = _load_deeplob_runtime()
    if not ok:
        info['reason'] = 'deeplob_runtime_unavailable'
        return zero_emb, zero_cov, info

    cnn = DeepLOBCNN(brain_file=deeplob_model_path)
    if getattr(cnn, 'model', None) is None or not getattr(cnn, '_fitted', False):
        info['reason'] = 'deeplob_model_not_ready'
        return zero_emb, zero_cov, info

    row_to_tensor, _, _ = _align_lob_to_rows(
        df,
        lob_timestamps,
        max_tensors=len(lob_tensors),
    )
    tensor_ids = np.unique(row_to_tensor[row_to_tensor >= 0]).astype(np.int32)
    if len(tensor_ids) == 0:
        info['reason'] = 'no_aligned_tensors'
        return zero_emb, zero_cov, info

    X_eval = np.asarray(lob_tensors[tensor_ids], dtype=np.float32)
    emb_eval = np.asarray(cnn.get_embeddings(X_eval), dtype=np.float32)
    emb_map = {int(tensor_id): emb_eval[i] for i, tensor_id in enumerate(tensor_ids)}

    row_embs = np.zeros((n_rows, VISUAL_EMB_DIM), dtype=np.float32)
    row_cov = np.zeros(n_rows, dtype=bool)
    for row_idx, tensor_idx in enumerate(row_to_tensor):
        tensor_idx = int(tensor_idx)
        if tensor_idx < 0 or tensor_idx not in emb_map:
            continue
        row_embs[row_idx] = emb_map[tensor_idx]
        row_cov[row_idx] = True

    info.update({
        'enabled': True,
        'reason': 'ok',
        'rows_with_visual': int(np.sum(row_cov)),
        'coverage_ratio': float(np.mean(row_cov)) if n_rows else 0.0,
        'tensor_rows': int(np.sum(row_to_tensor >= 0)),
        'unique_tensors': int(len(tensor_ids)),
    })
    return row_embs, row_cov, info


def _generate_end_to_end_holdout_report(
    full_df: pd.DataFrame,
    *,
    output_dir: str,
    split_time: pd.Timestamp | str | None,
    train_frac: float,
    event_gate_cfg: dict | None,
    lob_tensors=None,
    lob_timestamps: pd.Series | None = None,
) -> dict:
    split_ts = _parse_optional_timestamp(split_time)
    if len(full_df) == 0 or split_ts is None:
        return {
            'enabled': False,
            'reason': 'missing_holdout_split',
        }

    split_ctx = _sequence_split_context(
        full_df,
        seq_len=SEQ_LEN,
        train_frac=train_frac,
        split_time=split_ts,
    )
    split_idx = int(split_ctx.get('split_idx', 0))
    if split_idx >= len(full_df):
        return {
            'enabled': False,
            'reason': 'empty_holdout_rows',
            'split_idx': split_idx,
        }

    gate_roll_window = int((event_gate_cfg or {}).get('roll_window', DEFAULT_EVENT_ROLL_WINDOW))
    context_rows = max(int(SEQ_LEN - 1), int(gate_roll_window - 1))
    start_idx = max(0, split_idx - context_rows)
    report_start_idx = int(split_idx - start_idx)
    eval_df = full_df.iloc[start_idx:].copy().reset_index(drop=True)
    expected_holdout_rows = int(len(eval_df) - report_start_idx)
    if expected_holdout_rows <= 0:
        return {
            'enabled': False,
            'reason': 'empty_holdout_rows',
            'split_idx': split_idx,
        }

    visual_embeddings, visual_coverage, visual_info = _build_live_visual_embeddings_for_rows(
        eval_df,
        output_dir=output_dir,
        lob_tensors=lob_tensors,
        lob_timestamps=lob_timestamps,
    )

    try:
        from predict_v19 import V19PredictionEngine
    except Exception as exc:
        return {
            'enabled': False,
            'reason': f'prediction_engine_import_failed: {exc}',
        }

    engine = V19PredictionEngine(models_dir=output_dir, run_mode='backtest')
    results = engine.run_backtest(
        eval_df,
        already_scaled=True,
        visual_embeddings=visual_embeddings,
    )
    results_df = pd.DataFrame(results)
    if results_df.empty or 'idx' not in results_df.columns:
        return {
            'enabled': False,
            'reason': 'no_scored_rows',
            'expected_holdout_rows': expected_holdout_rows,
        }

    row_idx = pd.to_numeric(results_df.get('idx', -1), errors='coerce').fillna(-1).astype(np.int32)
    results_df = results_df.loc[row_idx >= int(report_start_idx)].copy().reset_index(drop=True)
    if results_df.empty:
        return {
            'enabled': False,
            'reason': 'holdout_rows_filtered_empty',
            'expected_holdout_rows': expected_holdout_rows,
        }

    y_true = pd.to_numeric(results_df.get('true_bias', 2), errors='coerce').fillna(2).astype(np.int32).to_numpy()
    y_pred = pd.to_numeric(results_df.get('bias_idx', 2), errors='coerce').fillna(2).astype(np.int32).to_numpy()
    gate_pass = results_df.get('event_gate_passed', pd.Series(False, index=results_df.index)).fillna(False).astype(bool).to_numpy()
    tradeable = results_df.get('tradeable', pd.Series(False, index=results_df.index)).fillna(False).astype(bool).to_numpy()

    class_report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=['LONG', 'SHORT', 'NEUTRAL'],
        zero_division=0,
    )
    class_report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=['LONG', 'SHORT', 'NEUTRAL'],
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    directional_mask = np.isin(y_true, [0, 1])
    directional_metrics = {
        'precision_macro': 0.0,
        'recall_macro': 0.0,
        'f1_macro': 0.0,
        'rows': int(np.sum(directional_mask)),
    }
    if np.any(directional_mask):
        d_precision, d_recall, d_f1, _ = precision_recall_fscore_support(
            y_true[directional_mask],
            y_pred[directional_mask],
            labels=[0, 1],
            average='macro',
            zero_division=0,
        )
        directional_metrics.update({
            'precision_macro': float(d_precision),
            'recall_macro': float(d_recall),
            'f1_macro': float(d_f1),
        })

    true_neutral_mask = y_true == 2
    pred_neutral_mask = y_pred == 2
    summary = {
        'enabled': True,
        'split_time': str(split_ts),
        'context_rows': int(report_start_idx),
        'expected_holdout_rows': int(expected_holdout_rows),
        'scored_rows': int(len(results_df)),
        'scored_row_coverage': float(len(results_df) / max(expected_holdout_rows, 1)),
        'accuracy_3class': float(np.mean(y_true == y_pred)),
        'macro_f1_3class': float(class_report_dict.get('macro avg', {}).get('f1-score', 0.0)),
        'weighted_f1_3class': float(class_report_dict.get('weighted avg', {}).get('f1-score', 0.0)),
        'directional_metrics': directional_metrics,
        'event_gate_pass_rate': float(np.mean(gate_pass)) if len(gate_pass) else 0.0,
        'event_gate_pass_rate_directional_true': float(np.mean(gate_pass[directional_mask])) if np.any(directional_mask) else 0.0,
        'event_gate_pass_rate_neutral_true': float(np.mean(gate_pass[true_neutral_mask])) if np.any(true_neutral_mask) else 0.0,
        'tradeable_rate': float(np.mean(tradeable)) if len(tradeable) else 0.0,
        'predicted_neutral_rate': float(np.mean(pred_neutral_mask)) if len(pred_neutral_mask) else 0.0,
        'predicted_neutral_rows': int(np.sum(pred_neutral_mask)),
        'correct_neutral_rows': int(np.sum(pred_neutral_mask & true_neutral_mask)),
        'missed_directional_rows_as_neutral': int(np.sum(pred_neutral_mask & directional_mask)),
        'true_label_counts': {str(int(k)): int(v) for k, v in pd.Series(y_true).value_counts().sort_index().to_dict().items()},
        'pred_label_counts': {str(int(k)): int(v) for k, v in pd.Series(y_pred).value_counts().sort_index().to_dict().items()},
        'reason_counts': {str(k): int(v) for k, v in results_df.get('reason', pd.Series(dtype='object')).fillna('').value_counts().head(15).to_dict().items()},
        'event_gate_reason_counts': {str(k): int(v) for k, v in results_df.get('event_gate_reason', pd.Series(dtype='object')).fillna('').value_counts().head(15).to_dict().items()},
        'classification_report': class_report_dict,
        'confusion_matrix_labels': ['LONG', 'SHORT', 'NEUTRAL'],
        'confusion_matrix': cm,
        'visual_runtime': visual_info,
    }

    report_payload = {
        'summary': summary,
        'classification_report_text': class_report_text,
    }
    json_path = os.path.join(output_dir, 'end_to_end_holdout_report.json')
    txt_path = os.path.join(output_dir, 'end_to_end_holdout_report.txt')
    with open(json_path, 'w') as f:
        json.dump(report_payload, f, indent=2)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("End-to-End Holdout Report\n")
        f.write("=" * 32 + "\n")
        f.write(
            f"split_time={summary['split_time']} | context_rows={summary['context_rows']} | "
            f"scored_rows={summary['scored_rows']}/{summary['expected_holdout_rows']}\n"
        )
        f.write(
            f"accuracy_3class={summary['accuracy_3class']:.4f} | "
            f"macro_f1_3class={summary['macro_f1_3class']:.4f} | "
            f"event_gate_pass_rate={summary['event_gate_pass_rate']:.4f} | "
            f"tradeable_rate={summary['tradeable_rate']:.4f}\n\n"
        )
        f.write(class_report_text)
        f.write("\nConfusion Matrix [LONG, SHORT, NEUTRAL]\n")
        for row in cm:
            f.write(" ".join(str(int(x)) for x in row) + "\n")

    print(
        "  ✅ End-to-End Holdout: "
        f"rows={summary['scored_rows']:,}/{summary['expected_holdout_rows']:,} "
        f"| acc_3c={summary['accuracy_3class']:.2%} "
        f"| macro_f1_3c={summary['macro_f1_3class']:.3f} "
        f"| gate_pass={summary['event_gate_pass_rate']:.1%} "
        f"| neutral={summary['predicted_neutral_rate']:.1%}"
    )
    return {
        **summary,
        'json_path': json_path,
        'text_path': txt_path,
    }


def _resolve_meta_learner_profile(
    train_sequences: int,
    visual_seq_coverage: float,
) -> dict:
    train_sequences = int(max(train_sequences, 0))
    visual_seq_coverage = float(np.clip(visual_seq_coverage, 0.0, 1.0))
    compact_due_to_small_data = bool(train_sequences < 2500)
    compact_due_to_visual_coverage = bool(visual_seq_coverage < 0.85)
    if train_sequences < 2500 or visual_seq_coverage < 0.85:
        reasons = []
        if compact_due_to_small_data:
            reasons.append('small_training_set')
        if compact_due_to_visual_coverage:
            reasons.append('low_visual_coverage')
        return {
            'name': 'compact',
            'lstm_units_1': 48,
            'lstm_units_2': 24,
            'visual_dropout_rate': 0.35,
            'confidence_threshold': 0.62,
            'force_rebuild': True,
            'profile_reason': '+'.join(reasons) if reasons else 'compact_default',
            'compact_due_to_small_data': compact_due_to_small_data,
            'compact_due_to_visual_coverage': compact_due_to_visual_coverage,
            'visual_seq_coverage': float(visual_seq_coverage),
            'train_sequences': int(train_sequences),
        }
    return {
        'name': 'standard',
        'lstm_units_1': 128,
        'lstm_units_2': 64,
        'visual_dropout_rate': 0.25,
        'confidence_threshold': 0.65,
        'force_rebuild': False,
        'profile_reason': 'standard_capacity',
        'compact_due_to_small_data': False,
        'compact_due_to_visual_coverage': False,
        'visual_seq_coverage': float(visual_seq_coverage),
        'train_sequences': int(train_sequences),
    }


def stage3_meta_learner_v19(
    df: pd.DataFrame,
    meta_features: np.ndarray,
    visual_embeddings: np.ndarray,
    coverage_mask: np.ndarray,
    inference_scaler_params: dict,
    output_dir: str,
    stat_features: list[str] | None = None,
    meta_feature_names: list[str] | None = None,
    event_gate_cfg: dict | None = None,
    full_df: pd.DataFrame | None = None,
    lob_tensors=None,
    lob_timestamps: pd.Series | None = None,
    epochs: int = 100,
    batch: int = 64,
    train_frac: float = 0.80,
    split_time: pd.Timestamp | str | None = None,
    min_seq_coverage: float = 0.80,
) -> dict:
    print("\n" + "═" * 65)
    print("🧠 STAGE 3 — V19 MetaLearner (Safe Sequence Split)")
    print("═" * 65)
    MetaLearnerLSTM = _load_meta_learner_class()
    stat_features = list(stat_features or CATBOOST_ADVISOR_FEATURES)
    meta_feature_names = list(meta_feature_names or resolve_meta_feature_names(meta_dim=int(meta_features.shape[1])))
    meta_layout = infer_meta_feature_layout(meta_feature_names)
    if int(meta_features.shape[1]) != len(meta_feature_names):
        expected_dim = int(len(meta_feature_names))
        actual_dim = int(meta_features.shape[1])
        expected_layout = _meta_layout_label(meta_feature_names)
        actual_names = resolve_meta_feature_names(meta_dim=actual_dim)
        actual_layout = _meta_layout_label(actual_names)
        raise ValueError(
            '❌ MetaLearner stage meta feature contract mismatch: '
            f'expected_dim={expected_dim} layout={expected_layout}, '
            f'actual_dim={actual_dim} layout={actual_layout}. '
            f'expected_features={meta_feature_names}'
        )

    sequence_aux_mode = SEQUENCE_AUX_ALL_STEPS
    X_stat = _build_scaled_stat_matrix(df, stat_features, inference_scaler_params)
    X_rows = np.concatenate([X_stat, meta_features, visual_embeddings], axis=1).astype(np.float32)
    print(
        "  Meta Surface Layout: "
        f"base_models={[m['name'] for m in meta_layout['base_models']]} "
        f"| base_prob_dim={meta_layout['base_prob_dim']} "
        f"| regime_dim={len(meta_layout['regime_meta_cols'])}"
    )
    print(
        "  Input Shapes: "
        f"X_stat={X_stat.shape} | meta={meta_features.shape} | visual={visual_embeddings.shape} | rows={X_rows.shape}"
    )

    X_tr, yb_tr, yc_tr, X_val, yb_val, yc_val, split_stats = build_safe_sequences(
        df,
        X_rows,
        coverage_mask=coverage_mask,
        seq_len=SEQ_LEN,
        train_frac=train_frac,
        split_time=split_time,
        min_seq_coverage=min_seq_coverage,
        n_stat_feat=len(stat_features),
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

    visual_seq_coverage = float(
        np.mean(np.linalg.norm(np.asarray(visual_embeddings, dtype=np.float32), axis=1) > 0)
    ) if len(visual_embeddings) else 0.0
    profile = _resolve_meta_learner_profile(
        train_sequences=len(X_tr),
        visual_seq_coverage=visual_seq_coverage,
    )
    stage3_summary = {
        'profile_name': str(profile.get('name', 'unknown')),
        'profile_reason': str(profile.get('profile_reason', 'unknown')),
        'visual_seq_coverage': float(profile.get('visual_seq_coverage', visual_seq_coverage)),
        'train_sequences': int(profile.get('train_sequences', len(X_tr))),
        'compact_due_to_visual_coverage': bool(profile.get('compact_due_to_visual_coverage', False)),
        'compact_due_to_small_data': bool(profile.get('compact_due_to_small_data', False)),
        'end_to_end_holdout': None,
    }
    if profile.get('name') == 'compact':
        print(
            "  ⚠️ MetaLearner compact profile selected: "
            f"reason={profile.get('profile_reason')} | "
            f"visual_seq_coverage={float(profile.get('visual_seq_coverage', visual_seq_coverage)):.1%} | "
            f"train_sequences={int(profile.get('train_sequences', len(X_tr))):,}"
        )
    meta_brain_path = os.path.join(output_dir, 'meta_learner_v19.keras')
    if profile.get('force_rebuild') and os.path.exists(meta_brain_path):
        try:
            os.remove(meta_brain_path)
        except OSError:
            pass

    meta = MetaLearnerLSTM(
        seq_len=SEQ_LEN,
        n_stat_feat=len(stat_features),
        n_meta_feat=int(meta_features.shape[1]),
        n_visual_emb=VISUAL_EMB_DIM,
        brain_file=meta_brain_path,
        lstm_units_1=int(profile['lstm_units_1']),
        lstm_units_2=int(profile['lstm_units_2']),
        dropout=float(profile['visual_dropout_rate']),
        confidence_threshold=float(profile['confidence_threshold']),
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
        val_pred = meta.model.predict(X_val, verbose=0) if getattr(meta, 'model', None) is not None else None
        bias_val_probs = (
            np.asarray(val_pred.get('bias_out'), dtype=np.float32)
            if isinstance(val_pred, dict) and 'bias_out' in val_pred
            else np.zeros((len(X_val), 2), dtype=np.float32)
        )
        threshold_split = dict(getattr(meta, 'bias_threshold_split', {}) or {})
        calibration_rows = int(threshold_split.get('calibration_rows', 0))
        report_rows = int(threshold_split.get('report_rows', 0))
        temp_y = yb_val
        temp_probs = bias_val_probs
        if calibration_rows > 0 and report_rows > 0:
            temp_y = yb_val[:calibration_rows]
            temp_probs = bias_val_probs[:calibration_rows]
        temperature, temperature_report = _fit_temperature_from_probs(temp_y, temp_probs)
        temperature_report.update({
            'selection_source': str(threshold_split.get('selection_source', 'full_validation_fallback')),
            'calibration_rows': int(len(temp_y)),
            'report_rows': int(report_rows),
        })
        temperature_path = os.path.join(output_dir, 'meta_temperature_v19.json')
        with open(temperature_path, 'w') as f:
            json.dump(
                {
                    **temperature_report,
                    'temperature': None if temperature is None else float(temperature),
                },
                f,
                indent=2,
            )
        hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        event_gate_cfg = event_gate_cfg or _infer_event_gate_schema(df)
        e2e_report = None
        history_path = os.path.join(output_dir, 'meta_learner_v19_history.json')
        history_payload = {
            'history': hist_dict,
            'split': split_stats,
            'bias_class_weights': {str(k): float(v) for k, v in bias_class_weights.items()},
            'event_gate': event_gate_cfg,
            'profile': profile,
            'bias_long_threshold': float(getattr(meta, 'bias_long_threshold', 0.5)),
            'threshold_metrics': getattr(meta, 'bias_threshold_metrics', {}),
            'threshold_split': getattr(meta, 'bias_threshold_split', {}),
            'confidence_head_enabled': bool(getattr(meta, 'confidence_head_enabled', True)),
            'confidence_loss_weight': float(getattr(meta, 'current_conf_loss_weight', 0.3)),
            'confidence_target_std': float(getattr(meta, 'confidence_target_std', 0.0)),
            'temperature_scaling': temperature_report,
        }
        artifacts = {
            'catboost_model': 'catboost_advisor_v19.cbm',
            'catboost_classes': 'catboost_classes_v19.json',
            'catboost_calibrator': 'catboost_calibrator_v19.pkl',
            'decision_policy': DEFAULT_DECISION_POLICY_ARTIFACT,
            'meta_model': 'meta_learner_v19.keras',
            'meta_temperature': 'meta_temperature_v19.json',
            'regime_model': 'regime_classifier.pkl',
            'scaler_params': 'scaler_params.json',
            'deeplob_model': 'deeplob_cnn_v19.keras',
            'visual_embeddings': 'visual_embeddings_live_v19.npy',
            'visual_embeddings_oof': 'visual_embeddings_v19.npy',
            'visual_embeddings_live': 'visual_embeddings_live_v19.npy',
            'meta_features_oof': 'meta_features_oof_v19.npy',
            'meta_features_live': 'meta_features_live_v19.npy',
            'meta_feature_names': 'meta_feature_names_v19.json',
        }
        if any(spec.get('name') == 'xgboost' for spec in meta_layout.get('base_models', [])):
            artifacts.update(
                {
                    'xgboost_model': 'xgboost_advisor_v19.json',
                    'xgboost_classes': 'xgboost_classes_v19.json',
                    'xgboost_calibrator': 'xgboost_calibrator_v19.pkl',
                }
            )

        schema = {
            'version': SCHEMA_VERSION,
            'seq_len': SEQ_LEN,
            'stat_features': stat_features,
            'meta_features': meta_feature_names,
            'visual_features': VISUAL_FEATURE_NAMES,
            'sequence_aux_mode': sequence_aux_mode,
            'passthrough_cols': TRAINING_PASSTHROUGH_COLS,
            'timestamp_cols': ['ts_event', 'label_end_ts'],
            'input_dim': int(X_rows.shape[1]),
            'regime_meta_features': [*REGIME_ONE_HOT_COLS, *REGIME_META_SCORE_COLS],
            'regime_meta_semantics': 'posterior_probabilities',
            'regime_source': 'hmm' if HMM_AVAILABLE else 'fallback_rules',
            'stacking_scaler_contract': 'OOF uses fold-local scalers; live uses inference scaler',
            'base_models': meta_layout['base_models'],
            'decision_policy_artifact': DEFAULT_DECISION_POLICY_ARTIFACT,
            'deeplob': {
                'enabled': bool(len(VISUAL_FEATURE_NAMES)),
                'required_runtime': bool(len(VISUAL_FEATURE_NAMES)),
                'aux_target_mode': 'directional_signed_quality_weighted',
            },
            'meta_learner': {
                'bias_long_threshold': float(getattr(meta, 'bias_long_threshold', 0.5)),
                'threshold_metrics': getattr(meta, 'bias_threshold_metrics', {}),
                'confidence_head_enabled': bool(getattr(meta, 'confidence_head_enabled', True)),
                'confidence_loss_weight': float(getattr(meta, 'current_conf_loss_weight', 0.3)),
                'confidence_target_std': float(getattr(meta, 'confidence_target_std', 0.0)),
                'temperature_scaling': temperature_report,
            },
            'artifacts': artifacts,
            'event_gate': event_gate_cfg,
        }
        with open(os.path.join(output_dir, 'feature_schema_v19.json'), 'w') as f:
            json.dump(schema, f, indent=2)
        if full_df is not None:
            e2e_report = _generate_end_to_end_holdout_report(
                full_df,
                output_dir=output_dir,
                split_time=split_time,
                train_frac=train_frac,
                event_gate_cfg=event_gate_cfg,
                lob_tensors=lob_tensors,
                lob_timestamps=lob_timestamps,
            )
            stage3_summary['end_to_end_holdout'] = e2e_report
            history_payload['end_to_end_holdout'] = e2e_report
        with open(history_path, 'w') as f:
            json.dump(history_payload, f, indent=2)
        print("  ✅ MetaLearner V19 history + schema محفوظان")
    return stage3_summary


def _load_required_stage1_artifacts(
    output_dir: str,
    n_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    meta_path = os.path.join(output_dir, 'meta_features_oof_v19.npy')
    coverage_path = os.path.join(output_dir, 'meta_coverage_v19.npy')
    meta_features = np.load(meta_path)
    coverage = np.load(coverage_path).astype(bool)
    meta_arr = np.asarray(meta_features)
    if meta_arr.ndim != 2:
        raise ValueError(f'❌ Stage1 cached meta surface must be 2D, got {meta_arr.shape}')
    meta_feature_names = _load_stage1_meta_feature_names(output_dir, meta_dim=int(meta_arr.shape[1]))
    layout = infer_meta_feature_layout(meta_feature_names)
    required_files = [
        meta_path,
        coverage_path,
        os.path.join(output_dir, 'catboost_advisor_v19.cbm'),
        os.path.join(output_dir, 'catboost_classes_v19.json'),
        os.path.join(output_dir, 'regime_classifier.pkl'),
    ]
    if any(spec.get('name') == 'xgboost' for spec in layout.get('base_models', [])):
        required_files.extend(
            [
                os.path.join(output_dir, 'xgboost_advisor_v19.json'),
                os.path.join(output_dir, 'xgboost_classes_v19.json'),
            ]
        )
    missing = [path for path in required_files if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            '❌ Stage1 base-model artifacts missing. '
            'شغّل المرحلة الثانية أولاً:\n'
            'python train_v19.py --data <stage1_artifact_dir> --output <dir> --phase catboost\n'
            f'Missing: {missing}'
        )
    expected_meta_dim = len(meta_feature_names)
    meta_arr = np.asarray(meta_features)
    if int(meta_arr.shape[1]) != expected_meta_dim:
        actual_dim = int(meta_arr.shape[1])
        expected_layout = _meta_layout_label(meta_feature_names)
        actual_names = resolve_meta_feature_names(meta_dim=actual_dim)
        actual_layout = _meta_layout_label(actual_names)
        raise ValueError(
            '❌ Stage1 cached meta surface mismatch: '
            f'expected_dim={expected_meta_dim} layout={expected_layout}, '
            f'actual_dim={actual_dim} layout={actual_layout}. '
            f'expected_features={meta_feature_names}'
        )
    if n_rows is not None and (len(meta_features) != int(n_rows) or len(coverage) != int(n_rows)):
        raise ValueError(
            f'❌ Stage1 cached artifacts shape mismatch: meta={len(meta_features)}, coverage={len(coverage)}, expected={int(n_rows)}'
        )
    print(
        "  ✅ Stage1 cache layout: "
        f"meta={meta_arr.shape} | base_models={[m['name'] for m in layout['base_models']]}"
    )
    return meta_features, coverage, meta_feature_names


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
    include_xgboost: bool | None = None,
    training_mode: str | None = None,
    quality_weight_strong: float | None = None,
    quality_weight_weak: float | None = None,
    split_time: str | None = None,
    train_days: float | None = None,
    backtest_days: float | None = None,
    window_end: str | None = None,
    stat_feature_limit: int = DEFAULT_STAT_FEATURE_LIMIT,
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
    include_xgboost = bool(
        train_cfg.get('include_xgboost', False)
        if include_xgboost is None
        else include_xgboost
    )
    training_mode = training_mode or str(train_cfg.get('mode', TRAIN_MODE_EVENT_BINARY))
    quality_weight_strong = float(
        quality_weight_strong if quality_weight_strong is not None else train_cfg.get('quality_weight_strong', 2.0)
    )
    quality_weight_weak = float(
        quality_weight_weak if quality_weight_weak is not None else train_cfg.get('quality_weight_weak', 1.0)
    )

    df_loaded = load_training_csv(csv_path)
    source_contract = _load_source_refinery_contract(csv_path)
    effective_split_time = split_time if split_time is not None else source_contract.get('split_time')
    df_full, training_window = _resolve_training_window(
        df_loaded,
        train_frac=train_frac,
        split_time=effective_split_time,
        train_days=train_days,
        backtest_days=backtest_days,
        window_end=window_end,
    )
    _assert_single_contract_df(df_full, context='resolved_training_window')
    print(
        "  🗓️ Training Window: "
        f"mode={training_window['mode']} | "
        f"rows={training_window['rows_total']:,} | "
        f"train_rows={training_window['train_rows']:,} | "
        f"holdout_rows={training_window['holdout_rows']:,} | "
        f"split={training_window['split_time']}"
    )
    event_df, event_view_info = build_event_training_view(
        df_full,
        mode=training_mode,
        quality_weight_strong=quality_weight_strong,
        quality_weight_weak=quality_weight_weak,
        train_frac=train_frac,
        split_time=training_window.get('split_time'),
    )
    with open(os.path.join(output_dir, 'event_training_view.json'), 'w') as f:
        json.dump(event_view_info, f, indent=2)
    copied_artifacts = copy_inference_artifacts(csv_path, output_dir)
    if copied_artifacts:
        print(f"  ✅ Inference artifacts copied: {list(copied_artifacts)}")
    active_stat_features, active_stat_info = _resolve_active_stat_features(
        event_df,
        artifacts_dir=output_dir,
        feature_limit=stat_feature_limit,
    )
    print(
        "  ✅ Active stat features: "
        f"{len(active_stat_features)}/{max(int(stat_feature_limit), 1)} "
        f"| selected_file_used={active_stat_info['selected_features_used']} "
        f"| features={active_stat_features}"
    )
    print(
        "  🧱 Stage1 Base Models: "
        f"{'catboost + xgboost + regime' if include_xgboost else 'catboost + regime'}"
    )

    effective_source_contract = dict(source_contract)
    effective_source_contract.update({
        'split_time': training_window.get('split_time'),
        'train_start_time': training_window.get('train_start_time'),
        'holdout_start_time': training_window.get('holdout_start_time'),
        'holdout_end_time_exclusive': training_window.get('holdout_end_time_exclusive'),
        'requested_train_days': training_window.get('requested_train_days'),
        'requested_backtest_days': training_window.get('requested_backtest_days'),
        'window_mode': training_window.get('mode'),
    })

    inference_scaler_params, scaler_info = build_inference_scaler_params(
        event_df,
        active_stat_features,
        train_frac=train_frac,
        split_time=training_window.get('split_time'),
    )
    scaler_path = _save_scaler_params(output_dir, inference_scaler_params)
    print(
        f"  ✅ Model scaler saved: {scaler_path} | "
        f"rows={scaler_info['scaler_train_rows']:,} | split={scaler_info['split_time']}"
    )
    feature_drift_report_path = _write_feature_coverage_drift_report(
        event_df,
        stat_features=active_stat_features,
        split_time=training_window.get('split_time'),
        output_dir=output_dir,
        protected_features=set(active_stat_features) & {'cvd', 'obi', 'micro_atr', 'kyle_lambda', 'hawkes_intensity', 'vwap_z_score'},
    )
    print(f"  ✅ Feature coverage/drift report: {feature_drift_report_path}")

    default_lob_path, default_lob_ts_path = _resolve_default_lob_paths(csv_path)
    if not lob_path:
        lob_path = default_lob_path
    if not lob_ts_path:
        lob_ts_path = default_lob_ts_path
    if resolved_phase in (PHASE_FULL, PHASE_VISUAL) and (not lob_path or not os.path.exists(lob_path)):
        try:
            artifact_root = resolve_artifact_root(csv_path)
        except Exception:
            artifact_root = os.path.dirname(os.path.abspath(csv_path))
        print(f"  ⚠️ LOB tensors not found under artifact root: {artifact_root}")

    splits, split_t0, split_t1, split_meta = build_time_splits(
        event_df,
        n_folds=n_folds,
        test_size=test_size,
        embargo_pct=embargo_pct,
        min_train_pct=min_train_pct,
        embargo_min_pct=float(train_cfg.get('embargo_min_pct', embargo_pct)),
        embargo_horizon_quantile=float(train_cfg.get('embargo_horizon_quantile', 0.95)),
    )
    with open(os.path.join(output_dir, 'time_split_report.json'), 'w') as f:
        json.dump(
            {
                **split_meta,
                'n_splits': int(len(splits)),
                'rows': int(len(event_df)),
                'folds': [
                    {
                        'fold': int(i + 1),
                        'train_rows': int(len(train_idx)),
                        'test_rows': int(len(test_idx)),
                        'train_start_ts': str(split_t0.iloc[train_idx[0]]) if len(train_idx) else None,
                        'train_end_ts': str(split_t0.iloc[train_idx[-1]]) if len(train_idx) else None,
                        'test_start_ts': str(split_t0.iloc[test_idx[0]]) if len(test_idx) else None,
                        'test_end_ts': str(split_t0.iloc[test_idx[-1]]) if len(test_idx) else None,
                    }
                    for i, (train_idx, test_idx) in enumerate(splits)
                ],
            },
            f,
            indent=2,
        )
    event_gate_train_df = _event_gate_train_prefix(
        event_df,
        train_frac=train_frac,
        split_time=training_window.get('split_time'),
    )
    event_gate_cfg = _infer_event_gate_schema(event_gate_train_df)

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
            stat_features=active_stat_features,
            splits=splits,
            n_folds=n_folds,
            test_size=test_size,
            embargo_pct=embargo_pct,
            min_train_pct=min_train_pct,
            t0=split_t0,
            t1=split_t1,
            inference_scaler_params=inference_scaler_params,
            catboost_device=catboost_device,
            include_xgboost=include_xgboost,
            quality_weight_strong=quality_weight_strong,
            quality_weight_weak=quality_weight_weak,
            cost_config=(config_snapshot or {}).get('backtest', {}),
        )
        meta_feature_names = resolve_meta_feature_names(meta_dim=int(meta_features.shape[1]))
    else:
        meta_features, coverage, meta_feature_names = _load_required_stage1_artifacts(output_dir, n_rows=len(event_df))
        print(f"✅ Stage1 artifacts loaded from cache: {meta_features.shape} | layout={_meta_layout_label(meta_feature_names)}")

    if resolved_phase == PHASE_CATBOOST:
        elapsed = (datetime.datetime.now() - started_at).total_seconds()
        summary = {
            'rows_full': int(len(df_full)),
            'rows_event': int(len(event_df)),
            'training_mode': training_mode,
            'active_stat_features': active_stat_features,
            'meta_shape': list(meta_features.shape),
            'meta_coverage_ratio': float(np.mean(coverage)),
            'visual_shape': None,
            'visual_coverage_ratio': None,
            'scaler_train_rows': int(scaler_info['scaler_train_rows']),
            'training_window': training_window,
            'split_meta': split_meta,
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
                'split_time': training_window.get('split_time'),
                'train_days': train_days,
                'backtest_days': backtest_days,
                'window_end': window_end,
                'stat_feature_limit': stat_feature_limit,
            },
            inputs={
                'csv': csv_path,
                'lob': lob_path,
                'lob_ts': lob_ts_path,
            },
            metrics=summary,
            extra={
                'source_contract': effective_source_contract,
                'training_window': training_window,
                'meta_feature_dim': int(meta_features.shape[1]),
                'active_stat_features': active_stat_features,
                'active_stat_info': active_stat_info,
            },
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
            'feature_coverage_drift_report': feature_drift_report_path,
            'time_split_report': os.path.join(output_dir, 'time_split_report.json'),
            'calibration_report': os.path.join(output_dir, 'calibration_report.json'),
        }

    lob_tensors, lob_timestamps = _load_lob_inputs(lob_path, lob_ts_path)
    if resolved_phase in (PHASE_FULL, PHASE_VISUAL):
        if lob_timestamps is not None:
            lob_alignment_stats = _assert_lob_event_alignment(
                event_df,
                lob_timestamps,
                tolerance='1s',
                min_overlap_ratio=0.90,
                max_tensors=None if lob_tensors is None else len(lob_tensors),
            )
            print(
                "  ✅ LOB/event timestamp overlap: "
                f"{lob_alignment_stats['matched_rows']:,}/{lob_alignment_stats['rows_total']:,} "
                f"({lob_alignment_stats['matched_ratio']:.1%}) | tolerance={lob_alignment_stats['tolerance']}"
            )
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

    stage3_summary = {
        'profile_name': None,
        'profile_reason': None,
        'visual_seq_coverage': float(np.mean(visual_coverage)) if len(visual_coverage) else 0.0,
        'train_sequences': None,
        'compact_due_to_visual_coverage': False,
        'compact_due_to_small_data': False,
    }
    if resolved_phase == PHASE_VISUAL:
        elapsed = (datetime.datetime.now() - started_at).total_seconds()
        summary = {
            'rows_full': int(len(df_full)),
            'rows_event': int(len(event_df)),
            'training_mode': training_mode,
            'active_stat_features': active_stat_features,
            'meta_shape': list(meta_features.shape),
            'meta_coverage_ratio': float(np.mean(coverage)),
            'visual_shape': list(visual_embeddings.shape),
            'visual_coverage_ratio': float(np.mean(visual_coverage)),
            'scaler_train_rows': int(scaler_info['scaler_train_rows']),
            'training_window': training_window,
            'split_meta': split_meta,
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
                'split_time': training_window.get('split_time'),
                'train_days': train_days,
                'backtest_days': backtest_days,
                'window_end': window_end,
                'stat_feature_limit': stat_feature_limit,
            },
            inputs={
                'csv': csv_path,
                'lob': lob_path,
                'lob_ts': lob_ts_path,
            },
            metrics=summary,
            extra={
                'source_contract': effective_source_contract,
                'training_window': training_window,
                'meta_feature_dim': int(meta_features.shape[1]),
                'active_stat_features': active_stat_features,
                'active_stat_info': active_stat_info,
            },
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
            'feature_coverage_drift_report': feature_drift_report_path,
            'time_split_report': os.path.join(output_dir, 'time_split_report.json'),
            'calibration_report': os.path.join(output_dir, 'calibration_report.json'),
        }

    if resolved_phase in (PHASE_FULL, PHASE_TRAIN):
        try:
            _load_meta_learner_class()
        except Exception as e:
            raise RuntimeError(
                "❌ TensorFlow/MetaLearner غير متاح. المرحلة الثالثة لا يمكن تشغيلها الآن.\n"
                "شغّل bash install_tf_gpu_cu12.sh داخل .venv، أو ثبّت TensorFlow للـ CPU فقط إذا كنت لا تحتاج DeepLOB GPU."
            ) from e
        stage3_summary = stage3_meta_learner_v19(
            event_df,
            meta_features,
            visual_embeddings,
            coverage_mask=coverage,
            inference_scaler_params=inference_scaler_params,
            output_dir=output_dir,
            stat_features=active_stat_features,
            meta_feature_names=meta_feature_names,
            event_gate_cfg=event_gate_cfg,
            full_df=df_full,
            lob_tensors=lob_tensors,
            lob_timestamps=lob_timestamps,
            epochs=epochs,
            batch=batch,
            train_frac=train_frac,
            split_time=training_window.get('split_time'),
            min_seq_coverage=min_seq_coverage,
        )

    elapsed = (datetime.datetime.now() - started_at).total_seconds()
    summary = {
        'rows_full': int(len(df_full)),
        'rows_event': int(len(event_df)),
        'training_mode': training_mode,
        'active_stat_features': active_stat_features,
        'active_stat_info': active_stat_info,
        'meta_shape': list(meta_features.shape),
        'meta_coverage_ratio': float(np.mean(coverage)),
        'visual_shape': list(visual_embeddings.shape),
        'visual_coverage_ratio': float(np.mean(visual_coverage)),
        'scaler_train_rows': int(scaler_info['scaler_train_rows']),
        'training_window': training_window,
        'split_meta': split_meta,
        'event_gate_schema': event_gate_cfg,
        'meta_learner_profile': stage3_summary,
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
            'split_time': training_window.get('split_time'),
            'train_days': train_days,
            'backtest_days': backtest_days,
            'window_end': window_end,
            'stat_feature_limit': stat_feature_limit,
        },
        inputs={
            'csv': csv_path,
            'lob': lob_path,
            'lob_ts': lob_ts_path,
        },
        metrics=summary,
        extra={
            'source_contract': effective_source_contract,
            'training_window': training_window,
            'meta_feature_dim': int(meta_features.shape[1]),
            'event_gate_schema': event_gate_cfg,
            'meta_learner_profile': stage3_summary,
            'active_stat_features': active_stat_features,
            'active_stat_info': active_stat_info,
            'stacking_scaler_contract': 'OOF uses fold-local scalers; live uses inference scaler',
        },
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
        'feature_coverage_drift_report': feature_drift_report_path,
        'time_split_report': os.path.join(output_dir, 'time_split_report.json'),
        'calibration_report': os.path.join(output_dir, 'calibration_report.json'),
    }


def main():
    defaults = load_v19_config().get('training', {})
    p = argparse.ArgumentParser(description='QuantSystem V19 leakage-safe training')
    p.add_argument('--data', '--csv', dest='data', required=True, help='stage1 artifact dir/manifest/parquet for label_mode=v19')
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
    p.set_defaults(include_xgboost=bool(defaults.get('include_xgboost', False)))
    p.add_argument('--include_xgboost', dest='include_xgboost', action='store_true', help='enable XGBoost alongside CatBoost in Stage 1')
    p.add_argument('--no_xgboost', dest='include_xgboost', action='store_false', help='disable XGBoost and use CatBoost + regime only')
    p.add_argument('--training_mode', default=str(defaults.get('mode', TRAIN_MODE_EVENT_BINARY)))
    p.add_argument('--quality_weight_strong', type=float, default=float(defaults.get('quality_weight_strong', 2.0)))
    p.add_argument('--quality_weight_weak', type=float, default=float(defaults.get('quality_weight_weak', 1.0)))
    p.add_argument('--split_time', default=None, help='explicit holdout start timestamp (UTC/parsible string)')
    p.add_argument('--train_days', type=float, default=None, help='limit training window to N days immediately before split_time')
    p.add_argument('--backtest_days', type=float, default=None, help='limit holdout/backtest window to the last N days before window_end or dataset end')
    p.add_argument('--window_end', default=None, help='exclusive end timestamp for the train/backtest window')
    p.add_argument('--stat_feature_limit', type=int, default=int(defaults.get('stat_feature_limit', DEFAULT_STAT_FEATURE_LIMIT)))
    p.add_argument('--config', default=None, help='optional config file to override defaults')
    args = p.parse_args()

    cfg = load_v19_config(args.config)
    run_training_pipeline(
        csv_path=args.data,
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
        include_xgboost=args.include_xgboost,
        training_mode=args.training_mode,
        quality_weight_strong=args.quality_weight_strong,
        quality_weight_weak=args.quality_weight_weak,
        split_time=args.split_time,
        train_days=args.train_days,
        backtest_days=args.backtest_days,
        window_end=args.window_end,
        stat_feature_limit=args.stat_feature_limit,
        config_snapshot=cfg,
    )


if __name__ == '__main__':
    main()
