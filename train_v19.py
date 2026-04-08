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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prepare_training_data import (
    BINARY_FEATURES,
    CATBOOST_ADVISOR_FEATURES,
    RAW_STAT_PREFIX,
    TEMPORAL_DROP_COLS,
)
try:
    from modules.deeplob_cnn import DeepLOBCNN, VISUAL_EMB_DIM
    DEEPLOB_IMPORT_OK = True
except ImportError:
    DEEPLOB_IMPORT_OK = False
    VISUAL_EMB_DIM = 8
from modules.config_v19 import load_v19_config
from modules.feature_factory_v19 import (
    DEFAULT_PASSTHROUGH_COLS,
    apply_scaler_params_to_frame,
    prepare_feature_frame,
)
from modules.manifest_v19 import write_manifest
from modules.meta_learner import MetaLearnerLSTM
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
N_CB_PROBS = 3
SEQ_LEN = 50
META_FEATURE_NAMES = [
    'cb_prob_long', 'cb_prob_short', 'cb_prob_neutral',
    'cluster_0', 'cluster_1', 'cluster_2', 'cluster_3',
]
VISUAL_FEATURE_NAMES = [f'vis_emb_{i}' for i in range(VISUAL_EMB_DIM)]
TRAINING_PASSTHROUGH_COLS = [
    col for col in DEFAULT_PASSTHROUGH_COLS
    if col in {
        'ts_event',
        'label_end_ts',
        'price',
        'bias_label',
        'conf_label',
        'regime_label',
        'is_expansion',
        'liq_score',
        'forward_return',
        'label_horizon_steps',
    }
]


def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    protected = {
        'bias_label', 'setup_label', 'conf_label', 'regime_label', 'is_expansion',
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


def _raw_feature_name(col: str) -> str:
    return f'{RAW_STAT_PREFIX}{col}'


def _raw_stat_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
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


def _class_weights(y: np.ndarray) -> dict[int, float]:
    counts = np.bincount(y.astype(np.int32), minlength=3)
    total = int(len(y))
    return {k: total / (3 * max(c, 1)) for k, c in enumerate(counts)}


def _time_series(df: pd.DataFrame, col: str, fallback: str | None = None) -> pd.Series:
    if col in df.columns:
        s = pd.to_datetime(df[col], utc=True, errors='coerce').dt.tz_localize(None)
    elif fallback and fallback in df.columns:
        s = pd.to_datetime(df[fallback], utc=True, errors='coerce').dt.tz_localize(None)
    else:
        s = pd.Series(pd.date_range('2026-01-01', periods=len(df), freq='s'))

    s = s.ffill().bfill()
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
            shutil.copy2(src, dst)
            copied[name] = dst
    return copied


def build_time_splits(
    df: pd.DataFrame,
    n_folds: int = 6,
    test_size: float = 0.10,
    embargo_pct: float = 0.02,
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
    t0: pd.Series | None = None,
    t1: pd.Series | None = None,
    inference_scaler_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    print("\n" + "═" * 65)
    print("🐱 STAGE 1 — V19 OOF CatBoost + Regime Meta-Features")
    print("═" * 65)

    n = len(df)
    raw_stat = _raw_stat_frame(df, CATBOOST_ADVISOR_FEATURES)
    y = df['bias_label'].fillna(2).astype(np.int32).values
    if splits is None:
        splits, t0, t1 = build_time_splits(
            df,
            n_folds=n_folds,
            test_size=test_size,
            embargo_pct=embargo_pct,
        )
    elif t0 is None or t1 is None:
        _, t0, t1 = build_time_splits(
            df,
            n_folds=n_folds,
            test_size=test_size,
            embargo_pct=embargo_pct,
        )

    print(f"  Splits: {len(splits)} | Rows: {n:,}")

    priors = np.bincount(y, minlength=N_CB_PROBS).astype(np.float32)
    priors = priors / max(priors.sum(), 1.0)

    if not CB_AVAILABLE:
        print("  ⚠️ CatBoost غير متاح — سيتم استخدام priors فقط")
        oof_probs = np.repeat(priors.reshape(1, -1), n, axis=0)
        regime_oh = np.zeros((n, N_CLUSTERS), dtype=np.float32)
        regime_oh[:, 0] = 1.0
        coverage = np.zeros(n, dtype=bool)
        np.save(
            os.path.join(output_dir, 'meta_features_live_v19.npy'),
            np.concatenate([oof_probs, regime_oh], axis=1).astype(np.float32),
        )
    else:
        regime_tmp_dir = tempfile.mkdtemp(prefix='_oof_regime_tmp_', dir=output_dir)
        os.makedirs(regime_tmp_dir, exist_ok=True)

        def _cb_predict(train_idx, test_idx, fold_no):
            inner_train, inner_val = _build_inner_time_split(train_idx, t0, t1, embargo_pct)
            fit_idx = inner_train if inner_train is not None else train_idx
            fold_scaler = _fit_scaler_params_from_frame(raw_stat.iloc[fit_idx])
            X_fit = _apply_scaler_to_stat_frame(raw_stat.iloc[fit_idx], fold_scaler).values.astype(np.float32)
            X_test = _apply_scaler_to_stat_frame(raw_stat.iloc[test_idx], fold_scaler).values.astype(np.float32)

            cw = _class_weights(y[fit_idx])
            sw = np.array([cw[label] for label in y[fit_idx]], dtype=np.float32)
            model = CatBoostClassifier(
                iterations=400,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=3.0,
                loss_function='MultiClass',
                eval_metric='Accuracy',
                early_stopping_rounds=50 if inner_val is not None else None,
                use_best_model=inner_val is not None,
                verbose=0,
                random_seed=42 + fold_no,
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
            acc = float(np.mean(np.argmax(preds, axis=1) == y[test_idx]))
            return preds, {'accuracy': acc}

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

        final_scaler = inference_scaler_params or _fit_scaler_params_from_frame(raw_stat)
        X_final = _apply_scaler_to_stat_frame(raw_stat, final_scaler).values.astype(np.float32)
        final_cw = _class_weights(y)
        final_sw = np.array([final_cw[label] for label in y], dtype=np.float32)
        final_model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            loss_function='MultiClass',
            eval_metric='Accuracy',
            early_stopping_rounds=50,
            use_best_model=False,
            verbose=50,
            random_seed=42,
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
    return tensors, pd.Series(ts)


def _align_lob_to_rows(
    df: pd.DataFrame,
    lob_timestamps: pd.Series,
    max_age: str = '5s',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_ts = _time_series(df, 'ts_event')
    row_df = pd.DataFrame({
        'ts_event': row_ts,
        'row_idx': np.arange(len(df), dtype=np.int32),
    })
    lob_df = pd.DataFrame({
        'ts_event': pd.to_datetime(lob_timestamps, utc=True, errors='coerce').dt.tz_localize(None),
        'tensor_idx': np.arange(len(lob_timestamps), dtype=np.int32),
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
    tensor_targets = np.zeros(len(lob_timestamps), dtype=np.float32)
    tensor_target_seen = np.zeros(len(lob_timestamps), dtype=bool)
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

    if not DEEPLOB_IMPORT_OK:
        print("  ⚠️ DeepLOB import غير متاح — visual embeddings = 0")
        np.save(os.path.join(output_dir, 'visual_embeddings_v19.npy'), zero_emb)
        np.save(os.path.join(output_dir, 'visual_embeddings_live_v19.npy'), zero_emb)
        np.save(os.path.join(output_dir, 'visual_coverage_v19.npy'), zero_cov.astype(np.uint8))
        return zero_emb, zero_cov

    row_to_tensor, tensor_targets, tensor_target_seen = _align_lob_to_rows(df, lob_timestamps)
    row_embs = np.zeros((n_rows, VISUAL_EMB_DIM), dtype=np.float32)
    row_cov = np.zeros(n_rows, dtype=bool)

    metrics = {
        'n_rows': int(n_rows),
        'n_tensors': int(len(lob_tensors)),
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    n = len(df)
    split_ctx = _sequence_split_context(df, seq_len=seq_len, train_frac=train_frac)
    split_idx = split_ctx['split_idx']
    split_time = split_ctx['split_time']

    y_bias = df['bias_label'].fillna(2).astype(np.int32).values
    y_conf = df.get('conf_label', pd.Series(np.zeros(n))).fillna(0).astype(np.float32).values

    train_row_ok = split_ctx['train_row_ok']
    val_row_ok = split_ctx['val_row_ok']

    X_tr, yb_tr, yc_tr = [], [], []
    X_val, yb_val, yc_val = [], [], []

    for end_idx in range(seq_len - 1, split_idx):
        start_idx = end_idx - seq_len + 1
        if not train_row_ok[end_idx]:
            continue
        if not coverage_mask[start_idx:end_idx + 1].all():
            continue
        X_tr.append(X_rows[start_idx:end_idx + 1])
        yb_tr.append(y_bias[end_idx])
        yc_tr.append(y_conf[end_idx])

    for end_idx in range(split_idx + seq_len - 1, n):
        start_idx = end_idx - seq_len + 1
        if start_idx < split_idx:
            continue
        if not val_row_ok[end_idx]:
            continue
        if not coverage_mask[start_idx:end_idx + 1].all():
            continue
        X_val.append(X_rows[start_idx:end_idx + 1])
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
    }
    return X_tr, yb_tr, yc_tr, X_val, yb_val, yc_val, stats


def stage3_meta_learner_v19(
    df: pd.DataFrame,
    meta_features: np.ndarray,
    visual_embeddings: np.ndarray,
    coverage_mask: np.ndarray,
    inference_scaler_params: dict,
    output_dir: str,
    epochs: int = 100,
    batch: int = 64,
    train_frac: float = 0.80,
) -> None:
    print("\n" + "═" * 65)
    print("🧠 STAGE 3 — V19 MetaLearner (Safe Sequence Split)")
    print("═" * 65)

    X_stat = _build_scaled_stat_matrix(df, CATBOOST_ADVISOR_FEATURES, inference_scaler_params)
    X_rows = np.concatenate([X_stat, meta_features, visual_embeddings], axis=1).astype(np.float32)

    X_tr, yb_tr, yc_tr, X_val, yb_val, yc_val, split_stats = build_safe_sequences(
        df, X_rows, coverage_mask=coverage_mask, seq_len=SEQ_LEN, train_frac=train_frac
    )
    if len(X_tr) == 0 or len(X_val) == 0:
        raise RuntimeError(
            '❌ لا توجد sequences كافية بعد تطبيق coverage + safe split. '
            'جرّب زيادة الداتا أو تقليل test_size/embargo.'
        )

    print(f"  Train Sequences: {len(X_tr):,}")
    print(f"  Val Sequences:   {len(X_val):,}")

    meta = MetaLearnerLSTM(
        seq_len=SEQ_LEN,
        n_stat_feat=len(CATBOOST_ADVISOR_FEATURES),
        n_visual_emb=VISUAL_EMB_DIM,
        brain_file=os.path.join(output_dir, 'meta_learner_v19.keras'),
        lstm_units_1=128,
        lstm_units_2=64,
        dropout=0.25,
        confidence_threshold=0.65,
    )

    class_weights = _class_weights(yb_tr)
    history = meta.fit_train_val(
        X_tr, yb_tr, yc_tr,
        X_val, yb_val, yc_val,
        epochs=epochs,
        batch=batch,
        output_dir=output_dir,
        class_weights=class_weights,
    )

    if history is not None:
        hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(os.path.join(output_dir, 'meta_learner_v19_history.json'), 'w') as f:
            json.dump({'history': hist_dict, 'split': split_stats}, f, indent=2)
        schema = {
            'version': 'v19-alpha',
            'seq_len': SEQ_LEN,
            'stat_features': CATBOOST_ADVISOR_FEATURES,
            'meta_features': META_FEATURE_NAMES,
            'visual_features': VISUAL_FEATURE_NAMES,
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
        }
        with open(os.path.join(output_dir, 'feature_schema_v19.json'), 'w') as f:
            json.dump(schema, f, indent=2)
        print("  ✅ MetaLearner V19 history + schema محفوظان")


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
    train_frac: float = 0.80,
    stage: int = 0,
    config_snapshot: dict | None = None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    t0 = datetime.datetime.now()

    print('=' * 65)
    print('🚀 QuantSystem V19 — Leakage-Safe Training Foundation')
    print(f'   Output: {output_dir}')
    print('=' * 65)

    df = load_training_csv(csv_path)
    copied_artifacts = copy_inference_artifacts(csv_path, output_dir)
    if copied_artifacts:
        print(f"  ✅ Inference artifacts copied: {list(copied_artifacts)}")

    inference_scaler_params, scaler_info = build_inference_scaler_params(
        df,
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

    splits, t0, t1 = build_time_splits(
        df,
        n_folds=n_folds,
        test_size=test_size,
        embargo_pct=embargo_pct,
    )

    meta_path = os.path.join(output_dir, 'meta_features_oof_v19.npy')
    coverage_path = os.path.join(output_dir, 'meta_coverage_v19.npy')
    visual_path = os.path.join(output_dir, 'visual_embeddings_v19.npy')
    visual_cov_path = os.path.join(output_dir, 'visual_coverage_v19.npy')

    if stage in (0, 1) or not (os.path.exists(meta_path) and os.path.exists(coverage_path)):
        meta_features, coverage = stage1_oof_meta(
            df,
            output_dir,
            splits=splits,
            n_folds=n_folds,
            test_size=test_size,
            embargo_pct=embargo_pct,
            t0=t0,
            t1=t1,
            inference_scaler_params=inference_scaler_params,
        )
    else:
        meta_features = np.load(meta_path)
        coverage = np.load(coverage_path).astype(bool)
        print(f'✅ Stage 1 محمّل من cache: {meta_features.shape}')

    lob_tensors, lob_timestamps = _load_lob_inputs(lob_path, lob_ts_path)
    if stage in (0, 2) or not (os.path.exists(visual_path) and os.path.exists(visual_cov_path)):
        visual_embeddings, visual_coverage = stage2_oof_visual_embeddings(
            df,
            output_dir,
            splits=splits,
            lob_tensors=lob_tensors,
            lob_timestamps=lob_timestamps,
        )
    else:
        visual_embeddings = np.load(visual_path)
        visual_coverage = np.load(visual_cov_path).astype(bool)
        print(f'✅ Stage 2 محمّل من cache: {visual_embeddings.shape}')

    if stage in (0, 3):
        stage3_meta_learner_v19(
            df,
            meta_features,
            visual_embeddings,
            coverage_mask=coverage,
            inference_scaler_params=inference_scaler_params,
            output_dir=output_dir,
            epochs=epochs,
            batch=batch,
            train_frac=train_frac,
        )

    elapsed = (datetime.datetime.now() - t0).total_seconds()
    summary = {
        'rows': int(len(df)),
        'meta_shape': list(meta_features.shape),
        'meta_coverage_ratio': float(np.mean(coverage)),
        'visual_shape': list(visual_embeddings.shape),
        'visual_coverage_ratio': float(np.mean(visual_coverage)),
        'scaler_train_rows': int(scaler_info['scaler_train_rows']),
        'elapsed_seconds': float(elapsed),
        'stage': int(stage),
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
            'train_frac': train_frac,
            'stage': stage,
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
    p.add_argument('--train_frac', type=float, default=float(defaults.get('train_frac', 0.80)))
    p.add_argument('--stage', type=int, default=int(defaults.get('stage', 0)), help='0=all, 1=stage1 only, 2=stage2 only, 3=stage3 only')
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
        train_frac=args.train_frac,
        stage=args.stage,
        config_snapshot=cfg,
    )


if __name__ == '__main__':
    main()
