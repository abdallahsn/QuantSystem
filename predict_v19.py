"""
predict_v19.py - QuantSystem V19 inference engine
=================================================
V19 inference uses the exact feature schema and scaler artifacts produced by
the training pipeline, then reconstructs the same step vector used in training:

  [25 scaled statistical features] + [base-model directional probs] +
  [4 regime one-hot + 3 regime scores] + [8 visual embeddings if available]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import deque

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _configure_stdio_utf8() -> None:
    """Avoid UnicodeEncodeError on Windows consoles when printing non-ASCII log lines."""
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        reconfigure = getattr(stream, 'reconfigure', None)
        if callable(reconfigure):
            try:
                reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass


_configure_stdio_utf8()

from modules.failsafe_v19 import decide_runtime_mode, evaluate_system_health
from modules.decision_policy_v19 import (
    DEFAULT_DECISION_POLICY_ARTIFACT,
    evaluate_decision_policy,
    structure_bucket_from_row,
)
from modules.dynamic_labels import EventGate
from modules.feature_artifact_v19 import load_feature_artifact
from modules.feature_factory_v19 import apply_scaler_params_to_frame, infer_meta_feature_layout
from modules.logging_v19 import DataQualityLogger, EventLogWriter, PredictionLogger, RiskLogger, feature_hash_from_dict
from modules.meta_learner import MetaLearnerLSTM
from modules.oof_stacking import align_probability_columns
from modules.preprocessing_v19 import V19FeaturePreprocessor
from modules.regime_classifier import (
    REGIME_META_SCORE_COLS,
    REGIME_NAMES,
    REGIME_ONE_HOT_COLS,
    RegimeClassifier,
)
from modules.slippage_model import DailyLossGuard, position_size_from_prediction
VISUAL_MODEL_DEEPLOB = 'deeplob'
VISUAL_MODEL_LOB_TRANSFORMER = 'lob_transformer'
SUPPORTED_VISUAL_MODELS = {VISUAL_MODEL_DEEPLOB, VISUAL_MODEL_LOB_TRANSFORMER}

try:
    from modules.deeplob_cnn import DeepLOBCNN
    DEEPLOB_AVAILABLE = True
except Exception as exc:
    DEEPLOB_AVAILABLE = False
    print(f"  ⚠️ DeepLOB inference غير متاح — {exc}")

try:
    from modules.lob_transformer import LOBTransformer
    LOB_TRANSFORMER_AVAILABLE = True
except Exception as exc:
    LOB_TRANSFORMER_AVAILABLE = False
    print(f"  ⚠️ LOBTransformer inference unavailable — {exc}")

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

BIAS_LABELS = {0: 'LONG', 1: 'SHORT', 2: 'NEUTRAL'}
N_CLUSTERS = 4
N_CB_PROBS = 2
N_XGB_PROBS = 2
STAGE1_TARGET_SOFT_LABEL = 'soft_label'


def _resolve_visual_model_type(raw: str | None) -> str:
    model_type = str(raw or VISUAL_MODEL_DEEPLOB).strip().lower()
    if model_type not in SUPPORTED_VISUAL_MODELS:
        model_type = VISUAL_MODEL_DEEPLOB
    return model_type


def _pseudo_prob_head(pred: np.ndarray) -> np.ndarray:
    """Scalar regression output in (0,1) -> [p, 1-p] aligned with meta stacking."""
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    p = np.clip(p, 1e-6, 1.0 - 1e-6).astype(np.float32)
    return np.column_stack([p, (1.0 - p.astype(np.float64)).astype(np.float32)])


def _directional_metrics_from_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            'directional_precision_macro': 0.0,
            'directional_recall_macro': 0.0,
            'directional_f1_macro': 0.0,
        }
    df_r = pd.DataFrame(rows)
    if 'true_bias' not in df_r.columns or 'bias_idx' not in df_r.columns:
        return {
            'directional_precision_macro': 0.0,
            'directional_recall_macro': 0.0,
            'directional_f1_macro': 0.0,
        }
    y_true = pd.to_numeric(df_r['true_bias'], errors='coerce').fillna(2).astype(int).values
    y_pred = pd.to_numeric(df_r['bias_idx'], errors='coerce').fillna(2).astype(int).values
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        average='macro',
        zero_division=0,
    )
    return {
        'directional_precision_macro': round(float(precision), 4),
        'directional_recall_macro': round(float(recall), 4),
        'directional_f1_macro': round(float(f1), 4),
    }


class V19PredictionEngine:
    def __init__(
        self,
        models_dir: str = 'outputs_v19',
        run_mode: str = 'live',
        event_writer: EventLogWriter | None = None,
        manifest_path: str | None = None,
        symbol: str = '',
        failsafe_policy: dict | None = None,
        *,
        policy_require_positive_ev: bool = True,
        policy_edge_prob_override: float | None = None,
        skip_event_gate: bool = False,
    ):
        self.models_dir = models_dir
        self.run_mode = run_mode
        self.manifest_path = manifest_path or os.path.join(models_dir, 'manifest.json')
        self.symbol = symbol
        self.failsafe_policy = failsafe_policy or {}
        print(f"\n🔧 V19 Engine — loading artifacts from: {models_dir}")

        self.pre = V19FeaturePreprocessor(models_dir)
        self.factory = self.pre.factory
        self.scaler_params = self.pre.scaler_params
        self.schema = self.factory.schema
        self.seq_len = self.factory.seq_len
        self.stat_features = self.factory.stat_features
        self.meta_features = self.factory.meta_features
        self.visual_features = self.factory.visual_features
        self.input_dim = self.factory.input_dim
        self.meta_layout = infer_meta_feature_layout(self.meta_features)
        self.base_models = list(self.meta_layout.get('base_models', []))
        self.base_prob_dim = int(self.meta_layout.get('base_prob_dim', 0))
        self.regime_meta_features = list(self.meta_layout.get('regime_meta_cols', []))
        self.sequence_aux_mode = str(self.factory.schema.get('sequence_aux_mode', 'full_window')).strip() or 'full_window'
        print(
            "  Meta feature layout: "
            f"base_models={[m['name'] for m in self.base_models]} "
            f"| base_prob_dim={self.base_prob_dim} "
            f"| regime_dim={len(self.regime_meta_features)}"
        )
        gate_cfg = self.factory.schema.get('event_gate', {}) or {}
        self.event_gate = EventGate(
            roll_window=int(gate_cfg.get('roll_window', 50)),
            vol_mult=float(gate_cfg.get('vol_mult', 1.05)),        # FIX: 1.10 → 1.05
            obi_thr=float(gate_cfg.get('obi_thr', 0.05)),          # FIX: 0.08 → 0.05
            wall_str_thr=float(gate_cfg.get('wall_str_thr', 0.60)), # FIX: 0.70 → 0.60
            shift_z_thr=float(gate_cfg.get('shift_z_thr', 0.65)),  # FIX: 0.75 → 0.65
            score_threshold=float(gate_cfg.get('score_threshold', 0.0)),
        )
        # FIX: DailyContextEngine بـ rolling ADR حقيقي بدل 80pip ثابت
        from modules.context_features import DailyContextEngine
        self._daily_ctx = DailyContextEngine(
            default_adr=float(gate_cfg.get('default_adr_pips', 80.0)),
            adr_lookback_days=int(gate_cfg.get('adr_lookback_days', 20)),
            tick_size=float(gate_cfg.get('tick_size', 0.0001)),
        )
        artifacts = self.factory.schema.get('artifacts', {})
        meta_path = os.path.join(models_dir, artifacts.get('meta_model', 'meta_learner_v19.keras'))
        if run_mode == 'backtest':
            visual_artifact = artifacts.get('visual_embeddings_oof', artifacts.get('visual_embeddings', 'visual_embeddings_v19.npy'))
        else:
            visual_artifact = artifacts.get('visual_embeddings_live', artifacts.get('visual_embeddings', 'visual_embeddings_v19.npy'))
        self.visual_emb_path = os.path.join(models_dir, visual_artifact)

        self.cb_advisor = None
        self.cb_calibrator = None
        self._cb_soft_regression = False
        cb_path = os.path.join(models_dir, artifacts.get('catboost_model', 'catboost_advisor_v19.cbm'))
        self.cb_classes = None
        cb_classes_path = os.path.join(models_dir, artifacts.get('catboost_classes', 'catboost_classes_v19.json'))
        required_base_models = {str(spec.get('name')) for spec in self.base_models}
        if os.path.exists(cb_classes_path):
            try:
                with open(cb_classes_path, 'r') as f:
                    _cbc = json.load(f)
                    self.cb_classes = _cbc.get('classes')
                    self._cb_soft_regression = str(_cbc.get('stage1_target', '')).strip().lower() == STAGE1_TARGET_SOFT_LABEL
            except Exception:
                self.cb_classes = None
                self._cb_soft_regression = False
        if CB_AVAILABLE and os.path.exists(cb_path):
            self.cb_advisor = (
                CatBoostRegressor()
                if self._cb_soft_regression
                else CatBoostClassifier()
            )
            self.cb_advisor.load_model(cb_path)
            print(f"  ✅ CatBoost V19: {cb_path} ({'regress-soft' if self._cb_soft_regression else 'classify'})")
        else:
            print(f"  ⚠️ CatBoost V19 missing: {cb_path}")
        if run_mode != 'backtest' and 'catboost' in required_base_models and self.cb_advisor is None:
            raise FileNotFoundError(
                "❌ CatBoost artifact/runtime required by the feature schema but was not found."
            )
        calibrator_path = os.path.join(models_dir, artifacts.get('catboost_calibrator', 'catboost_calibrator_v19.pkl'))
        if os.path.exists(calibrator_path):
            try:
                with open(calibrator_path, 'rb') as f:
                    self.cb_calibrator = pickle.load(f)
                print("  ✅ CatBoost calibrator loaded")
            except Exception as exc:
                self.cb_calibrator = None
                print(f"  ⚠️ CatBoost calibrator unavailable: {exc}")

        self.xgb_advisor = None
        self.xgb_calibrator = None
        self.xgb_classes = None
        self._xgb_soft_regression = False
        xgb_path = os.path.join(models_dir, artifacts.get('xgboost_model', 'xgboost_advisor_v19.json'))
        xgb_classes_path = os.path.join(models_dir, artifacts.get('xgboost_classes', 'xgboost_classes_v19.json'))
        if os.path.exists(xgb_classes_path):
            try:
                with open(xgb_classes_path, 'r') as f:
                    _xjc = json.load(f)
                    self.xgb_classes = _xjc.get('classes')
                    self._xgb_soft_regression = str(_xjc.get('stage1_target', '')).strip().lower() == STAGE1_TARGET_SOFT_LABEL
            except Exception:
                self.xgb_classes = None
                self._xgb_soft_regression = False
        if XGB_AVAILABLE and os.path.exists(xgb_path):
            self.xgb_advisor = XGBRegressor() if self._xgb_soft_regression else XGBClassifier()
            self.xgb_advisor.load_model(xgb_path)
            print(f"  ✅ XGBoost V19: {xgb_path} ({'regress-soft' if self._xgb_soft_regression else 'classify'})")
        else:
            print(f"  ⚠️ XGBoost V19 missing: {xgb_path}")
        if run_mode != 'backtest' and 'xgboost' in required_base_models and self.xgb_advisor is None:
            raise FileNotFoundError(
                "❌ XGBoost artifact/runtime required by the feature schema but was not found."
            )
        xgb_calibrator_path = os.path.join(models_dir, artifacts.get('xgboost_calibrator', 'xgboost_calibrator_v19.pkl'))
        if os.path.exists(xgb_calibrator_path):
            try:
                with open(xgb_calibrator_path, 'rb') as f:
                    self.xgb_calibrator = pickle.load(f)
                print("  ✅ XGBoost calibrator loaded")
            except Exception as exc:
                self.xgb_calibrator = None
                print(f"  ⚠️ XGBoost calibrator unavailable: {exc}")

        self.decision_policy = None
        decision_policy_name = (
            artifacts.get('decision_policy')
            or self.schema.get('decision_policy_artifact')
            or DEFAULT_DECISION_POLICY_ARTIFACT
        )
        decision_policy_path = os.path.join(models_dir, decision_policy_name)
        if os.path.exists(decision_policy_path):
            try:
                with open(decision_policy_path, 'r') as f:
                    self.decision_policy = json.load(f)
                print(f"  ✅ Decision policy loaded: {decision_policy_name}")
            except Exception as exc:
                self.decision_policy = None
                print(f"  ⚠️ Decision policy unavailable: {exc}")

        self.regime_clf = RegimeClassifier(n_regimes=N_CLUSTERS)
        regime_ok = self.regime_clf.load(models_dir)
        if regime_ok:
            print("  ✅ Regime classifier loaded")
        else:
            print("  ⚠️ Regime classifier missing")

        regime_meta_required = len(self.regime_meta_features) > 0
        if regime_meta_required and not regime_ok:
            raise FileNotFoundError(
                "❌ Regime classifier artifact is required by the feature schema but was not found."
            )

        deeplob_cfg = self.schema.get('deeplob', {}) or {}
        visual_model_type = _resolve_visual_model_type(deeplob_cfg.get('model_type'))
        visual_artifact = (
            deeplob_cfg.get('model_artifact')
            or (
                artifacts.get('lob_transformer_model')
                if visual_model_type == VISUAL_MODEL_LOB_TRANSFORMER
                else artifacts.get('deeplob_model')
            )
            or ('lob_transformer_v19.keras' if visual_model_type == VISUAL_MODEL_LOB_TRANSFORMER else 'deeplob_cnn_v19.keras')
        )
        visual_model_path = os.path.join(models_dir, visual_artifact)
        self.visual_model_type = visual_model_type
        self.visual_encoder = None
        if visual_model_type == VISUAL_MODEL_LOB_TRANSFORMER:
            if LOB_TRANSFORMER_AVAILABLE and os.path.exists(visual_model_path):
                self.visual_encoder = LOBTransformer(brain_file=visual_model_path)
        else:
            if DEEPLOB_AVAILABLE and os.path.exists(visual_model_path):
                self.visual_encoder = DeepLOBCNN(brain_file=visual_model_path)
        if self.visual_encoder is not None and not bool(getattr(self.visual_encoder, '_fitted', False)):
            self.visual_encoder = None
        if self.visual_encoder is None:
            print(f"  ⚠️ Visual encoder missing or not fitted ({visual_model_type})")
        else:
            print(f"  ✅ Visual encoder loaded ({visual_model_type})")
        if (
            run_mode == 'live'
            and bool(deeplob_cfg.get('required_runtime', False))
            and len(self.visual_features) > 0
            and self.visual_encoder is None
        ):
            raise RuntimeError(
                f"❌ Visual runtime/model ({visual_model_type}) required by schema, but unavailable."
            )

        if os.path.exists(meta_path):
            meta_cfg = self.schema.get('meta_learner', {}) or {}
            self.meta = MetaLearnerLSTM(
                seq_len=self.seq_len,
                n_stat_feat=len(self.stat_features),
                n_meta_feat=len(self.meta_features),
                n_visual_emb=len(self.visual_features),
                brain_file=meta_path,
                bias_long_threshold=float(meta_cfg.get('bias_long_threshold', 0.50)),
            )
            self.meta.wall_scale_bid = float(meta_cfg.get('wall_scale_bid', getattr(self.meta, 'wall_scale_bid', 1.0)))
            self.meta.wall_scale_ask = float(meta_cfg.get('wall_scale_ask', getattr(self.meta, 'wall_scale_ask', 1.0)))
        else:
            self.meta = None

        if self.meta is None or not self.meta._fitted:
            self.meta = None
            print("  ⚠️ MetaLearner V19 missing or not fitted")
        else:
            print("  ✅ MetaLearner V19 loaded")
        if run_mode == 'backtest' and self.meta is not None:
            self.meta.confidence_head_enabled = False
            print("  [backtest] Meta: MC confidence head bypassed for fast causal replay")
        self.meta_temperature = None
        meta_temp_path = os.path.join(models_dir, artifacts.get('meta_temperature', 'meta_temperature_v19.json'))
        if os.path.exists(meta_temp_path):
            try:
                with open(meta_temp_path) as f:
                    temp_payload = json.load(f)
                temp_value = temp_payload.get('temperature')
                if temp_payload.get('enabled', False) and temp_value is not None:
                    self.meta_temperature = float(temp_value)
                    print(f"  ✅ Meta temperature loaded: T={self.meta_temperature:.3f}")
            except Exception as exc:
                print(f"  ⚠️ Meta temperature unavailable: {exc}")

        self.loss_guard = DailyLossGuard(
            max_daily_loss_pct=0.02,
            max_daily_trades=20,
            max_drawdown_pct=0.05,
        )
        self.policy_require_positive_ev = bool(policy_require_positive_ev)
        self.policy_edge_prob_override = policy_edge_prob_override
        self.skip_event_gate = bool(skip_event_gate)
        if self.skip_event_gate and run_mode == 'backtest':
            print("  [backtest] EventGate skipped (--skip_event_gate)")
        if not self.policy_require_positive_ev and run_mode == 'backtest':
            print("  [backtest] Cost-aware policy: EV>0 not required (--relax_policy_ev)")
        if self.policy_edge_prob_override is not None and run_mode == 'backtest':
            print(f"  [backtest] Cost-aware policy: edge prob override = {self.policy_edge_prob_override}")
        self._seq_buffer = deque(maxlen=self.seq_len)
        self._regime_buffer = deque(maxlen=max(self.seq_len * 4, 128))
        self.event_writer = event_writer
        schema_version = str(self.factory.schema.get('version', 'v19'))
        self.pred_logger = PredictionLogger(event_writer, run_mode=run_mode, manifest_path=self.manifest_path, model_version='v19', schema_version=schema_version, symbol=symbol)
        self.risk_logger = RiskLogger(event_writer, run_mode=run_mode, manifest_path=self.manifest_path, model_version='v19', schema_version=schema_version, symbol=symbol)
        self.data_logger = DataQualityLogger(event_writer, run_mode=run_mode, manifest_path=self.manifest_path, model_version='v19', schema_version=schema_version, symbol=symbol)
        print(f"  ✅ V19 Engine ready | seq_len={self.seq_len} | dim={self.input_dim}\n")

    def reset_state(self):
        self._seq_buffer.clear()
        self._regime_buffer.clear()
        self.event_gate.reset()
        self.loss_guard.reset_daily()

    def get_runtime_status(self) -> dict:
        return {
            'catboost_available': self.cb_advisor is not None,
            'xgboost_available': self.xgb_advisor is not None,
            'regime_available': bool(self.regime_clf._fitted),
            'visual_available': self.visual_encoder is not None,
            'meta_available': self.meta is not None,
            'decision_policy_available': isinstance(self.decision_policy, dict),
            'event_gate_available': self.event_gate is not None,
            'manifest_exists': os.path.exists(self.manifest_path),
            'schema_exists': os.path.exists(os.path.join(self.models_dir, 'feature_schema_v19.json')),
            'run_mode': self.run_mode,
        }

    def _base_fallback_min_edge(self) -> float:
        """Meta-off: soft regression path uses a lower bar than classifier probs."""
        if getattr(self, '_cb_soft_regression', False) or getattr(self, '_xgb_soft_regression', False):
            return 0.52
        return 0.60

    def _neutral_result(
        self,
        *,
        reason: str,
        sequence_ready: bool,
        feature_hash: str,
        event_gate_passed: bool = False,
        event_gate_reason: str = 'quiet',
    ) -> dict:
        return {
            'bias': 'NEUTRAL',
            'bias_idx': 2,
            'confidence': 0.0,
            'uncertainty': 1.0,
            'tradeable': False,
            'reason': reason,
            'sequence_ready': bool(sequence_ready),
            'feature_hash': feature_hash,
            'event_gate_passed': bool(event_gate_passed),
            'event_gate_reason': event_gate_reason,
            'direction_probs': {'LONG': 0.0, 'SHORT': 0.0},
        }

    def _get_cb_probs(self, X_stat: np.ndarray) -> np.ndarray:
        n = X_stat.shape[0]
        if self.cb_advisor is None:
            return np.ones((n, N_CB_PROBS), dtype=np.float32) / N_CB_PROBS
        try:
            if getattr(self, '_cb_soft_regression', False):
                probs = _pseudo_prob_head(self.cb_advisor.predict(X_stat))
                aligned = align_probability_columns(probs, N_CB_PROBS)
            else:
                probs = self.cb_advisor.predict_proba(X_stat)
                aligned = align_probability_columns(
                    probs,
                    N_CB_PROBS,
                    classes=getattr(self.cb_advisor, 'classes_', None) or self.cb_classes,
                )
            return self._apply_long_calibrator(aligned)
        except Exception:
            return np.ones((n, N_CB_PROBS), dtype=np.float32) / N_CB_PROBS

    def _get_xgb_probs(self, X_stat: np.ndarray) -> np.ndarray:
        n = X_stat.shape[0]
        if self.xgb_advisor is None:
            return np.ones((n, N_XGB_PROBS), dtype=np.float32) / N_XGB_PROBS
        try:
            if getattr(self, '_xgb_soft_regression', False):
                probs = _pseudo_prob_head(self.xgb_advisor.predict(X_stat))
                aligned = align_probability_columns(probs, N_XGB_PROBS)
            else:
                probs = self.xgb_advisor.predict_proba(X_stat)
                aligned = align_probability_columns(
                    probs,
                    N_XGB_PROBS,
                    classes=getattr(self.xgb_advisor, 'classes_', None) or self.xgb_classes,
                )
            return self._apply_long_calibrator(aligned, calibrator=self.xgb_calibrator)
        except Exception:
            return np.ones((n, N_XGB_PROBS), dtype=np.float32) / N_XGB_PROBS

    def _get_base_model_prob_block(self, X_stat: np.ndarray) -> np.ndarray:
        n = X_stat.shape[0]
        if not self.base_models:
            return np.zeros((n, 0), dtype=np.float32)
        blocks = []
        for spec in self.base_models:
            name = str(spec.get('name', '')).strip().lower()
            if name == 'catboost':
                blocks.append(self._get_cb_probs(X_stat))
            elif name == 'xgboost':
                blocks.append(self._get_xgb_probs(X_stat))
            else:
                raise ValueError(f'Unsupported base model in schema: {name}')
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def _build_base_model_stat_matrix(self, stat_df: pd.DataFrame) -> np.ndarray:
        # Tree base models are trained on the same scaler params but without
        # clipping, so we reconstruct that view from the preserved raw columns.
        raw_data = {}
        for col in self.stat_features:
            raw_col = f"raw__{col}"
            src = raw_col if raw_col in stat_df.columns else col
            raw_data[col] = pd.to_numeric(stat_df[src], errors="coerce").fillna(0.0).astype(np.float32)
        raw_df = pd.DataFrame(raw_data, index=stat_df.index)
        scaled = apply_scaler_params_to_frame(raw_df, self.scaler_params, clip_range=None)
        return scaled[self.stat_features].values.astype(np.float32)

    def _apply_long_calibrator(
        self,
        probs: np.ndarray,
        calibrator=None,
    ) -> np.ndarray:
        arr = np.asarray(probs, dtype=np.float32)
        calibrator = self.cb_calibrator if calibrator is None else calibrator
        if calibrator is None or arr.ndim != 2 or arr.shape[1] < 2:
            return arr
        p_long = np.clip(arr[:, 0].astype(np.float64), 1e-6, 1.0 - 1e-6)
        cal_long = np.asarray(calibrator.transform(p_long), dtype=np.float64)
        cal_long = np.clip(cal_long, 1e-6, 1.0 - 1e-6)
        out = np.zeros_like(arr, dtype=np.float32)
        out[:, 0] = cal_long.astype(np.float32)
        out[:, 1] = (1.0 - cal_long).astype(np.float32)
        return out

    def _apply_temperature(self, probs: np.ndarray) -> np.ndarray:
        arr = np.asarray(probs, dtype=np.float32)
        temp = self.meta_temperature
        if temp is None or temp <= 0 or arr.ndim != 2 or arr.shape[1] < 2:
            return arr
        logits = np.log(np.clip(arr.astype(np.float64), 1e-6, 1.0 - 1e-6)) / float(temp)
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        cal = exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-9, None)
        return cal.astype(np.float32)

    def _update_regime_context(self, stat_df: pd.DataFrame) -> None:
        if stat_df is None or len(stat_df) == 0:
            return
        for _, row in stat_df.iterrows():
            self._regime_buffer.append(row.to_dict())

    def _get_regime_meta_block(self, stat_df: pd.DataFrame) -> np.ndarray:
        n = len(stat_df)
        expected_dim = len(self.regime_meta_features)
        out = np.zeros((n, expected_dim), dtype=np.float32)
        if expected_dim == 0:
            return out
        if not self.regime_clf._fitted:
            raise RuntimeError('❌ Regime meta surface required by schema but classifier is not fitted.')
        if n == 1 and len(self._regime_buffer):
            recent_df = pd.DataFrame(list(self._regime_buffer))
            meta_df = self.regime_clf.predict_regime_meta(recent_df).iloc[-1:]
        else:
            meta_df = self.regime_clf.predict_regime_meta(stat_df)
        meta_cols = list(self.regime_meta_features)
        # Backward-compatible alias normalization between schema naming and
        # regime classifier outputs.
        alias_pairs = (
            ('regime_lowliq_score', 'regime_low_liq_score'),
            ('regime_low_liq_score', 'regime_lowliq_score'),
        )
        for target_col, source_col in alias_pairs:
            if target_col in meta_cols and target_col not in meta_df.columns and source_col in meta_df.columns:
                meta_df[target_col] = pd.to_numeric(meta_df[source_col], errors='coerce').fillna(0.0)
        missing = [col for col in meta_cols if col not in meta_df.columns]
        if missing:
            raise ValueError(
                f'❌ Regime meta surface missing required columns: {missing}'
            )
        out = meta_df.reindex(columns=meta_cols, fill_value=0.0).values.astype(np.float32)
        return out

    def _get_visual_embeddings(self, n: int, lob_tensor: np.ndarray = None) -> np.ndarray:
        if len(self.visual_features) == 0:
            return np.zeros((n, 0), dtype=np.float32)

        if lob_tensor is None or self.visual_encoder is None:
            return self.factory.zero_visual_embeddings(n)

        try:
            emb = self.visual_encoder.get_embeddings(lob_tensor)
            emb = np.asarray(emb, dtype=np.float32)
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            if emb.shape[1] < len(self.visual_features):
                pad = np.zeros((emb.shape[0], len(self.visual_features) - emb.shape[1]), dtype=np.float32)
                emb = np.concatenate([emb, pad], axis=1)
            return emb[:, :len(self.visual_features)]
        except Exception:
            return self.factory.zero_visual_embeddings(n)

    def _build_step_vectors(
        self,
        stat_df: pd.DataFrame,
        visual_emb: np.ndarray | None = None,
        meta_override: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_stat = stat_df[self.stat_features].values.astype(np.float32)
        base_model_X_stat = self._build_base_model_stat_matrix(stat_df)
        if meta_override is not None:
            meta_override = np.asarray(meta_override, dtype=np.float32)
            if meta_override.ndim == 1:
                meta_override = meta_override.reshape(1, -1)
            expected_dim = len(self.meta_features)
            if meta_override.shape[1] != expected_dim:
                raise ValueError(
                    f'❌ meta override columns ({meta_override.shape[1]}) do not match schema ({expected_dim})'
                )
            cb_probs = meta_override[:, :self.base_prob_dim]
            regime_meta = meta_override[:, self.base_prob_dim:expected_dim]
            # Defensive fallback: some legacy OOF/meta artifacts may contain
            # all-zero base/regime blocks (or NaNs), which suppresses runtime
            # CatBoost/XGBoost/regime signals. In that case, recompute online.
            if cb_probs.size > 0:
                cb_row_mass = np.abs(cb_probs).sum(axis=1)
                cb_invalid = (~np.isfinite(cb_probs)).any(axis=1) | (cb_row_mass <= 1e-8)
                if bool(np.any(cb_invalid)):
                    live_cb = self._get_base_model_prob_block(base_model_X_stat)
                    cb_probs = cb_probs.copy()
                    cb_probs[cb_invalid] = live_cb[cb_invalid]
            if regime_meta.size > 0 and self.regime_clf._fitted:
                rg_row_mass = np.abs(regime_meta).sum(axis=1)
                rg_invalid = (~np.isfinite(regime_meta)).any(axis=1) | (rg_row_mass <= 1e-8)
                if bool(np.any(rg_invalid)):
                    live_regime = self._get_regime_meta_block(stat_df)
                    regime_meta = regime_meta.copy()
                    regime_meta[rg_invalid] = live_regime[rg_invalid]
        else:
            cb_probs = self._get_base_model_prob_block(base_model_X_stat)
            regime_meta = self._get_regime_meta_block(stat_df)
        if visual_emb is None:
            visual_emb = np.zeros((len(stat_df), len(self.visual_features)), dtype=np.float32)
        X_step = np.concatenate([X_stat, cb_probs, regime_meta, visual_emb], axis=1).astype(np.float32)
        return X_step, cb_probs, regime_meta

    def _meta_decision(self, seq: np.ndarray) -> dict:
        if self.meta is None:
            last_step = seq[-1]
            offset = len(self.stat_features)
            base_block = last_step[offset:offset + self.base_prob_dim]
            if base_block.size >= N_CB_PROBS:
                probs = base_block.reshape(-1, N_CB_PROBS).mean(axis=0).astype(np.float32)
            else:
                probs = np.ones(N_CB_PROBS, dtype=np.float32) / N_CB_PROBS
            bias_idx = int(np.argmax(probs))
            min_edge = self._base_fallback_min_edge()
            return {
                'bias': BIAS_LABELS[bias_idx],
                'bias_idx': bias_idx,
                'bias_probs': probs.tolist(),
                'confidence': float(probs[bias_idx]),
                'uncertainty': 0.5,
                'tradeable': float(probs[bias_idx]) >= min_edge,
                'source': 'BaseModels_Fallback_V19',
            }
        result = self.meta.predict(seq)
        result['source'] = 'MetaLearner_V19'
        return result

    def _project_sequence_for_meta(self, seq: np.ndarray) -> np.ndarray:
        arr = np.asarray(seq, dtype=np.float32)
        if self.sequence_aux_mode != 'last_step_only' or arr.ndim != 2:
            return arr
        n_stat = len(self.stat_features)
        if arr.shape[1] <= n_stat:
            return arr
        out = arr.copy()
        out[:-1, n_stat:] = 0.0
        return out

    def _runtime_penalty(self, runtime_mode: dict | None) -> float:
        runtime_mode = runtime_mode or {}
        if not runtime_mode.get('allow_shadow', True):
            return 0.0
        penalty = 1.0
        degraded_components = list(runtime_mode.get('degraded_components', []))
        if degraded_components:
            penalty *= max(0.5, 1.0 - 0.10 * len(degraded_components))
        if runtime_mode.get('blocking_issues'):
            penalty *= 0.75
        if self.run_mode == 'rollout' and not runtime_mode.get('allow_rollout', False):
            return 0.0
        if self.run_mode == 'paper' and not runtime_mode.get('allow_paper', False):
            return 0.0
        return float(np.clip(penalty, 0.0, 1.0))

    def predict_step(self,
                     stat_features: dict,
                     visual_embedding: np.ndarray | None = None,
                     meta_override: np.ndarray | None = None,
                     lob_tensor: np.ndarray = None,
                     ts=None,
                     already_scaled: bool = False) -> dict:
        t0 = time.perf_counter()
        feature_hash = feature_hash_from_dict(stat_features or {})
        can_trade, reason = self.loss_guard.can_trade(ts)
        if not can_trade:
            result = self._neutral_result(
                reason=reason,
                sequence_ready=len(self._seq_buffer) >= self.seq_len,
                feature_hash=feature_hash,
            )
            result['latency_ms'] = (time.perf_counter() - t0) * 1000.0
            self.risk_logger.log_block(reason, ts=ts, extra={'loss_guard_status': self.loss_guard.status()})
            self.pred_logger.log_prediction(result, features=stat_features or {}, ts=ts, input_source='predict_step', latency_ms=result['latency_ms'])
            return result

        stat_df = self.factory.prepare_row(
            stat_features,
            ts=ts,
            already_scaled=already_scaled,
            include_meta=True,
        )
        if meta_override is None:
            self._update_regime_context(stat_df)
        health = evaluate_system_health(
            models_dir=self.models_dir,
            engine_status=self.get_runtime_status(),
            feature_row=stat_features,
            ts=ts,
            policy=self.failsafe_policy,
            manifest_path=self.manifest_path,
            loss_guard_status=self.loss_guard.status(),
        )
        runtime_mode = decide_runtime_mode(health, self.failsafe_policy)
        if not runtime_mode.get('allow_shadow', True):
            result = self._neutral_result(
                reason=runtime_mode.get('reason', 'Failsafe blocked'),
                sequence_ready=False,
                feature_hash=feature_hash,
            )
            result['latency_ms'] = (time.perf_counter() - t0) * 1000.0
            self.risk_logger.log_block(result['reason'], ts=ts, extra=runtime_mode, event_type='rollout_guard_triggered')
            self.pred_logger.log_prediction(result, features=stat_features or {}, ts=ts, input_source='predict_step', latency_ms=result['latency_ms'])
            return result

        if 'visual_branch_unavailable' in runtime_mode.get('degraded_components', []):
            self.data_logger.log_data_issue('visual_branch_unavailable', reason='visual branch unavailable, using stat+meta only', ts=ts, extra=runtime_mode)
        if 'regime_fallback' in runtime_mode.get('degraded_components', []):
            self.data_logger.log_data_issue('regime_fallback_used', reason='regime model unavailable, defaulting to cluster_0', ts=ts, extra=runtime_mode)
        if health.get('missing_critical_features'):
            self.data_logger.log_data_issue('feature_missing', reason='critical features missing/imputed', ts=ts, extra={'missing_critical_features': health.get('missing_critical_features')})
        if 'timestamp_stale' in runtime_mode.get('blocking_issues', []):
            self.data_logger.log_data_issue('data_gap_detected', reason='timestamp stale for incoming row', ts=ts, extra=runtime_mode)

        if self.skip_event_gate:
            gate_result = {'passed': True, 'reason': 'gate_skipped'}
        else:
            gate_result = self.event_gate.evaluate(stat_features or {})
            if not gate_result.get('passed', False):
                result = self._neutral_result(
                    reason=f"Event gate blocked ({gate_result.get('reason', 'quiet')})",
                    sequence_ready=len(self._seq_buffer) >= self.seq_len,
                    feature_hash=feature_hash,
                    event_gate_passed=False,
                    event_gate_reason=gate_result.get('reason', 'quiet'),
                )
                result['latency_ms'] = (time.perf_counter() - t0) * 1000.0
                self.pred_logger.log_prediction(result, features=stat_features or {}, ts=ts, input_source='predict_step', latency_ms=result['latency_ms'])
                return result

        if visual_embedding is not None:
            visual_emb = self.factory.prepare_visual_embeddings(visual_embedding, n_rows=len(stat_df))
        else:
            visual_emb = self._get_visual_embeddings(len(stat_df), lob_tensor=lob_tensor)
        step_rows, cb_probs, regime_meta = self._build_step_vectors(
            stat_df,
            visual_emb=visual_emb,
            meta_override=meta_override,
        )
        self._seq_buffer.append(step_rows[0])

        if len(self._seq_buffer) < self.seq_len:
            result = self._neutral_result(
                reason=f'Warming up ({len(self._seq_buffer)}/{self.seq_len})',
                sequence_ready=False,
                feature_hash=feature_hash,
                event_gate_passed=True,
                event_gate_reason=gate_result.get('reason', 'event'),
            )
            result['latency_ms'] = (time.perf_counter() - t0) * 1000.0
            self.pred_logger.log_prediction(result, features=stat_features or {}, ts=ts, input_source='predict_step', latency_ms=result['latency_ms'])
            return result

        seq = np.array(list(self._seq_buffer), dtype=np.float32)
        seq = self._project_sequence_for_meta(seq)
        result = self._meta_decision(seq)
        raw_bias_probs = np.asarray(result.get('bias_probs', cb_probs[0]), dtype=np.float32).reshape(1, -1)
        calibrated_bias_probs = self._apply_temperature(raw_bias_probs)
        if calibrated_bias_probs.shape[1] >= 2:
            result['raw_bias_probs'] = raw_bias_probs[0].tolist()
            result['bias_probs'] = calibrated_bias_probs[0].tolist()
            if self.meta is not None and hasattr(self.meta, '_labels_from_long_probs'):
                bias_idx = int(
                    self.meta._labels_from_long_probs(
                        np.array([calibrated_bias_probs[0, 0]], dtype=np.float32),
                        getattr(self.meta, 'bias_long_threshold', 0.5),
                    )[0]
                )
            else:
                bias_idx = int(np.argmax(calibrated_bias_probs[0]))
            result['bias_idx'] = bias_idx
            result['bias'] = BIAS_LABELS.get(bias_idx, 'NEUTRAL')
            result['chosen_threshold'] = float(getattr(self.meta, 'bias_long_threshold', 0.5)) if self.meta is not None else 0.5
        runtime_block_reason = None
        runtime_block_event = 'risk_blocked'
        if self.run_mode == 'rollout' and not runtime_mode.get('allow_rollout', False):
            runtime_block_reason = runtime_mode.get('reason', 'Rollout blocked')
            runtime_block_event = 'rollout_guard_triggered'
        elif self.run_mode == 'paper' and not runtime_mode.get('allow_paper', False):
            runtime_block_reason = runtime_mode.get('reason', 'Paper blocked')
        cluster = int(np.argmax(regime_meta[0, :N_CLUSTERS])) if regime_meta.shape[1] >= N_CLUSTERS else 0
        bias_probs = np.asarray(result.get('bias_probs', cb_probs[0]), dtype=np.float32).reshape(-1)
        direction_probs = {
            'LONG': round(float(bias_probs[0]) if len(bias_probs) > 0 else 0.0, 4),
            'SHORT': round(float(bias_probs[1]) if len(bias_probs) > 1 else 0.0, 4),
        }
        result['direction_probs'] = direction_probs
        if 'raw_bias_probs' in result:
            raw_bias_probs_flat = np.asarray(result['raw_bias_probs'], dtype=np.float32).reshape(-1)
            result['raw_direction_probs'] = {
                'LONG': round(float(raw_bias_probs_flat[0]) if len(raw_bias_probs_flat) > 0 else 0.0, 4),
                'SHORT': round(float(raw_bias_probs_flat[1]) if len(raw_bias_probs_flat) > 1 else 0.0, 4),
            }
        result['cb_probs'] = {
            'LONG': round(float(cb_probs[0, 0]), 4),
            'SHORT': round(float(cb_probs[0, 1]), 4),
        }
        if len(self.visual_features):
            result['visual_norm'] = round(float(np.linalg.norm(visual_emb[0])), 4)
        result['cluster'] = cluster
        result['cluster_name'] = REGIME_NAMES.get(cluster, f'Cluster_{cluster}')
        score_offset = N_CLUSTERS
        result['regime_scores'] = {
            'volatile': round(float(regime_meta[0, score_offset + 0]), 4) if regime_meta.shape[1] > score_offset + 0 else 0.0,
            'trend': round(float(regime_meta[0, score_offset + 1]), 4) if regime_meta.shape[1] > score_offset + 1 else 0.0,
            'low_liq': round(float(regime_meta[0, score_offset + 2]), 4) if regime_meta.shape[1] > score_offset + 2 else 0.0,
        }
        if self.base_prob_dim >= 4 and cb_probs.shape[1] >= 4:
            result['xgb_probs'] = {
                'LONG': round(float(cb_probs[0, 2]), 4),
                'SHORT': round(float(cb_probs[0, 3]), 4),
            }
        structure_context = stat_df.iloc[-1].to_dict() if len(stat_df) else {}
        structure_context.update(stat_features or {})
        structure_bucket = structure_bucket_from_row(structure_context)
        result['structure_bucket'] = structure_bucket
        decision = evaluate_decision_policy(
            self.decision_policy,
            direction_probs=direction_probs,
            regime_probs=regime_meta[0, :N_CLUSTERS] if regime_meta.shape[1] >= N_CLUSTERS else None,
            structure_bucket=structure_bucket,
            uncertainty=float(result.get('uncertainty', 0.0) or 0.0),
            runtime_penalty=self._runtime_penalty(runtime_mode),
            require_positive_ev=self.policy_require_positive_ev,
            edge_prob_override=self.policy_edge_prob_override,
        )
        if decision is not None:
            for key, value in decision.items():
                if key in {'long_policy', 'short_policy'}:
                    continue
                result[key] = value
            result['policy_available'] = True
        else:
            result['policy_available'] = False
        if runtime_block_reason:
            result['tradeable'] = False
            result['reason'] = runtime_block_reason
            self.risk_logger.log_block(result['reason'], ts=ts, extra=runtime_mode, event_type=runtime_block_event)
        result['sequence_ready'] = True
        result['feature_hash'] = feature_hash
        result['event_gate_passed'] = True
        result['event_gate_reason'] = gate_result.get('reason', 'event')
        max_contracts = int(self.failsafe_policy.get('max_contracts', 5) or 5)
        result['position_size'] = position_size_from_prediction(
            result,
            base_size=1,
            max_size=max_contracts,
            fraction=float(self.failsafe_policy.get('fractional_kelly', 0.25) or 0.25),
        ) if result.get('tradeable', False) else 0
        result['latency_ms'] = (time.perf_counter() - t0) * 1000.0

        # FIX: تمرير remaining_fuel و adr_pips الحقيقيين من DailyContextEngine
        try:
            ts_parsed = pd.Timestamp(ts, tz='UTC') if ts is not None else pd.Timestamp.utcnow()
            price_now = float(stat_features.get('price', stat_features.get('close', 0.0)))
            if price_now > 0:
                daily_ctx_values = self._daily_ctx.update(ts_parsed, price_now)
                ib_st, rem_fuel, fuel_ex = daily_ctx_values[:3]
                adr_p = (
                    daily_ctx_values[3]
                    if len(daily_ctx_values) >= 4
                    else float(getattr(self._daily_ctx, 'adr_pips', 80.0))
                )
                result['remaining_fuel'] = round(rem_fuel, 6)
                result['adr_pips']       = round(adr_p, 1)
                result['ib_status']      = ib_st
                result['fuel_exhausted'] = fuel_ex
        except Exception:
            result['remaining_fuel'] = 0.0060
            result['adr_pips']       = 80.0
        self.pred_logger.log_prediction(
            result,
            features=stat_features or {},
            ts=ts,
            input_source='predict_step',
            latency_ms=result['latency_ms'],
        )
        return result

    def predict_live(self, stat_features: dict, lob_tensor: np.ndarray = None, ts=None, already_scaled: bool = False) -> dict:
        return self.predict_step(
            stat_features,
            visual_embedding=None,
            lob_tensor=lob_tensor,
            ts=ts,
            already_scaled=already_scaled,
        )

    def run_backtest(
        self,
        df: pd.DataFrame,
        already_scaled: bool = False,
        visual_embeddings: np.ndarray | None = None,
        meta_features: np.ndarray | None = None,
        lob_tensors: np.ndarray | None = None,
        row_to_lob_tensor: np.ndarray | None = None,
    ) -> list[dict]:
        print(f"\n📊 V19 Backtest: {len(df):,} rows | already_scaled={already_scaled}")
        self.reset_state()
        source_has_labels = 'bias_label' in df.columns
        canonical_df = self.factory.prepare_frame(df, already_scaled=already_scaled, include_meta=True)
        if visual_embeddings is None and len(self.visual_features) and os.path.exists(self.visual_emb_path):
            try:
                vis = np.load(self.visual_emb_path)
                if len(vis) == len(canonical_df):
                    visual_embeddings = np.asarray(vis[:len(canonical_df)], dtype=np.float32)
                    print(f"  Visual Embeddings loaded: {visual_embeddings.shape}")
                else:
                    raise ValueError(
                        '❌ Refusing to auto-load cached visual embeddings because their '
                        f'row count ({len(vis)}) does not match the current CSV rows '
                        f'({len(canonical_df)}). Pass --visual_npy for a row-aligned file.'
                    )
            except Exception:
                raise
        if visual_embeddings is None:
            visual_embeddings = self.factory.zero_visual_embeddings(len(canonical_df))
        if meta_features is not None:
            meta_features = np.asarray(meta_features, dtype=np.float32)
            if len(meta_features) != len(canonical_df):
                raise ValueError(
                    f'❌ meta features rows ({len(meta_features)}) do not match CSV rows ({len(canonical_df)})'
                )
            if meta_features.ndim != 2 or meta_features.shape[1] != len(self.meta_features):
                raise ValueError(
                    f'❌ meta features columns ({meta_features.shape[1] if meta_features.ndim == 2 else meta_features.shape}) '
                    f'do not match schema ({len(self.meta_features)})'
                )

        y_bias = canonical_df['bias_label'].values.astype(np.int32) if source_has_labels else None

        results = []
        for i, (_, row) in enumerate(canonical_df.iterrows()):
            lob_tensor = None
            if row_to_lob_tensor is not None and lob_tensors is not None:
                tensor_idx = int(row_to_lob_tensor[i])
                if 0 <= tensor_idx < len(lob_tensors):
                    lob_tensor = np.asarray(lob_tensors[tensor_idx], dtype=np.float32)
                    row['lob_tensor_id'] = tensor_idx
            pred = self.predict_step(
                row.to_dict(),
                visual_embedding=None if lob_tensor is not None else (visual_embeddings[i] if len(self.visual_features) else None),
                meta_override=meta_features[i] if meta_features is not None else None,
                lob_tensor=lob_tensor,
                ts=canonical_df['ts_event'].iloc[i] if 'ts_event' in canonical_df.columns else None,
                already_scaled=True,
            )
            if pred.get('reason', '').startswith('Warming up'):
                continue

            pred['idx'] = i
            if lob_tensor is not None:
                pred['lob_tensor_id'] = int(row.get('lob_tensor_id', -1))
                pred['entrydata_source'] = 'raw_lob_tensor'
            if len(self.visual_features) and 'visual_norm' not in pred:
                pred['visual_norm'] = round(float(np.linalg.norm(visual_embeddings[i])), 4)

            if y_bias is not None:
                pred['true_bias'] = int(y_bias[i])
                pred['true_label'] = BIAS_LABELS.get(int(y_bias[i]), '?')
                pred['correct'] = pred.get('bias_idx') == int(y_bias[i])

            results.append(pred)

        if results and y_bias is not None:
            metrics = _directional_metrics_from_rows(results)
            tradeable = int(sum(1 for r in results if r.get('tradeable', False)))
            print(
                "  Directional Metrics: "
                f"P={metrics['directional_precision_macro']:.2%} "
                f"R={metrics['directional_recall_macro']:.2%} "
                f"F1={metrics['directional_f1_macro']:.2%}"
            )
            print(f"  Tradeable:     {tradeable:,}/{len(results):,} ({tradeable/max(len(results),1):.1%})")
        return results


def main():
    p = argparse.ArgumentParser(description='QuantSystem V19 prediction engine')
    p.add_argument('--models', default='outputs_v19')
    p.add_argument('--data', '--csv', dest='data', default=None, help='optional stage1 artifact dir/manifest/parquet for backtest mode')
    p.add_argument('--mode', choices=['backtest', 'live'], default='backtest')
    p.add_argument('--output', default='outputs_v19')
    p.add_argument('--visual_npy', default=None, help='optional precomputed visual embeddings for the same rows')
    p.add_argument('--lob', default=None, help='optional raw lob_tensors.npy for EntryData replay')
    p.add_argument('--lob_map_npy', default=None, help='optional row-aligned row_to_lob_tensor_id_v19.npy')
    p.add_argument('--input_scaled', action='store_true',
                   help='set this when using final stage1 artifact (already scaled)')
    args = p.parse_args()

    engine = V19PredictionEngine(args.models)

    if args.mode == 'backtest':
        if not args.data:
            raise ValueError('❌ --data مطلوب في backtest mode')
        df = load_feature_artifact(args.data)
        visual_embeddings = None
        if args.visual_npy and os.path.exists(args.visual_npy):
            visual_embeddings = np.load(args.visual_npy)
        lob_tensors = np.load(args.lob, mmap_mode='r') if args.lob and os.path.exists(args.lob) else None
        row_to_lob = (
            np.asarray(np.load(args.lob_map_npy), dtype=np.int32).reshape(-1)
            if args.lob_map_npy and os.path.exists(args.lob_map_npy)
            else (
                pd.to_numeric(df['lob_tensor_id'], errors='coerce').fillna(-1).astype(np.int32).values
                if 'lob_tensor_id' in df.columns
                else None
            )
        )
        results = engine.run_backtest(
            df,
            already_scaled=args.input_scaled,
            visual_embeddings=visual_embeddings,
            lob_tensors=lob_tensors,
            row_to_lob_tensor=row_to_lob,
        )

        os.makedirs(args.output, exist_ok=True)
        out_csv = os.path.join(args.output, 'v19_backtest_results.csv')
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"\n✅ Backtest results: {out_csv}")

        if results and 'correct' in results[0]:
            df_r = pd.DataFrame(results)
            metrics = _directional_metrics_from_rows(results)
            summary = {
                'total': int(len(df_r)),
                'tradeable': int(df_r['tradeable'].sum()),
                **metrics,
                'event_gate_rate': round(float(df_r['event_gate_passed'].mean()), 4) if 'event_gate_passed' in df_r.columns else 0.0,
                'long': int((df_r['bias'] == 'LONG').sum()),
                'short': int((df_r['bias'] == 'SHORT').sum()),
                'neutral': int((df_r['bias'] == 'NEUTRAL').sum()),
            }
            with open(os.path.join(args.output, 'v19_summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"  Summary: {summary}")


if __name__ == '__main__':
    main()
