"""
build_lean_model.py
===================
Lean CatBoost trainer for directional V19 rows using a simple chronological
80/20 split and a compact 13-feature surface.

Why this script exists:
  - avoid the instability introduced by many tiny temporal folds
  - keep training focused on the directional event subset only
  - let the model see the full intraday range inside one chronological train slice
  - add lightweight adaptive context via cyclical time features + relative volatility
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from modules.feature_artifact_v19 import load_feature_artifact

try:
    from catboost import CatBoostClassifier, Pool
except ImportError as e:
    raise SystemExit(
        "❌ CatBoost غير مثبّت في هذه البيئة.\n"
        "نفّذ أولًا:\n"
        "pip install -r requirements.txt"
    ) from e


DEFAULT_LEAN_FEATURES = [
    # 10 compact structural features
    "cvd",
    "obi",
    "absorption_intensity",
    "cancel_ratio",
    "micro_atr",
    "volume_burst",
    "inter_event_time",
    "kyle_lambda",
    "hawkes_intensity",
    "vwap_z_score",
    # 3 adaptive/context features
    "hour_sin",
    "hour_cos",
    "rel_vol",
]

LABEL_MAP = {0: "LONG", 1: "SHORT"}


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _load_feature_list(args: argparse.Namespace) -> list[str]:
    if args.features:
        return [str(col).strip() for col in args.features if str(col).strip()]
    if args.features_file:
        with open(args.features_file, "r") as f:
            cols = [line.strip() for line in f if line.strip()]
        if cols:
            return cols
    return list(DEFAULT_LEAN_FEATURES)


def _build_requested_columns(feature_names: list[str], label_col: str, event_col: str) -> list[str]:
    cols = ["ts_event", label_col, event_col, "signal_quality"]
    for feature in feature_names:
        cols.append(feature)
        cols.append(f"raw__{feature}")
    return list(dict.fromkeys(cols))


def _ensure_timestamp_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if "ts_event" not in df.columns:
        raise ValueError("Lean model requires 'ts_event' in the artifact")
    out = df.copy()
    out["ts_event"] = pd.to_datetime(out["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
    out = out.dropna(subset=["ts_event"]).sort_values("ts_event").reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid timestamped rows found after ts_event normalization")
    return out


def _ensure_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
    minute_of_day = (ts.dt.hour.fillna(0).astype(float) * 60.0) + ts.dt.minute.fillna(0).astype(float)
    radians = 2.0 * np.pi * minute_of_day / (24.0 * 60.0)
    out["hour_sin"] = np.sin(radians).astype(np.float32)
    out["hour_cos"] = np.cos(radians).astype(np.float32)
    return out


def _ensure_relative_volatility(
    df: pd.DataFrame,
    *,
    output_col: str = "rel_vol",
    window: int = 200,
) -> pd.DataFrame:
    out = df.copy()

    atr_source = None
    for candidate in ("raw__micro_atr", "micro_atr"):
        if candidate in out.columns:
            atr_source = candidate
            break

    if atr_source is None and output_col in out.columns and pd.to_numeric(out[output_col], errors="coerce").notna().any():
        return out
    if atr_source is None:
        out[output_col] = np.float32(1.0)
        return out

    atr = pd.to_numeric(out[atr_source], errors="coerce").abs().fillna(0.0)
    baseline = atr.rolling(window=max(int(window), 1), min_periods=1).mean().replace(0.0, np.nan)
    rel_vol = (atr / baseline).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out[output_col] = rel_vol.astype(np.float32)
    return out


def _filter_directional_rows(df: pd.DataFrame, label_col: str, event_col: str) -> tuple[pd.DataFrame, dict]:
    labels = pd.to_numeric(df.get(label_col), errors="coerce")
    directional_mask = labels.isin([0, 1])
    selection_source = "directional_labels_only"

    if event_col in df.columns:
        event_mask = pd.to_numeric(df[event_col], errors="coerce").fillna(0).astype(int).eq(1)
        candidate_mask = directional_mask & event_mask
        if bool(candidate_mask.any()):
            directional_mask = candidate_mask
            selection_source = f"{event_col}_and_directional_labels"

    filtered = df.loc[directional_mask].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No directional rows remained after filtering")

    info = {
        "rows_full": int(len(df)),
        "rows_directional": int(len(filtered)),
        "directional_rate": float(len(filtered) / max(len(df), 1)),
        "selection_source": selection_source,
        "class_counts": {
            str(int(k)): int(v)
            for k, v in filtered[label_col].value_counts().sort_index().items()
        },
    }
    return filtered, info


def _resolve_feature_matrix(
    df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    frame = pd.DataFrame(index=df.index.copy())
    mapping: dict[str, str] = {}
    missing: list[str] = []

    for feature in feature_names:
        actual_col = None
        if feature in {"hour_sin", "hour_cos"}:
            if feature in df.columns:
                actual_col = feature
        elif feature == "rel_vol":
            if "raw__rel_vol" in df.columns:
                actual_col = "raw__rel_vol"
            elif feature in df.columns:
                actual_col = feature
        else:
            raw_col = f"raw__{feature}"
            if raw_col in df.columns:
                actual_col = raw_col
            elif feature in df.columns:
                actual_col = feature

        if actual_col is None:
            missing.append(feature)
            continue

        series = pd.to_numeric(df[actual_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        frame[feature] = series.astype(np.float32)
        mapping[feature] = actual_col

    return frame, mapping, missing


def _chronological_split(df: pd.DataFrame, train_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    n = len(df)
    if n < 20:
        raise ValueError(f"Need at least 20 directional rows for lean training, got {n}")

    split_idx = min(max(int(n * float(train_frac)), 1), n - 1)
    train_df = df.iloc[:split_idx].copy()
    holdout_df = df.iloc[split_idx:].copy()

    if train_df.empty or holdout_df.empty:
        raise ValueError("Chronological split produced an empty train or holdout slice")

    info = {
        "split_idx": int(split_idx),
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "train_start": str(train_df["ts_event"].iloc[0]),
        "train_end": str(train_df["ts_event"].iloc[-1]),
        "holdout_start": str(holdout_df["ts_event"].iloc[0]),
        "holdout_end": str(holdout_df["ts_event"].iloc[-1]),
    }
    return train_df, holdout_df, info


def _precision_recall_table(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    metrics = {}
    for idx, label in enumerate((0, 1)):
        metrics[LABEL_MAP[label]] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
    return metrics


def _hourly_holdout_report(
    holdout_df: pd.DataFrame,
    y_pred: np.ndarray,
    *,
    label_col: str,
) -> pd.DataFrame:
    out = holdout_df.loc[:, ["ts_event", label_col]].copy()
    out["pred_label"] = y_pred.astype(int)
    out["hour_utc"] = pd.to_datetime(out["ts_event"], utc=True, errors="coerce").dt.tz_localize(None).dt.hour

    rows: list[dict] = []
    for hour, group in out.groupby("hour_utc", sort=True):
        if len(group) == 0:
            continue
        y_true = group[label_col].to_numpy(dtype=int)
        y_hat = group["pred_label"].to_numpy(dtype=int)
        rows.append({
            "hour_utc": int(hour),
            "rows": int(len(group)),
            "accuracy": float(accuracy_score(y_true, y_hat)),
            "macro_f1": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
            "long_rate_true": float(np.mean(y_true == 0)),
            "long_rate_pred": float(np.mean(y_hat == 0)),
        })

    return pd.DataFrame(rows).sort_values("hour_utc").reset_index(drop=True) if rows else pd.DataFrame()


def _write_text_report(path: str, payload: dict) -> None:
    lines = [
        "Lean CatBoost Report",
        "=" * 72,
        f"Rows full: {payload['filter']['rows_full']:,}",
        f"Rows directional: {payload['filter']['rows_directional']:,} ({payload['filter']['directional_rate']:.1%})",
        f"Selection source: {payload['filter']['selection_source']}",
        f"Train rows: {payload['split']['train_rows']:,}",
        f"Holdout rows: {payload['split']['holdout_rows']:,}",
        f"Train window: {payload['split']['train_start']} -> {payload['split']['train_end']}",
        f"Holdout window: {payload['split']['holdout_start']} -> {payload['split']['holdout_end']}",
        "",
        f"Features ({len(payload['features']['canonical'])}): {payload['features']['canonical']}",
        f"Resolved columns: {payload['features']['resolved']}",
        "",
        "Holdout Metrics",
        "-" * 72,
        f"Accuracy: {payload['metrics']['accuracy']:.4f}",
        f"Macro F1: {payload['metrics']['macro_f1']:.4f}",
        f"LogLoss: {payload['metrics']['logloss']:.4f}",
        f"ROC AUC (LONG): {payload['metrics']['roc_auc_long']:.4f}",
        f"Pred LONG rate: {payload['metrics']['pred_long_rate']:.4f}",
        "",
        f"LONG:  {payload['per_class']['LONG']}",
        f"SHORT: {payload['per_class']['SHORT']}",
        "",
        "Confusion Matrix [rows=true, cols=pred]",
        "-" * 72,
        str(payload["metrics"]["confusion_matrix"]),
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def train_lean_model(args: argparse.Namespace) -> dict:
    feature_names = _load_feature_list(args)
    requested_columns = _build_requested_columns(feature_names, args.label_col, args.event_col)
    df = load_feature_artifact(args.data, columns=requested_columns)
    df = _ensure_timestamp_sorted(df)
    df = _ensure_cyclical_time_features(df)
    df = _ensure_relative_volatility(df, output_col="rel_vol", window=args.rel_vol_window)

    directional_df, filter_info = _filter_directional_rows(df, args.label_col, args.event_col)
    train_df, holdout_df, split_info = _chronological_split(directional_df, args.train_frac)

    X_all, feature_mapping, missing_features = _resolve_feature_matrix(
        directional_df,
        feature_names,
    )
    if missing_features:
        raise ValueError(
            "Missing lean features after artifact resolution: "
            + ", ".join(missing_features)
        )

    X_train = X_all.iloc[: len(train_df)].copy()
    X_holdout = X_all.iloc[len(train_df):].copy()
    y_train = pd.to_numeric(train_df[args.label_col], errors="coerce").astype(int).to_numpy()
    y_holdout = pd.to_numeric(holdout_df[args.label_col], errors="coerce").astype(int).to_numpy()

    train_pool = Pool(X_train, y_train, feature_names=feature_names)
    holdout_pool = Pool(X_holdout, y_holdout, feature_names=feature_names)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=int(args.iterations),
        depth=int(args.depth),
        learning_rate=float(args.learning_rate),
        l2_leaf_reg=float(args.l2_leaf_reg),
        auto_class_weights="Balanced",
        random_seed=int(args.seed),
        od_type="Iter",
        od_wait=int(args.od_wait),
        verbose=int(args.verbose_every),
        task_type="CPU",
    )
    model.fit(train_pool, eval_set=holdout_pool, use_best_model=True)

    holdout_proba = model.predict_proba(X_holdout)
    classes = [int(x) for x in list(model.classes_)]
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    ordered_proba = np.column_stack([
        holdout_proba[:, class_to_idx[0]],
        holdout_proba[:, class_to_idx[1]],
    ])
    holdout_pred = model.predict(X_holdout).reshape(-1).astype(int)
    holdout_long_proba = ordered_proba[:, 0]
    holdout_short_proba = ordered_proba[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_holdout, holdout_pred)),
        "macro_f1": float(f1_score(y_holdout, holdout_pred, average="macro", zero_division=0)),
        "logloss": float(log_loss(y_holdout, ordered_proba, labels=[0, 1])),
        "roc_auc_long": float(roc_auc_score((y_holdout == 0).astype(int), holdout_long_proba)),
        "pred_long_rate": float(np.mean(holdout_pred == 0)),
        "confusion_matrix": confusion_matrix(y_holdout, holdout_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_holdout,
            holdout_pred,
            labels=[0, 1],
            target_names=["LONG", "SHORT"],
            zero_division=0,
            output_dict=True,
        ),
    }

    out_dir = _ensure_dir(os.path.abspath(os.path.expanduser(args.output)))
    model_path = os.path.join(out_dir, "lean_catboost.cbm")
    model.save_model(model_path)

    feature_importance = model.get_feature_importance(train_pool, type="FeatureImportance")
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importance.astype(float),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    importance_df.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)

    prediction_df = holdout_df.loc[:, ["ts_event", args.label_col]].copy()
    prediction_df["pred_label"] = holdout_pred
    prediction_df["pred_name"] = prediction_df["pred_label"].map(LABEL_MAP)
    prediction_df["prob_long"] = holdout_long_proba.astype(float)
    prediction_df["prob_short"] = holdout_short_proba.astype(float)
    prediction_df.to_csv(os.path.join(out_dir, "holdout_predictions.csv"), index=False)

    hourly_df = _hourly_holdout_report(holdout_df, holdout_pred, label_col=args.label_col)
    if not hourly_df.empty:
        hourly_df.to_csv(os.path.join(out_dir, "holdout_hourly_metrics.csv"), index=False)

    payload = {
        "data": os.path.abspath(os.path.expanduser(args.data)),
        "output_dir": out_dir,
        "filter": filter_info,
        "split": split_info,
        "features": {
            "canonical": feature_names,
            "resolved": feature_mapping,
        },
        "model": {
            "classes": [int(x) for x in list(model.classes_)],
            "best_iteration": int(getattr(model, "best_iteration_", -1)),
            "tree_count": int(model.tree_count_),
            "params": {
                "iterations": int(args.iterations),
                "depth": int(args.depth),
                "learning_rate": float(args.learning_rate),
                "l2_leaf_reg": float(args.l2_leaf_reg),
                "od_wait": int(args.od_wait),
                "seed": int(args.seed),
            },
        },
        "metrics": metrics,
        "per_class": _precision_recall_table(y_holdout, holdout_pred),
        "artifacts": {
            "model": model_path,
            "metrics_json": os.path.join(out_dir, "lean_metrics.json"),
            "metrics_txt": os.path.join(out_dir, "lean_metrics.txt"),
            "predictions_csv": os.path.join(out_dir, "holdout_predictions.csv"),
            "feature_importance_csv": os.path.join(out_dir, "feature_importance.csv"),
            "hourly_metrics_csv": (
                os.path.join(out_dir, "holdout_hourly_metrics.csv")
                if not hourly_df.empty else None
            ),
        },
    }

    with open(payload["artifacts"]["metrics_json"], "w") as f:
        json.dump(payload, f, indent=2)
    _write_text_report(payload["artifacts"]["metrics_txt"], payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QuantSystem V19 lean CatBoost trainer")
    p.add_argument("--data", required=True, help="Artifact root, manifest, final dir, or parquet shard")
    p.add_argument("--output", default="outputs_v19_lean", help="Output directory for lean artifacts")
    p.add_argument("--label-col", default="bias_label", help="Target label column")
    p.add_argument("--event-col", default="train_event_flag", help="Directional event gate column")
    p.add_argument("--train-frac", type=float, default=0.80, help="Chronological train fraction")
    p.add_argument("--features", nargs="*", default=None, help="Optional override feature list")
    p.add_argument("--features-file", default=None, help="Optional newline-separated feature list")
    p.add_argument("--rel-vol-window", type=int, default=200, help="Rolling window for rel_vol")
    p.add_argument("--iterations", type=int, default=700)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--l2-leaf-reg", type=float, default=5.0)
    p.add_argument("--od-wait", type=int, default=75)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cpu"], default="cpu")
    p.add_argument("--verbose-every", type=int, default=50)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = train_lean_model(args)
    print("\n✅ Lean model training completed")
    print(f"  directional_rows={payload['filter']['rows_directional']:,}")
    print(
        "  split: "
        f"train={payload['split']['train_rows']:,} "
        f"| holdout={payload['split']['holdout_rows']:,}"
    )
    print(f"  features={payload['features']['canonical']}")
    print(
        "  holdout: "
        f"accuracy={payload['metrics']['accuracy']:.4f} "
        f"| macro_f1={payload['metrics']['macro_f1']:.4f} "
        f"| logloss={payload['metrics']['logloss']:.4f}"
    )
    print(f"  report={payload['artifacts']['metrics_txt']}")


if __name__ == "__main__":
    main()
