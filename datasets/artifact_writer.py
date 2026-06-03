"""Feature artifact writer with v20 manifest metadata."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import os

import pandas as pd

from modules.feature_artifact_v19 import FINAL_FEATURE_DIR, write_parquet_shards
from .artifact_manifest import ArtifactManifest, write_artifact_manifest


@dataclass(frozen=True)
class ArtifactWriteResult:
    output_dir: str
    manifest_path: str
    rows: int
    shards: tuple[dict, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _feature_columns(df: pd.DataFrame) -> tuple[str, ...]:
    forbidden = {
        "ts_event",
        "ts_recv",
        "feature_ts",
        "label_start_ts",
        "mbo_state_ts",
        "label_end_ts",
        "horizon_end_ts",
        "bias_label",
        "direction_label",
        "tradeability_label",
        "event_flag",
        "train_event_flag",
        "forward_return",
        "realized_return",
        "label_outcome",
        "barrier_hit_type",
        "symbol",
        "contract_symbol",
        "instrument_id",
    }
    return tuple(str(col) for col in df.columns if str(col) not in forbidden and not str(col).startswith("label_"))


def write_feature_artifact(
    df: pd.DataFrame,
    output_dir: str,
    *,
    rows_per_shard: int = 250_000,
    config: dict | None = None,
    reports: dict | None = None,
    inputs: dict | None = None,
    row_counts: dict | None = None,
    symbol: str | None = None,
    contract: str | None = None,
    date_range: dict | None = None,
    metadata_columns: tuple[str, ...] | list[str] | None = None,
    tick_size: float | None = None,
    horizon: int | None = None,
    label_params: dict | None = None,
    git_commit: str | None = None,
    extra: dict | None = None,
) -> ArtifactWriteResult:
    if "ts_event" not in df.columns or "label_end_ts" not in df.columns:
        raise ValueError("v20 artifacts require ts_event and label_end_ts")
    final_dir = os.path.join(output_dir, FINAL_FEATURE_DIR)
    shards = write_parquet_shards(df, final_dir, stem="features", rows_per_shard=rows_per_shard)
    manifest = ArtifactManifest(
        rows=int(len(df)),
        inputs=dict(inputs or {}),
        row_counts=dict(row_counts or {}),
        symbol=symbol,
        contract=contract,
        date_range=dict(date_range or {}),
        feature_columns=_feature_columns(df),
        metadata_columns=tuple(str(col) for col in (metadata_columns or ())),
        shards=tuple(shards),
        reports=dict(reports or {}),
        config=dict(config or {}),
        tick_size=tick_size,
        horizon=horizon,
        label_params=dict(label_params or {}),
        git_commit=git_commit,
        extra={
            **dict(extra or {}),
            "final_feature_shards": shards,
        },
        research_notes=(
            "MLOFI follows multi-level order-flow imbalance research.",
            "Labels must be cost-aware and include label_end_ts for purged validation.",
        ),
    )
    manifest_path = write_artifact_manifest(manifest, output_dir)
    return ArtifactWriteResult(
        output_dir=os.path.abspath(output_dir),
        manifest_path=manifest_path,
        rows=int(len(df)),
        shards=tuple(shards),
    )
