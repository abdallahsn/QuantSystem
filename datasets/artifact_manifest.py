"""Artifact manifest schema for QuantSystem v20."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
import datetime as _dt
import json
import os


def _utc_now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_version: str = "v20.0"
    schema_version: str = "v20.0"
    kind: str = "feature_dataset"
    created_at_utc: str = field(default_factory=_utc_now)
    rows: int = 0
    inputs: dict = field(default_factory=dict)
    row_counts: dict = field(default_factory=dict)
    symbol: str | None = None
    contract: str | None = None
    date_range: dict = field(default_factory=dict)
    feature_columns: tuple[str, ...] = ()
    compatibility_only_feature_columns: tuple[str, ...] = ()
    label_columns: tuple[str, ...] = ("bias_label", "direction_label", "tradeability_label")
    metadata_columns: tuple[str, ...] = ()
    timestamp_columns: tuple[str, ...] = ("ts_event", "label_end_ts")
    label_end_ts_column: str = "label_end_ts"
    tick_size: float | None = None
    horizon: int | None = None
    label_params: dict = field(default_factory=dict)
    mbo_flow_features_reliable: bool | None = None
    git_commit: str | None = None
    shards: tuple[dict, ...] = ()
    reports: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    research_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def write_artifact_manifest(manifest: ArtifactManifest, output_dir: str, filename: str = "artifact_manifest.json") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    return path
