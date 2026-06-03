"""Feature artifact schema validation for QuantSystem v20."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd


FORBIDDEN_FEATURE_TOKENS = (
    "label",
    "target",
    "future",
    "forward",
    "end_ts",
    "path_outcome",
)


@dataclass(frozen=True)
class ArtifactSchemaReport:
    rows: int
    missing_required_columns: tuple[str, ...]
    forbidden_feature_columns: tuple[str, ...]
    passed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def validate_artifact_schema(
    df: pd.DataFrame,
    *,
    feature_columns: list[str] | tuple[str, ...] | None = None,
) -> ArtifactSchemaReport:
    required = ("ts_event", "label_end_ts", "bias_label")
    missing = tuple(col for col in required if col not in df.columns)
    if feature_columns is None:
        feature_columns = [col for col in df.columns if str(col).startswith("raw__")]
    forbidden = tuple(
        str(col)
        for col in feature_columns
        if any(token in str(col).lower() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    return ArtifactSchemaReport(
        rows=int(len(df)),
        missing_required_columns=missing,
        forbidden_feature_columns=forbidden,
        passed=bool(not missing and not forbidden),
    )
