"""
feature_artifact_v19.py - Sharded parquet dataset artifacts for QuantSystem V19
"""

from __future__ import annotations

import glob
import json
import os
from typing import Iterable

import pandas as pd


ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"
FINAL_FEATURE_DIR = "final"
FINAL_FEATURE_PATTERN = "features_*.parquet"


def _abs(path: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path)))


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _read_json(path: str) -> dict:
    with open(path) as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _write_json(path: str, payload: dict) -> str:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".parquet", ".pq", ".snappy"}:
        return pd.read_parquet(path)
    if ext in {".zst", ".gz"}:
        return pd.read_csv(path, low_memory=False, compression="infer")
    if ext in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    return pd.read_csv(path, low_memory=False)


def write_table(df: pd.DataFrame, path: str, *, compression: str = "snappy") -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".parquet", ".pq", ".snappy"}:
        try:
            df.to_parquet(path, index=False, compression=compression)
        except Exception:
            df.to_pickle(path)
        return path
    if ext in {".pkl", ".pickle"}:
        df.to_pickle(path)
        return path
    df.to_csv(path, index=False)
    return path


def iter_table_chunks(path: str, chunk_rows: int | None = None) -> Iterable[pd.DataFrame]:
    path = _abs(path)
    if os.path.isdir(path):
        supported = (".parquet", ".pq", ".snappy", ".csv", ".csv.gz", ".csv.zst", ".gz", ".zst")
        files = [
            _abs(os.path.join(path, name))
            for name in sorted(os.listdir(path))
            if os.path.isfile(os.path.join(path, name)) and name.lower().endswith(supported)
        ]
        if not files:
            raise FileNotFoundError(f"No supported table files found under directory: {path}")
        for file_path in files:
            yield from iter_table_chunks(file_path, chunk_rows)
        return

    ext = os.path.splitext(path)[1].lower()
    rows = int(chunk_rows or 0)
    if ext in {".parquet", ".pq", ".snappy"}:
        df = read_table(path)
        if rows <= 0 or len(df) <= rows:
            yield df
            return
        for start in range(0, len(df), rows):
            yield df.iloc[start:start + rows].copy()
        return

    reader = pd.read_csv(
        path,
        low_memory=False,
        chunksize=None if rows <= 0 else rows,
        compression="infer" if ext in {".zst", ".gz"} else None,
    )
    if isinstance(reader, pd.DataFrame):
        yield reader
        return
    for chunk in reader:
        yield chunk


def parquet_shard_paths(directory: str, pattern: str = FINAL_FEATURE_PATTERN) -> list[str]:
    directory = _abs(directory)
    return sorted(_abs(path) for path in glob.glob(os.path.join(directory, pattern)))


def write_parquet_shards(
    df: pd.DataFrame,
    directory: str,
    *,
    stem: str = "features",
    rows_per_shard: int = 250_000,
    compression: str = "snappy",
) -> list[dict]:
    out_dir = _ensure_dir(directory)
    rows = max(int(rows_per_shard), 1)
    records: list[dict] = []
    for shard_idx, start in enumerate(range(0, len(df), rows)):
        shard = df.iloc[start:start + rows].copy()
        name = f"{stem}_{shard_idx:05d}.parquet"
        path = os.path.join(out_dir, name)
        write_table(shard, path, compression=compression)
        record = {
            "name": name,
            "path": _abs(path),
            "rows": int(len(shard)),
        }
        if "ts_event" in shard.columns and len(shard):
            ts = pd.to_datetime(shard["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
            ts = ts.dropna()
            if len(ts):
                record["ts_min"] = str(ts.min())
                record["ts_max"] = str(ts.max())
        records.append(record)
    return records


def resolve_artifact_root(path_or_manifest: str) -> str:
    path = _abs(path_or_manifest)
    if os.path.isdir(path):
        if os.path.basename(path) == FINAL_FEATURE_DIR:
            parent = os.path.dirname(path)
            if os.path.exists(os.path.join(parent, ARTIFACT_MANIFEST_NAME)):
                return parent
        return path
    if os.path.basename(path) == ARTIFACT_MANIFEST_NAME:
        return os.path.dirname(path)
    if path.lower().endswith((".parquet", ".pq", ".snappy", ".csv", ".gz", ".zst")):
        parent = os.path.dirname(path)
        if os.path.basename(parent) == FINAL_FEATURE_DIR:
            grandparent = os.path.dirname(parent)
            if os.path.exists(os.path.join(grandparent, ARTIFACT_MANIFEST_NAME)):
                return grandparent
        return parent
    raise FileNotFoundError(f"Unsupported artifact path: {path_or_manifest}")


def resolve_manifest_path(path_or_manifest: str) -> str | None:
    path = _abs(path_or_manifest)
    if os.path.isfile(path) and os.path.basename(path) == ARTIFACT_MANIFEST_NAME:
        return path
    root = resolve_artifact_root(path)
    manifest_path = os.path.join(root, ARTIFACT_MANIFEST_NAME)
    return manifest_path if os.path.exists(manifest_path) else None


def load_artifact_manifest(path_or_manifest: str) -> dict:
    manifest_path = resolve_manifest_path(path_or_manifest)
    return _read_json(manifest_path) if manifest_path and os.path.exists(manifest_path) else {}


def resolve_final_feature_paths(path_or_manifest: str) -> list[str]:
    path = _abs(path_or_manifest)
    if os.path.isfile(path) and path.lower().endswith((".parquet", ".pq", ".snappy", ".csv", ".gz", ".zst")):
        return [path]

    manifest = load_artifact_manifest(path)
    extra = manifest.get("extra", {}) or {}
    final_shards = extra.get("final_feature_shards") or []
    if final_shards:
        out: list[str] = []
        for entry in final_shards:
            if isinstance(entry, dict):
                shard_path = entry.get("path") or entry.get("name")
            else:
                shard_path = str(entry)
            if not shard_path:
                continue
            shard_path = shard_path if os.path.isabs(shard_path) else os.path.join(resolve_artifact_root(path), shard_path)
            out.append(_abs(shard_path))
        if out:
            return sorted(out)

    root = resolve_artifact_root(path)
    final_dir = os.path.join(root, FINAL_FEATURE_DIR)
    shard_paths = parquet_shard_paths(final_dir)
    if shard_paths:
        return shard_paths

    legacy_csv = os.path.join(root, "training_features_ready.csv")
    if os.path.exists(legacy_csv):
        return [_abs(legacy_csv)]
    legacy_parquet = os.path.join(root, "training_features_ready.parquet")
    if os.path.exists(legacy_parquet):
        return [_abs(legacy_parquet)]
    raise FileNotFoundError(f"No final feature dataset found under: {path_or_manifest}")


def load_feature_artifact(
    path_or_manifest: str,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    shard_paths = resolve_final_feature_paths(path_or_manifest)
    frames: list[pd.DataFrame] = []
    for shard_path in shard_paths:
        frame = read_table(shard_path)
        if columns is not None:
            keep = [col for col in columns if col in frame.columns]
            frame = frame.loc[:, keep]
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=columns or [])
    df = pd.concat(frames, ignore_index=True)
    for col in ("ts_event", "label_end_ts"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_localize(None)
    return df


def checkpoint_dir(root: str) -> str:
    return _ensure_dir(os.path.join(_abs(root), "checkpoints"))


def write_checkpoint(root: str, phase: str, payload: dict) -> str:
    return _write_json(os.path.join(checkpoint_dir(root), f"{phase}.json"), payload)


def read_checkpoint(root: str, phase: str) -> dict:
    path = os.path.join(checkpoint_dir(root), f"{phase}.json")
    return _read_json(path) if os.path.exists(path) else {}
