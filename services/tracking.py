# -*- coding: utf-8 -*-
"""Experiment Tracking：SQLite 存储，记录每次实验的完整上下文。"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from models.schemas import TrackedExperiment

DEFAULT_DB = Path("data/tracking/experiments.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    uid TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    task TEXT NOT NULL,
    target TEXT,
    dataset_name TEXT NOT NULL,
    dataset_fingerprint TEXT NOT NULL,
    n_rows INTEGER NOT NULL,
    n_cols INTEGER NOT NULL,
    feature_set TEXT NOT NULL,
    models_results TEXT NOT NULL,
    best_model TEXT,
    best_metric_name TEXT,
    best_metric_value REAL,
    notes TEXT,
    runtime_seconds REAL
)
"""

_LIST_SQL = (
    "SELECT uid, created_at, kind, task, target, dataset_name, n_rows, "
    "best_model, best_metric_name, best_metric_value "
    "FROM experiments ORDER BY created_at DESC, uid DESC LIMIT ?"
)


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(str([str(c) for c in df.columns]).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return digest.hexdigest()[:16]


class TrackingStore:
    def __init__(self, path: Path | str | None = None):
        env_path = os.getenv("RESEARCH_LAB_DB", "").strip()
        self.path = Path(path or env_path or DEFAULT_DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.path)
            self._conn.execute(_SCHEMA)
            self._conn.commit()
        return self._conn

    def track(
        self,
        *,
        kind: str,
        task: str,
        dataset_name: str,
        df: pd.DataFrame,
        feature_set: dict,
        models_results: list[dict],
        target: str | None = None,
        best_model: str | None = None,
        best_metric_name: str | None = None,
        best_metric_value: float | None = None,
        notes: str = "",
        runtime_seconds: float = 0.0,
    ) -> str:
        uid = uuid.uuid4().hex[:12]
        conn = self._connect()
        conn.execute(
            "INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                uid,
                datetime.now().isoformat(timespec="seconds"),
                kind,
                task,
                target,
                dataset_name,
                dataframe_fingerprint(df),
                int(df.shape[0]),
                int(df.shape[1]),
                json.dumps(feature_set, ensure_ascii=False),
                json.dumps(models_results, ensure_ascii=False),
                best_model,
                best_metric_name,
                best_metric_value,
                notes,
                runtime_seconds,
            ),
        )
        conn.commit()
        return uid

    def list_experiments(self, limit: int = 20) -> list[TrackedExperiment]:
        rows = self._connect().execute(_LIST_SQL, (int(limit),)).fetchall()
        return [
            TrackedExperiment(
                uid=r[0], created_at=r[1], kind=r[2], task=r[3], target=r[4],
                dataset_name=r[5], n_rows=r[6], best_model=r[7],
                best_metric_name=r[8], best_metric_value=r[9],
            )
            for r in rows
        ]

    def get_experiment(self, uid: str) -> dict | None:
        row = self._connect().execute(
            "SELECT uid, created_at, kind, task, target, dataset_name, dataset_fingerprint, "
            "n_rows, n_cols, feature_set, models_results, best_model, best_metric_name, "
            "best_metric_value, notes, runtime_seconds FROM experiments WHERE uid = ?",
            (uid,),
        ).fetchone()
        if row is None:
            return None
        return {
            "uid": row[0], "created_at": row[1], "kind": row[2], "task": row[3],
            "target": row[4], "dataset_name": row[5], "dataset_fingerprint": row[6],
            "n_rows": row[7], "n_cols": row[8],
            "feature_set": json.loads(row[9]),
            "models_results": json.loads(row[10]),
            "best_model": row[11], "best_metric_name": row[12],
            "best_metric_value": row[13], "notes": row[14], "runtime_seconds": row[15],
        }
