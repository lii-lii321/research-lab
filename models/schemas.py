# -*- coding: utf-8 -*-
"""数据画像相关的 Pydantic 模型。"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

ColumnType = Literal["numeric", "categorical", "boolean", "datetime", "identifier", "text", "empty"]
WarningLevel = Literal["info", "warning", "critical"]


class TopValue(BaseModel):
    value: str
    count: int
    rate: float


class NumericStats(BaseModel):
    count: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    q1: Optional[float] = None
    median: Optional[float] = None
    q3: Optional[float] = None
    max: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None


class ColumnProfile(BaseModel):
    name: str
    type: ColumnType
    n_missing: int
    missing_rate: float
    n_unique: int
    unique_ratio: float
    numeric: Optional[NumericStats] = None
    top_values: Optional[list[TopValue]] = None
    outlier_count: Optional[int] = None
    outlier_values: list[float] = Field(default_factory=list)
    description: str = ""


class DatasetOverview(BaseModel):
    n_rows: int
    n_cols: int
    missing_cells: int
    missing_rate: float
    duplicate_rows: int
    duplicate_rate: float
    memory_mb: float
    type_counts: dict[str, int]


class WarningItem(BaseModel):
    level: WarningLevel
    code: str
    message: str
    columns: list[str] = Field(default_factory=list)


class TargetCandidate(BaseModel):
    column: str
    reason: str


class CorrelationPair(BaseModel):
    column_a: str
    column_b: str
    coefficient: float


class ProfileReport(BaseModel):
    dataset: DatasetOverview
    columns: list[ColumnProfile]
    warnings: list[WarningItem]
    target_candidates: list[TargetCandidate]
    correlations: list[CorrelationPair]


class ResearchQuestion(BaseModel):
    id: str
    question: str
    rationale: str = ""
    variables: list[str] = Field(default_factory=list)
    suggested_method: str = ""


class ExperimentPlan(BaseModel):
    experiment_id: str
    question_id: str
    hypothesis: str
    method: str
    h0: str
    h1: str
    alpha: float = 0.05
    variables: list[str] = Field(default_factory=list)
    notes: str = ""
    source: Literal["llm", "rule"] = "rule"


class GroupStat(BaseModel):
    group: str
    n: int
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None


class ExperimentResult(BaseModel):
    experiment_id: str
    question_id: str
    method: str
    status: Literal["ok", "failed"]
    reason: str = ""
    alpha: float = 0.05
    n_used: int = 0
    n_dropped: int = 0
    statistic: Optional[float] = None
    statistic_name: str = ""
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    effect_name: str = ""
    decision: Literal["reject_h0", "fail_to_reject_h0", "none"] = "none"
    groups: list[GroupStat] = Field(default_factory=list)
    contingency: Optional[list[list[int]]] = None
    extra: dict[str, float] = Field(default_factory=dict)
    interpretation: str = ""
    interpretation_source: Literal["llm", "rule", "none"] = "none"


class ExperimentRecord(BaseModel):
    plan: ExperimentPlan
    result: ExperimentResult


class ReportBundle(BaseModel):
    markdown: str
    html: str
    filename_base: str


class ExcludedFeature(BaseModel):
    column: str
    reason: str


class MLModelResult(BaseModel):
    model: str
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    train_seconds: float = 0.0


class MLExperimentResult(BaseModel):
    status: Literal["ok", "failed"]
    reason: str = ""
    task: Literal["regression", "classification", "clustering"]
    target: Optional[str] = None
    features_numeric: list[str] = Field(default_factory=list)
    features_categorical: list[str] = Field(default_factory=list)
    excluded: list[ExcludedFeature] = Field(default_factory=list)
    n_train: int = 0
    n_test: int = 0
    n_clusters: Optional[int] = None
    cluster_sizes: dict[str, int] = Field(default_factory=dict)
    models: list[MLModelResult] = Field(default_factory=list)
    best_model: Optional[str] = None
    best_metric_name: str = ""
    best_metric_value: Optional[float] = None
    dataset_fingerprint: str = ""
    tracked_uid: str = ""
    runtime_seconds: float = 0.0


class TrackedExperiment(BaseModel):
    uid: str
    created_at: str
    kind: str
    task: str
    target: Optional[str] = None
    dataset_name: str
    n_rows: int
    best_model: Optional[str] = None
    best_metric_name: Optional[str] = None
    best_metric_value: Optional[float] = None


TYPE_CN = {
    "numeric": "数值",
    "categorical": "类别",
    "boolean": "布尔",
    "datetime": "时间",
    "identifier": "标识符",
    "text": "文本",
    "empty": "空列",
}
