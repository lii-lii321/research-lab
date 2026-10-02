"""数据画像相关的 Pydantic 模型。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ColumnType = Literal["numeric", "categorical", "boolean", "datetime", "identifier", "text", "empty"]
WarningLevel = Literal["info", "warning", "critical"]


class TopValue(BaseModel):
    value: str
    count: int
    rate: float


class NumericStats(BaseModel):
    count: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    q1: float | None = None
    median: float | None = None
    q3: float | None = None
    max: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None


class ColumnProfile(BaseModel):
    name: str
    type: ColumnType
    n_missing: int
    missing_rate: float
    n_unique: int
    unique_ratio: float
    numeric: NumericStats | None = None
    top_values: list[TopValue] | None = None
    outlier_count: int | None = None
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
    source: Literal["llm", "rule"] = "llm"


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
    required_n: int | None = None


class GroupStat(BaseModel):
    group: str
    n: int
    mean: float | None = None
    std: float | None = None
    median: float | None = None


class ExperimentResult(BaseModel):
    experiment_id: str
    question_id: str
    method: str
    status: Literal["ok", "failed"]
    reason: str = ""
    alpha: float = 0.05
    n_used: int = 0
    n_dropped: int = 0
    statistic: float | None = None
    statistic_name: str = ""
    p_value: float | None = None
    p_value_raw: float | None = None
    effect_size: float | None = None
    effect_name: str = ""
    decision: Literal["reject_h0", "fail_to_reject_h0", "none"] = "none"
    groups: list[GroupStat] = Field(default_factory=list)
    contingency: list[list[int]] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
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
    task: str
    target: str | None = None
    features_numeric: list[str] = Field(default_factory=list)
    features_categorical: list[str] = Field(default_factory=list)
    excluded: list[ExcludedFeature] = Field(default_factory=list)
    n_train: int = 0
    n_test: int = 0
    n_clusters: int | None = None
    cluster_sizes: dict[str, int] = Field(default_factory=dict)
    models: list[MLModelResult] = Field(default_factory=list)
    best_model: str | None = None
    best_metric_name: str = ""
    best_metric_value: float | None = None
    dataset_fingerprint: str = ""
    tracked_uid: str = ""
    runtime_seconds: float = 0.0


class TrackedExperiment(BaseModel):
    uid: str
    created_at: str
    kind: str
    task: str
    target: str | None = None
    dataset_name: str
    n_rows: int
    best_model: str | None = None
    best_metric_name: str | None = None
    best_metric_value: float | None = None


class AgentStep(BaseModel):
    name: str
    status: Literal["ok", "failed"]
    detail: str = ""
    seconds: float = 0.0


class PaperRef(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    year: str = ""
    summary: str = ""
    url: str = ""


class AgentRunResult(BaseModel):
    status: Literal["ok", "failed"]
    reason: str = ""
    task_description: str = ""
    steps: list[AgentStep] = Field(default_factory=list)
    questions: list[ResearchQuestion] = Field(default_factory=list)
    records: list[ExperimentRecord] = Field(default_factory=list)
    ml_result: MLExperimentResult | None = None
    references: list[PaperRef] = Field(default_factory=list)
    report_markdown: str = ""
    report_html: str = ""
    filename_base: str = "report"
    runtime_seconds: float = 0.0


TYPE_CN = {
    "numeric": "数值",
    "categorical": "类别",
    "boolean": "布尔",
    "datetime": "时间",
    "identifier": "标识符",
    "text": "文本",
    "empty": "空列",
}
