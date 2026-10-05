"""Dataset Profiler：纯 pandas/numpy 的确定性数据画像，不依赖 LLM。"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from models.schemas import (
    ColumnProfile,
    ColumnType,
    CorrelationPair,
    DatasetOverview,
    NumericStats,
    ProfileReport,
    TargetCandidate,
    TopValue,
    WarningItem,
    WarningLevel,
)

LOW_CARDINALITY_MAX = 50
ORDINAL_MAX_UNIQUE = 10
ORDINAL_MAX_RATIO = 0.2
IDENTIFIER_RATIO = 0.95
MIN_ROWS_FOR_ID = 20
ID_STRING_MAX_LEN = 32
DATETIME_PARSE_RATIO = 0.8
HIGH_MISSING_RATE = 0.4
CRITICAL_MISSING_RATE = 0.9
HIGH_CORRELATION = 0.8
DOMINANT_CLASS_RATE = 0.9
SKEW_ALERT = 2.0
OUTLIER_RATE_ALERT = 0.01
IQR_K = 1.5
TOP_VALUES_N = 5
OUTLIER_PREVIEW_N = 10
MAX_CORR_PAIRS = 10
SAMPLE_ROWS = 500_000

WARNING_SUGGESTIONS = {
    "DUPLICATE_ROWS": "先确认是否重复采集；确认后按业务键去重再分析。",
    "HIGH_MISSING": "缺失超过 60% 建议删列；否则用中位数/模型插补，并加缺失指示列。",
    "CONSTANT_COLUMN": "检查采集链路是否故障；建模前直接删除该列。",
    "IDENTIFIER_COLUMN": "从特征中排除，仅用于回查记录。",
    "SKEWED_DISTRIBUTION": "考虑对数/Box-Cox 变换，或改用秩方法（本项目计划层已自动升级）。",
    "OUTLIERS_DETECTED": "先核实是否录入错误；真实离群可用稳健检验或分箱处理。",
    "CLASS_IMBALANCE": "使用宏平均指标与分层抽样；严重不平衡考虑重采样或类权重。",
    "HIGH_CORRELATION": "两列择一或做降维；回归注意共线性（可看 VIF）。",
}

_ID_NAME_HINTS = ("id", "no", "number", "code", "uuid")
_TARGET_NUMERIC_HINTS = ("score", "price", "amount", "revenue", "target")
_TARGET_CATEGORICAL_HINTS = ("label", "target", "is_", "has_", "churn", "default", "passed", "fail")
_TARGET_EXACT = ("y",)


def _round(x, nd: int = 4):
    if x is None:
        return None
    x = float(x)
    if np.isnan(x) or np.isinf(x):
        return None
    return round(x, nd)


def _name_is_id_like(name: str) -> bool:
    lowered = name.lower()
    return any(h in lowered for h in _ID_NAME_HINTS)


def infer_column_type(series: pd.Series) -> ColumnType:
    non_null = series.dropna()
    if non_null.empty:
        return "empty"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        n_unique = int(non_null.nunique())
        ratio = n_unique / len(non_null)
        if n_unique <= 1:
            return "numeric"
        if n_unique <= ORDINAL_MAX_UNIQUE and ratio <= ORDINAL_MAX_RATIO:
            return "categorical"
        if (
            ratio >= IDENTIFIER_RATIO
            and len(non_null) >= MIN_ROWS_FOR_ID
            and _name_is_id_like(str(series.name))
        ):
            return "identifier"
        return "numeric"
    values = non_null.astype(str)
    n_unique = int(values.nunique())
    ratio = n_unique / len(values)
    if (
        ratio >= IDENTIFIER_RATIO
        and len(values) >= MIN_ROWS_FOR_ID
        and float(values.str.len().mean()) <= ID_STRING_MAX_LEN
    ):
        return "identifier"
    parsed = pd.to_datetime(values.head(200), errors="coerce", format="mixed")
    if parsed.notna().mean() >= DATETIME_PARSE_RATIO:
        return "datetime"
    if n_unique <= LOW_CARDINALITY_MAX:
        return "categorical"
    return "text"


def _profile_numeric(non_null: pd.Series) -> tuple[NumericStats, int, list[float]]:
    values = non_null.astype(float)
    q1, q2, q3 = values.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - IQR_K * iqr, q3 + IQR_K * iqr
    if iqr > 0:
        outlier_mask = (values < lo) | (values > hi)
    else:
        outlier_mask = pd.Series(False, index=values.index)
    outlier_values = [float(v) for v in values[outlier_mask].tolist()[:OUTLIER_PREVIEW_N]]
    stats = NumericStats(
        count=int(values.count()),
        mean=_round(values.mean()),
        std=_round(values.std()),
        min=_round(values.min()),
        q1=_round(q1),
        median=_round(q2),
        q3=_round(q3),
        max=_round(values.max()),
        skewness=_round(values.skew()),
        kurtosis=_round(values.kurtosis()),
    )
    return stats, int(outlier_mask.sum()), outlier_values


def profile_column(series: pd.Series) -> ColumnProfile:
    name = str(series.name)
    non_null = series.dropna()
    ctype = infer_column_type(series)
    n = len(series)
    n_missing = int(series.isna().sum())
    n_unique = int(series.nunique(dropna=True))
    ratio = n_unique / len(non_null) if len(non_null) else 0.0
    profile = ColumnProfile(
        name=name,
        type=ctype,
        n_missing=n_missing,
        missing_rate=round(n_missing / n, 4) if n else 1.0,
        n_unique=n_unique,
        unique_ratio=round(ratio, 4),
    )
    if ctype == "numeric":
        stats, outlier_count, outlier_values = _profile_numeric(non_null)
        profile.numeric = stats
        profile.outlier_count = outlier_count
        profile.outlier_values = outlier_values
    elif ctype in ("categorical", "boolean"):
        vc = non_null.astype(str).value_counts().head(TOP_VALUES_N)
        total = int(non_null.size) or 1
        profile.top_values = [
            TopValue(value=v, count=int(c), rate=round(c / total, 4)) for v, c in vc.items()
        ]
    elif ctype == "datetime":
        dt = pd.to_datetime(non_null, errors="coerce").dropna()
        if len(dt):
            profile.description = f"{dt.min()} ~ {dt.max()}"
    return profile


def _top_correlations(df: pd.DataFrame, columns: list[ColumnProfile]) -> list[CorrelationPair]:
    numeric_names = [
        c.name for c in columns if c.type == "numeric" and c.numeric is not None and c.numeric.std not in (None, 0)
    ]
    if len(numeric_names) < 2:
        return []
    corr = df[numeric_names].corr(numeric_only=True)
    cols = corr.columns.tolist()
    pairs: list[CorrelationPair] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.isna(r) or abs(float(r)) < HIGH_CORRELATION:
                continue
            pairs.append(
                CorrelationPair(column_a=cols[i], column_b=cols[j], coefficient=round(float(r), 4))
            )
    pairs.sort(key=lambda p: abs(p.coefficient), reverse=True)
    return pairs[:MAX_CORR_PAIRS]


def _suggest_targets(columns: list[ColumnProfile]) -> list[TargetCandidate]:
    hits: list[tuple[int, int, TargetCandidate]] = []
    for idx, c in enumerate(columns):
        lname = c.name.lower()
        matched = None
        if c.type == "numeric" and (
            any(h in lname for h in _TARGET_NUMERIC_HINTS) or lname in _TARGET_EXACT
        ):
            matched = "数值型且命名含常见目标词"
        elif c.type in ("categorical", "boolean") and any(h in lname for h in _TARGET_CATEGORICAL_HINTS):
            matched = "低基数类别型且命名含常见标签词"
        if matched is None:
            continue
        # 词尾命中（如 final_score 之于 score）通常比词中命中更接近"结果变量"
        priority = (
            0
            if any(
                lname.endswith(h)
                for h in _TARGET_NUMERIC_HINTS + _TARGET_CATEGORICAL_HINTS
            )
            else 1
        )
        hits.append((priority, idx, TargetCandidate(column=c.name, reason=matched)))
    hits.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in hits[:5]]


def _collect_warnings(
    columns: list[ColumnProfile], correlations: list[CorrelationPair], duplicate_rows: int
) -> list[WarningItem]:
    out: list[WarningItem] = []
    if duplicate_rows:
        out.append(
            WarningItem(
                level="warning",
                code="DUPLICATE_ROWS",
                message=f"存在 {duplicate_rows} 行完全重复记录",
            )
        )
    for c in columns:
        if c.missing_rate >= HIGH_MISSING_RATE:
            level: WarningLevel = "critical" if c.missing_rate >= CRITICAL_MISSING_RATE else "warning"
            out.append(
                WarningItem(
                    level=level,
                    code="HIGH_MISSING",
                    message=f"列 {c.name} 缺失率 {c.missing_rate:.1%}",
                    columns=[c.name],
                )
            )
        if c.type == "numeric" and c.numeric is not None and c.numeric.std == 0:
            out.append(
                WarningItem(
                    level="warning",
                    code="CONSTANT_COLUMN",
                    message=f"列 {c.name} 取值恒定，无信息量",
                    columns=[c.name],
                )
            )
        if c.type == "identifier":
            out.append(
                WarningItem(
                    level="info",
                    code="IDENTIFIER_COLUMN",
                    message=f"列 {c.name} 疑似唯一标识符，建模时应排除",
                    columns=[c.name],
                )
            )
        if (
            c.type == "numeric"
            and c.numeric is not None
            and c.numeric.skewness is not None
            and abs(c.numeric.skewness) >= SKEW_ALERT
        ):
            out.append(
                WarningItem(
                    level="warning",
                    code="SKEWED_DISTRIBUTION",
                    message=f"列 {c.name} 分布明显偏斜（skew={c.numeric.skewness}）",
                    columns=[c.name],
                )
            )
        if c.type == "numeric" and c.outlier_count:
            rate = c.outlier_count / max(c.numeric.count if c.numeric else 1, 1)
            if rate >= OUTLIER_RATE_ALERT:
                out.append(
                    WarningItem(
                        level="warning",
                        code="OUTLIERS_DETECTED",
                        message=f"列 {c.name} 检出 {c.outlier_count} 个 IQR 离群值",
                        columns=[c.name],
                    )
                )
        if c.type in ("categorical", "boolean") and c.top_values and len(c.top_values) >= 2:
            if c.top_values[0].rate >= DOMINANT_CLASS_RATE:
                out.append(
                    WarningItem(
                        level="warning",
                        code="CLASS_IMBALANCE",
                        message=f"列 {c.name} 类别不平衡：{c.top_values[0].value} 占 {c.top_values[0].rate:.0%}",
                        columns=[c.name],
                    )
                )
    for pair in correlations:
        out.append(
            WarningItem(
                level="warning",
                code="HIGH_CORRELATION",
                message=f"{pair.column_a} 与 {pair.column_b} 相关系数 {pair.coefficient:.2f}，存在冗余风险",
                columns=[pair.column_a, pair.column_b],
            )
        )
    for item in out:
        item.suggestion = WARNING_SUGGESTIONS.get(item.code, "")
    return out


def profile_dataset(df: pd.DataFrame, sample: bool = True) -> ProfileReport:
    """数据画像；超过 SAMPLE_ROWS 行时默认均匀采样并在 overview.sampled / overview.sample_note 标注。"""
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("数据集为空（0 行或 0 列），无法生成画像")
    sampled = False
    if sample and df.shape[0] > SAMPLE_ROWS:
        df = df.sample(n=SAMPLE_ROWS, random_state=42).reset_index(drop=True)
        sampled = True
    columns = [profile_column(df[col]) for col in df.columns]
    type_counts: dict[str, int] = dict(Counter(str(c.type) for c in columns))
    n_rows = int(df.shape[0])
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    overview = DatasetOverview(
        n_rows=n_rows,
        n_cols=int(df.shape[1]),
        missing_cells=missing_cells,
        missing_rate=round(missing_cells / (n_rows * df.shape[1]), 4),
        duplicate_rows=duplicate_rows,
        duplicate_rate=round(duplicate_rows / n_rows, 4),
        memory_mb=round(float(df.memory_usage(deep=True).sum()) / 1024 / 1024, 3),
        type_counts=type_counts,
        sampled=sampled,
        sample_note=f"基于 {SAMPLE_ROWS:,} 行均匀采样（随机种子 42）" if sampled else "",
    )
    correlations = _top_correlations(df, columns)
    warnings = _collect_warnings(columns, correlations, duplicate_rows)
    targets = _suggest_targets(columns)
    return ProfileReport(
        dataset=overview,
        columns=columns,
        warnings=warnings,
        target_candidates=targets,
        correlations=correlations,
    )
