"""演示数据集：学生成绩（固定随机种子，可复现）。

刻意埋入了典型数据质量问题供 Profiler 演示：
缺失值、离群值、重复行、类别不平衡、高相关字段对、右偏分布、唯一标识符。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG_SEED = 42
N_STUDENTS = 500
DUPLICATE_ROWS = 4


def build_sample_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    n = N_STUDENTS
    midterm = np.clip(rng.normal(70, 12, n), 0, 100)
    study_time = rng.choice([1, 2, 3, 4], size=n, p=[0.2, 0.35, 0.3, 0.15])
    df = pd.DataFrame(
        {
            "student_id": [f"S{i:04d}" for i in range(1, n + 1)],
            "gender": rng.choice(["M", "F"], size=n, p=[0.52, 0.48]),
            "school_type": rng.choice(["Public", "Private"], size=n, p=[0.7, 0.3]),
            "study_time": study_time,
            "attendance_rate": np.round(np.clip(rng.normal(85, 8, n), 40, 100), 1),
            "sleep_hours": np.round(np.clip(rng.normal(7, 1.2, n), 4, 10), 1),
            "volunteer_hours": np.round(np.clip(rng.normal(5, 3, n), 0, None), 1),
            "family_income": np.round(rng.lognormal(9.5, 0.8, n), 1),
            "scholarship": np.where(rng.random(n) < 0.03, "Yes", "No"),
            "midterm_score": np.round(midterm, 1),
            "final_score": np.round(
                np.clip(0.9 * midterm + 4 * study_time + rng.normal(0, 4, n), 0, 100), 1
            ),
        }
    )
    for col, rate in (("attendance_rate", 0.02), ("family_income", 0.05), ("volunteer_hours", 0.45)):
        idx = rng.choice(n, size=int(n * rate), replace=False)
        df.loc[idx, col] = np.nan
    outlier_idx = rng.choice(n, size=6, replace=False)
    df.loc[outlier_idx, "attendance_rate"] = 20.0
    dup_idx = rng.choice(n, size=DUPLICATE_ROWS, replace=False)
    return pd.concat([df, df.loc[dup_idx]], ignore_index=True)


def build_sales_dataframe(n_days: int = 120) -> pd.DataFrame:
    """演示数据集：门店日销售（季节性 + 周末效应 + 缺失 + 离群）。"""
    rng = np.random.default_rng(7)
    days = pd.date_range("2026-01-01", periods=n_days, freq="D")
    weekend = np.array([1.0 if d.weekday() >= 5 else 0.6 for d in days])
    base = 800 * weekend * (1 + 0.3 * np.sin(np.arange(n_days) / 14))
    promo = rng.choice([0, 1], n_days, p=[0.8, 0.2])
    revenue = np.round(base * (1 + 0.4 * promo) * rng.normal(1, 0.08, n_days), 1)
    df = pd.DataFrame(
        {
            "date": days.strftime("%Y-%m-%d"),
            "weekday": [d.weekday() for d in days],
            "is_promo": promo,
            "visitors": np.round(revenue / rng.uniform(40, 60, n_days), 0),
            "revenue": revenue,
        }
    )
    df.loc[rng.choice(n_days, size=int(n_days * 0.04), replace=False), "visitors"] = np.nan
    df.loc[rng.choice(n_days, size=5, replace=False), "revenue"] = np.nan
    df.loc[rng.choice(n_days, size=4, replace=False), "revenue"] = (
        df["revenue"].median() * 6
    )
    return df


def build_medical_dataframe(n_patients: int = 200) -> pd.DataFrame:
    """演示数据集：随访队列（类别不平衡 + 计数终点 + 协变量相关）。"""
    rng = np.random.default_rng(11)
    age = np.round(rng.normal(55, 12, n_patients), 0)
    bmi = np.round(24 + 0.1 * (age - 55) + rng.normal(0, 3, n_patients), 1)
    exercise = rng.choice(["规律", "偶尔", "不锻炼"], n_patients, p=[0.35, 0.4, 0.25])
    risk = 0.03 + 0.02 * (bmi > 28) + 0.02 * (exercise == "不锻炼") + 0.01 * (age > 60)
    outcome = rng.binomial(1, np.clip(risk, 0, 0.5))
    df = pd.DataFrame(
        {
            "patient_id": [f"P{i:04d}" for i in range(1, n_patients + 1)],
            "age": age,
            "bmi": bmi,
            "exercise": exercise,
            "followup_visits": rng.poisson(3, n_patients),
            "outcome_event": outcome,
        }
    )
    df.loc[rng.choice(n_patients, size=8, replace=False), "bmi"] = np.nan
    return df


DATASET_GALLERY = {
    "学生成绩": build_sample_dataframe,
    "门店销售": build_sales_dataframe,
    "医疗随访": build_medical_dataframe,
}
