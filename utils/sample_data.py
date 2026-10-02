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
