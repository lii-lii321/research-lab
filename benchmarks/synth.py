"""Benchmark 合成数据集：植入统计效应（确定性种子），生成公式里的系数即真值。

思想移植自 DataAnalystAgent 的 agent/synth.py（效应即真值 / 种子固定可复现），
数据集为本项目的目标用户（教学场景）量身设计：

- tutoring_score：两组差异（group B − A = +12）+ 线性相关（hours 斜率 3.0）+ 零效应对照（attention）
- exam_passed：类别关联（campusA 通过率约 0.75~0.79 vs campusB 约 0.35~0.50）
- course_rating：三组差异（教师均值 60 / 70 / 80）

对同一组 (n, seed) 参数，输出永远相同，因此「显著性 / 方向」由构造保证。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_tutoring_score(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = rng.integers(1, 21, n).astype(float)
    attention = rng.normal(50, 10, n).round(1)
    attention[rng.choice(n, 6, replace=False)] = np.nan
    group = rng.choice(["A", "B"], n)
    score = 55 + 3.0 * hours + 12.0 * (group == "B") + rng.normal(0, 8, n)
    return pd.DataFrame(
        {
            "hours": hours,
            "attention": attention,
            "group": group,
            "score": score.round(1),
        }
    )


def make_exam_passed(n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    study = rng.integers(0, 31, n).astype(float)
    school = rng.choice(["campusA", "campusB"], n)
    logit = -0.9 + 0.06 * study + 1.35 * (school == "campusA")
    prob = 1 / (1 + np.exp(-logit))
    passed = np.where(rng.random(n) < prob, "yes", "no")
    return pd.DataFrame({"study": study, "school": school, "passed": passed})


def make_course_rating(n: int = 240, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    instructor = rng.choice(["T1", "T2", "T3"], n)
    satisfaction_score = (
        60 + 10.0 * (instructor == "T2") + 20.0 * (instructor == "T3") + rng.normal(0, 10, n)
    )
    return pd.DataFrame(
        {"instructor": instructor, "satisfaction_score": satisfaction_score.round(1)}
    )


GENERATORS = {
    "tutoring_score.csv": make_tutoring_score,
    "exam_passed.csv": make_exam_passed,
    "course_rating.csv": make_course_rating,
}
