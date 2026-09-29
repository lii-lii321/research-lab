# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from models.schemas import ExperimentRecord, ResearchQuestion
from services.executor import run_experiment
from services.planner import _rule_plan
from services.profiler import profile_dataset
from services.report import build_html, build_markdown, build_report


def make_plan(method: str, variables: list[str]) -> "ExperimentPlan":
    from models.schemas import ExperimentPlan

    return ExperimentPlan(
        experiment_id="EXP-RQ1",
        question_id="RQ1",
        hypothesis="出勤率与最终成绩存在线性相关",
        method=method,
        h0="ρ = 0",
        h1="ρ ≠ 0",
        alpha=0.05,
        variables=variables,
        source="llm",
    )


def build_world():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "student_id": [f"S{i:04d}" for i in range(60)],
            "attendance_rate": np.round(np.clip(rng.normal(85, 8, 60), 40, 100), 1),
            "final_score": np.round(rng.normal(70, 10, 60), 1),
            "gender": rng.choice(["M", "F"], 60),
        }
    )
    profile = profile_dataset(df)
    plan = make_plan("pearson", ["attendance_rate", "final_score"])
    result = run_experiment(plan, df)
    record = ExperimentRecord(plan=plan, result=result)
    question = ResearchQuestion(
        id="RQ1",
        question="出勤率与最终成绩是否相关？",
        rationale="核心变量",
        variables=["attendance_rate", "final_score"],
        suggested_method="相关分析",
    )
    return df, profile, question, record


def test_markdown_contains_real_numbers_and_structure():
    _df, profile, question, record = build_world()
    md = build_markdown(profile, "t.csv", [question], [record], "2026-09-29 10:00")
    assert "AI 数据科学研究报告" in md
    assert "## 一、数据画像" in md
    assert "## 二、研究问题" in md
    assert "## 三、实验与结果" in md
    assert "## 四、结论与局限" in md
    assert "RQ1" in md and "出勤率与最终成绩是否相关？" in md
    assert f"{record.result.statistic:.3f}" in md
    assert "scipy" in md and "复现说明" in md
    assert "不构成因果推断" in md
    assert "探索性分析" in md and "BH" in md  # BH 校正列与多重比较声明
    assert record.result.interpretation in md
    assert ("LLM" in md) if record.result.interpretation_source == "llm" else ("规则模板" in md)


def test_markdown_handles_empty_and_failed():
    _df, profile, _q, _r = build_world()
    md = build_markdown(profile, "t.csv", [], [], "2026-09-29 10:00")
    assert "未生成研究问题" in md
    assert "未执行实验" in md
    failed_plan = make_plan("logistic_regression", ["attendance_rate", "final_score"])
    from models.schemas import ExperimentResult

    failed_result = ExperimentResult(
        experiment_id=failed_plan.experiment_id,
        question_id=failed_plan.question_id,
        method=failed_plan.method,
        status="failed",
        reason="逻辑回归属于 ML 实验范畴，将在 Phase 2 提供",
        alpha=0.05,
    )
    md2 = build_markdown(
        profile,
        "t.csv",
        [],
        [ExperimentRecord(plan=failed_plan, result=failed_result)],
        "2026-09-29 10:00",
    )
    assert "未完成" in md2 and "Phase 2" in md2


def test_html_escapes_and_contains_tables():
    _df, profile, question, record = build_world()
    html = build_html(profile, "t.csv", [question], [record], "2026-09-29 10:00")
    assert "<table" in html
    assert "AI 数据科学研究报告" in html
    assert "<!DOCTYPE html>" in html
    assert "scipy" in html


def test_html_escapes_hostile_column_names():
    import pandas as pd

    df = pd.DataFrame({"a<b": [1.0, 2.0, 3.0, 4.0, 5.0], "c&d": [2.0, 4.0, 6.0, 8.0, 10.0]})
    profile = profile_dataset(df)
    html = build_html(profile, "t.csv", [], [], "2026-09-29 10:00")
    assert "a&lt;b" in html and "c&amp;d" in html
    assert "a<b" not in html.replace("a&lt;b", "")


def test_build_report_pair_and_filename_friendly():
    _df, profile, question, record = build_world()
    md, html = build_report(profile, "student_performance.csv", [question], [record])
    assert md and html


def test_report_with_references_section():
    from models.schemas import PaperRef

    _df, profile, question, record = build_world()
    refs = [
        PaperRef(
            title="Predicting student performance with ML",
            authors=["Alice Wang", "Bob Li", "Carol Zhang", "Dave Chen"],
            year="2024",
            summary="A study on score prediction. " * 20,
            url="http://arxiv.org/abs/2401.0001v1",
        )
    ]
    md, html = build_report(profile, "t.csv", [question], [record], references=refs)
    assert "相关工作（文献引用）" in md
    assert "arXiv 关键词检索" in md
    assert "Predicting student performance with ML" in md
    assert "（2024）" in md and "等" in md  # 第四作者折叠为"等"
    assert "…" in md  # 摘要截断
    assert "相关工作（文献引用）" in html
    assert "<a href=" in html and "2401.0001" in html
