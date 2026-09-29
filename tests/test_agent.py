# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from services.agent import run_research_agent
from services.profiler import profile_dataset
from services.tracking import TrackingStore


def build_df(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "student_id": [f"S{i:04d}" for i in range(n)],
            "hours": rng.uniform(0, 10, n).round(2),
            "attendance": rng.uniform(50, 100, n).round(1),
            "gender": rng.choice(["M", "F"], n),
        }
    )
    df["score"] = (3 * df["hours"] + 0.5 * df["attendance"] + rng.normal(0, 1, n)).round(2)
    return df


def test_agent_full_pipeline_rule_mode(tmp_path):
    df = build_df()
    profile = profile_dataset(df)
    store = TrackingStore(tmp_path / "agent.db")
    result = run_research_agent(
        df,
        profile,
        task_description="研究影响学生成绩的因素",
        max_questions=2,
        client=None,
        store=store,
        dataset_name="students.csv",
    )
    assert result.status == "ok"
    names = [s.name for s in result.steps]
    assert names == ["数据画像", "研究问题", "统计实验", "ML 基线", "研究报告"]
    assert all(s.status == "ok" for s in result.steps)
    assert 1 <= len(result.questions) <= 2
    assert all(q.source == "rule" for q in result.questions)
    assert len(result.records) == len(result.questions)
    assert result.ml_result is not None and result.ml_result.status == "ok"
    assert result.ml_result.target == "score"
    assert "ML 基线实验" in result.report_markdown
    assert result.filename_base == "students"
    assert result.report_html.startswith("<!DOCTYPE html>")
    assert store.list_experiments()  # ML 实验已入库
    tracked_uids = {r.uid for r in store.list_experiments()}
    assert result.ml_result.tracked_uid in tracked_uids


def test_agent_survives_step_failures(tmp_path):
    df = build_df(10)  # 样本过少：统计实验与 ML 会失败，但流程要出报告
    profile = profile_dataset(df)
    result = run_research_agent(
        df, profile, task_description="小样本", max_questions=2,
        client=None, store=TrackingStore(tmp_path / "a.db"), dataset_name="small.csv",
    )
    failed_steps = [s.name for s in result.steps if s.status == "failed"]
    assert result.status in ("ok", "failed")
    assert any("统计实验" in name or "ML 基线" == name for name in failed_steps)
    assert result.report_markdown  # 报告始终产出


def test_agent_clustering_when_no_target():
    df = build_df().drop(columns=["score", "student_id", "gender"])
    profile = profile_dataset(df)
    result = run_research_agent(df, profile, task_description="聚类探索", max_questions=2, client=None)
    assert result.status == "ok"
    assert result.ml_result is not None
    assert result.ml_result.task == "clustering"
