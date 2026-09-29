# -*- coding: utf-8 -*-
import json

import numpy as np
import pandas as pd
import pytest

from models.schemas import ResearchQuestion
from services.llm import LLMError
from services.planner import generate_experiment_plan, suggest_method
from services.profiler import profile_dataset
from services.research_questions import (
    generate_research_questions,
    generate_research_questions_rule,
)


class FakeLLM:
    def __init__(self, response: str = ""):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        self.calls.append((system, user))
        return self.response


class BrokenLLM:
    def chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        raise LLMError("连接失败")


def build_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "attendance_rate": np.round(np.clip(rng.normal(85, 8, 60), 40, 100), 1),
            "final_score": np.round(rng.normal(70, 10, 60), 1),
            "gender": rng.choice(["M", "F"], 60),
            "school_type": rng.choice(["Public", "Private", "International"], 60),
        }
    )


RQ_JSON = json.dumps(
    [
        {
            "id": "RQ1",
            "question": "出勤率与最终成绩是否相关？",
            "rationale": "核心变量",
            "variables": ["attendance_rate", "final_score"],
            "suggested_method": "相关分析",
        },
        {
            "id": "RQ2",
            "question": "变量不存在的问题",
            "rationale": "",
            "variables": ["ghost_col"],
            "suggested_method": "",
        },
    ],
    ensure_ascii=False,
)


def test_generate_rq_drops_invalid_and_keeps_valid():
    df = build_df()
    report = profile_dataset(df)
    questions = generate_research_questions(df, report, FakeLLM(f"```json\n{RQ_JSON}\n```"))
    assert [q.id for q in questions] == ["RQ1"]
    assert questions[0].variables == ["attendance_rate", "final_score"]


def test_generate_rq_all_invalid_raises():
    df = build_df()
    report = profile_dataset(df)
    bad = json.dumps([{"id": "RQ1", "question": "x", "variables": ["ghost"]}])
    with pytest.raises(ValueError):
        generate_research_questions(df, report, FakeLLM(bad))


def test_suggest_method_rules():
    df = build_df()
    report = profile_dataset(df)
    assert suggest_method(report, ["attendance_rate", "final_score"]) == "pearson"
    assert suggest_method(report, ["gender", "final_score"]) == "welch_ttest"
    assert suggest_method(report, ["school_type", "final_score"]) == "anova"
    assert suggest_method(report, ["gender", "school_type"]) == "chi2"


def test_plan_rule_mode_without_client():
    df = build_df()
    report = profile_dataset(df)
    rq = ResearchQuestion(
        id="RQ1", question="出勤率与成绩相关吗？", variables=["attendance_rate", "final_score"]
    )
    plan = generate_experiment_plan(rq, report, None)
    assert plan.source == "rule"
    assert plan.method == "pearson"
    assert "ρ = 0" in plan.h0


def test_plan_llm_valid_method_kept():
    df = build_df()
    report = profile_dataset(df)
    rq = ResearchQuestion(id="RQ1", question="q", variables=["attendance_rate", "final_score"])
    plan_json = json.dumps(
        {"hypothesis": "h", "method": "spearman", "h0": "H0", "h1": "H1", "alpha": 0.05, "notes": "n"},
        ensure_ascii=False,
    )
    plan = generate_experiment_plan(rq, report, FakeLLM(f"```json\n{plan_json}\n```"))
    assert plan.source == "llm"
    assert plan.method == "spearman"


def test_plan_llm_bad_method_corrected_by_rule():
    df = build_df()
    report = profile_dataset(df)
    rq = ResearchQuestion(id="RQ1", question="q", variables=["attendance_rate", "final_score"])
    plan_json = json.dumps(
        {"hypothesis": "h", "method": "magic_regression", "h0": "H0", "h1": "H1"},
        ensure_ascii=False,
    )
    plan = generate_experiment_plan(rq, report, FakeLLM(plan_json))
    assert plan.source == "llm"
    assert plan.method == "pearson"
    assert "白名单" in plan.notes


def test_plan_llm_type_mismatch_corrected():
    """LLM 对两个数值变量给出 chi2：白名单内但类型组合不符 → 规则校正。"""
    df = build_df()
    report = profile_dataset(df)
    rq = ResearchQuestion(id="RQ1", question="q", variables=["attendance_rate", "final_score"])
    plan_json = json.dumps(
        {"hypothesis": "h", "method": "chi2", "h0": "H0", "h1": "H1"}, ensure_ascii=False
    )
    plan = generate_experiment_plan(rq, report, FakeLLM(plan_json))
    assert plan.source == "llm"
    assert plan.method == "pearson"
    assert "类型组合不符" in plan.notes


def test_suggest_method_skew_aware():
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {
            "x_skew": np.round(rng.lognormal(0.0, 1.0, 80), 2),
            "score": np.round(rng.normal(60, 10, 80), 1),
            "g": ["A"] * 40 + ["B"] * 40,
        }
    )
    report = profile_dataset(df)
    assert suggest_method(report, ["x_skew", "score"]) == "spearman"
    assert suggest_method(report, ["g", "x_skew"]) == "mannwhitney"
    normal_report = profile_dataset(build_df())
    assert suggest_method(normal_report, ["attendance_rate", "final_score"]) == "pearson"


def test_plan_falls_back_when_llm_breaks():
    df = build_df()
    report = profile_dataset(df)
    rq = ResearchQuestion(id="RQ1", question="q", variables=["gender", "final_score"])
    plan = generate_experiment_plan(rq, report, BrokenLLM())
    assert plan.source == "rule"
    assert plan.method == "welch_ttest"


def test_plan_rejects_unknown_variables():
    df = build_df()
    report = profile_dataset(df)
    rq = ResearchQuestion(id="RQ1", question="q", variables=["ghost"])
    with pytest.raises(ValueError):
        generate_experiment_plan(rq, report, None)


def test_rule_task_reorders_matching_questions():
    df = build_df()
    report = profile_dataset(df)
    qs = generate_research_questions_rule(
        df, report, max_questions=3, task="attendance_rate 如何影响 final_score"
    )
    assert qs
    assert "attendance_rate" in qs[0].variables


def test_rule_task_no_match_keeps_order():
    df = build_df()
    report = profile_dataset(df)
    with_task = generate_research_questions_rule(
        df, report, max_questions=3, task="完全无关的中文任务描述"
    )
    without_task = generate_research_questions_rule(df, report, max_questions=3)
    assert [q.question for q in with_task] == [q.question for q in without_task]


def test_llm_prompt_contains_task_constraint():
    df = build_df()
    report = profile_dataset(df)
    fake = FakeLLM("[]")
    with pytest.raises(ValueError):
        generate_research_questions(df, report, fake, task="出勤率对成绩的影响")
    assert fake.calls
    assert "研究任务" in fake.calls[0][1]
    assert "出勤率对成绩的影响" in fake.calls[0][1]
