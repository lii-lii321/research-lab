# -*- coding: utf-8 -*-
"""实验计划生成：LLM 起草，变量类型规则校验与兜底——数字永远不由 LLM 决定方法学。"""
from __future__ import annotations

from models.schemas import ExperimentPlan, ProfileReport, ResearchQuestion
from services.llm import LLMClient, LLMError
from utils.textjson import extract_json_object

ALLOWED_METHODS = {
    "pearson",
    "spearman",
    "independent_ttest",
    "welch_ttest",
    "mannwhitney",
    "paired_ttest",
    "anova",
    "kruskal",
    "chi2",
    "linear_regression",
    "logistic_regression",
}

_H0_H1 = {
    "pearson": ("两变量总体相关系数 ρ = 0", "ρ ≠ 0（存在线性相关）"),
    "spearman": ("两变量总体秩相关系数 ρs = 0", "ρs ≠ 0（存在单调相关）"),
    "welch_ttest": ("两组总体均值相等", "两组总体均值不相等"),
    "independent_ttest": ("两组总体均值相等（方差齐）", "两组总体均值不相等"),
    "mannwhitney": ("两组总体分布位置相同", "两组总体分布位置不同"),
    "paired_ttest": ("配对观测差值的总体均值为 0", "差值总体均值不为 0"),
    "anova": ("各组总体均值全部相等", "至少有一组总体均值不同"),
    "kruskal": ("各组总体分布位置相同", "至少有一组分布位置不同"),
    "chi2": ("两个类别变量相互独立", "两个类别变量不独立"),
    "linear_regression": ("所有回归系数均为 0（模型无解释力）", "至少一个回归系数非 0"),
    "logistic_regression": ("所有回归系数均为 0（模型无解释力）", "至少一个回归系数非 0"),
}

SYSTEM = "你是统计学实验设计助手。只输出 JSON 对象，不要输出任何其他内容。"

PROMPT_TEMPLATE = """研究问题：{question}
涉及变量：{variables}（类型：{types}）

请设计一个统计实验计划，输出 JSON 对象，字段：
- hypothesis: 假设陈述（一句话）
- method: 必须从以下方法中选一个：{methods}
- h0: 零假设
- h1: 备择假设
- alpha: 显著性水平（数字，通常 0.05）
- notes: 方法选择理由（一句话）
"""


def suggest_method(report: ProfileReport, variables: list[str]) -> str:
    """按变量类型确定统计方法的确定性规则（兜底与校验共用）。"""
    cols = {c.name: c for c in report.columns}
    selected = [cols[v] for v in variables if v in cols]
    numeric = [c for c in selected if c.type == "numeric"]
    categorical = [c for c in selected if c.type in ("categorical", "boolean")]
    if len(numeric) >= 2 and not categorical:
        return "pearson" if len(numeric) == 2 else "linear_regression"
    if len(numeric) == 1 and categorical:
        k = max(c.n_unique for c in categorical)
        return "welch_ttest" if k == 2 else "anova"
    return "chi2"


def _plan_from_dict(data: dict, rq: ResearchQuestion, variables: list[str]) -> ExperimentPlan:
    try:
        alpha = float(data.get("alpha", 0.05))
    except (TypeError, ValueError):
        alpha = 0.05
    if not 0 < alpha < 0.5:
        alpha = 0.05
    return ExperimentPlan(
        experiment_id=f"EXP-{rq.id}",
        question_id=rq.id,
        hypothesis=str(data.get("hypothesis") or rq.question),
        method=str(data.get("method", "")).strip().lower(),
        h0=str(data.get("h0", "")),
        h1=str(data.get("h1", "")),
        alpha=alpha,
        variables=variables,
        notes=str(data.get("notes", "")),
        source="llm",
    )


def _rule_plan(rq: ResearchQuestion, variables: list[str], method: str) -> ExperimentPlan:
    h0, h1 = _H0_H1.get(method, ("无效应", "存在效应"))
    return ExperimentPlan(
        experiment_id=f"EXP-{rq.id}",
        question_id=rq.id,
        hypothesis=rq.question,
        method=method,
        h0=h0,
        h1=h1,
        alpha=0.05,
        variables=variables,
        notes="由变量类型规则生成（LLM 未配置或输出不可用）",
        source="rule",
    )


def generate_experiment_plan(
    rq: ResearchQuestion, report: ProfileReport, client: LLMClient | None
) -> ExperimentPlan:
    known = {c.name for c in report.columns}
    variables = [v for v in rq.variables if v in known]
    if not variables:
        raise ValueError(f"研究问题 {rq.id} 的变量都不在数据集中")
    fallback = suggest_method(report, variables)

    if client is not None:
        type_map = {c.name: c.type for c in report.columns}
        prompt = PROMPT_TEMPLATE.format(
            question=rq.question,
            variables="、".join(variables),
            types="、".join(f"{v}({type_map.get(v, '?')})" for v in variables),
            methods=" / ".join(sorted(ALLOWED_METHODS)),
        )
        try:
            data = extract_json_object(client.chat(SYSTEM, prompt))
            plan = _plan_from_dict(data, rq, variables)
        except (LLMError, ValueError):
            plan = None
        if plan is not None:
            if plan.method not in ALLOWED_METHODS:
                plan.method = fallback
                plan.notes = "；".join(
                    filter(None, [plan.notes, f"方法超出白名单，已按变量类型规则改用 {fallback}"])
                )
            return plan
    return _rule_plan(rq, variables, fallback)
