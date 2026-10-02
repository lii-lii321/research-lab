"""实验计划生成：LLM 起草，变量类型规则校验与兜底——数字永远不由 LLM 决定方法学。"""
from __future__ import annotations

from models.schemas import ExperimentPlan, ProfileReport, ResearchQuestion
from services.executor import RUNNER_METHODS
from services.llm import LLMClient, LLMError
from services.power import required_n
from utils.textjson import extract_json_object

# 与执行层单一来源：白名单不再单独维护，杜绝两边漂移
ALLOWED_METHODS = RUNNER_METHODS

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


SKEW_ALERT_FOR_RANK = 2.0


def _abs_skew(col: object) -> float:
    numeric = getattr(col, "numeric", None)
    if numeric is None or numeric.skewness is None:
        return 0.0
    return abs(numeric.skewness)


def suggest_method(report: ProfileReport, variables: list[str]) -> str:
    """按变量类型与分布形态确定统计方法（兜底与校验共用）。

    数值列偏度 |skew| ≥ 2 时优先秩方法（spearman / mannwhitney / kruskal），
    消费 profiler 已产出的 SKEWED_DISTRIBUTION 信号而不是无视它。
    """
    cols = {c.name: c for c in report.columns}
    selected = [cols[v] for v in variables if v in cols]
    numeric = [c for c in selected if c.type == "numeric"]
    categorical = [c for c in selected if c.type in ("categorical", "boolean")]
    if len(numeric) >= 2 and not categorical:
        if len(numeric) == 2:
            skewed = any(_abs_skew(c) >= SKEW_ALERT_FOR_RANK for c in numeric)
            return "spearman" if skewed else "pearson"
        return "linear_regression"
    if len(numeric) == 1 and categorical:
        k = max(c.n_unique for c in categorical)
        if k == 2:
            skewed = any(_abs_skew(c) >= SKEW_ALERT_FOR_RANK for c in numeric)
            return "mannwhitney" if skewed else "welch_ttest"
        skewed = any(_abs_skew(c) >= SKEW_ALERT_FOR_RANK for c in numeric)
        return "kruskal" if skewed else "anova"
    return "chi2"


def method_fits_types(report: ProfileReport, variables: list[str], method: str) -> bool:
    """校验 LLM 给出的方法与变量类型组合是否匹配（白名单之外的恒 False）。"""
    cols = {c.name: c for c in report.columns}
    selected = [cols[v] for v in variables if v in cols]
    numeric = [c for c in selected if c.type == "numeric"]
    categorical = [c for c in selected if c.type in ("categorical", "boolean")]
    if method in ("pearson", "spearman", "paired_ttest"):
        return len(numeric) == 2 and not categorical
    if method in ("independent_ttest", "welch_ttest", "mannwhitney"):
        return len(numeric) == 1 and len(categorical) == 1 and categorical[0].n_unique == 2
    if method in ("anova", "kruskal"):
        return len(numeric) == 1 and len(categorical) == 1 and categorical[0].n_unique >= 2
    if method == "chi2":
        return not numeric and len(categorical) >= 2
    if method == "linear_regression":
        return len(numeric) >= 1 and len(numeric) + len(categorical) >= 2
    return False


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
    final: ExperimentPlan | None = None

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
            elif not method_fits_types(report, variables, plan.method):
                llm_method = plan.method
                plan.method = fallback
                plan.notes = "；".join(
                    filter(
                        None,
                        [
                            plan.notes,
                            f"{llm_method} 与变量类型组合不符，已按规则改用 {fallback}",
                        ],
                    )
                )
            final = plan
    if final is None:
        final = _rule_plan(rq, variables, fallback)

    final.required_n = required_n(final.method, final.alpha)
    if final.required_n:
        final.notes = "；".join(
            filter(
                None,
                [
                    final.notes,
                    f"功效分析：α={final.alpha}、power=0.8、中等效应假设下约需 {final.required_n} 例观测",
                ],
            )
        )
    return final
